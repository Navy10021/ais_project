"""
AIS Raw Data Cleaner
===================
Physical validity filtering + per-column missing value imputation.

Processing order:
  1. MMSI validity       — 9-digit civilian range; flag special MMSI types
  2. Coordinate validity — range check + AIS sentinel values (91 / 181)
  3. Kinematic sentinels — SOG=102.3, COG=360, Heading=511 → NaN
  4. Timestamps          — UTC parse, future / pre-2010 record removal
  5. Duplicate removal   — MMSI + BaseDateTime
  6. Missing imputation  — per-column strategy (see impute_missing)

Coding Conventions (CLAUDE.md):
  - Type hints on all public functions
  - Logging via logger (no bare print)
  - No hardcoded paths (use config/settings.yaml)
  - Reproducibility with SEED
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
import yaml
from dataclasses import dataclass
from typing import Dict, Any

logger = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)

import random
random.seed(SEED)


@dataclass
class CleanerConfig:
    """Configuration for AISCleaner"""
    min_mmsi: int = 200_000_000
    max_mmsi: int = 799_999_999
    invalid_lat: float = 91.0
    invalid_lon: float = 181.0
    invalid_sog: float = 102.3
    invalid_cog: float = 360.0
    invalid_heading: int = 511
    invalid_imo: str = "IMO0000000"
    timestamp_cutoff: str = "2010-01-01"


def load_config(config_path: str = "./config/settings.yaml") -> CleanerConfig:
    """Load configuration from YAML file"""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return CleanerConfig(
            min_mmsi=config.get("preprocessing", {}).get("min_mmsi", 200_000_000),
            max_mmsi=config.get("preprocessing", {}).get("max_mmsi", 799_999_999),
            invalid_lat=config.get("preprocessing", {}).get("invalid_lat", 91.0),
            invalid_lon=config.get("preprocessing", {}).get("invalid_lon", 181.0),
            invalid_sog=config.get("preprocessing", {}).get("invalid_sog", 102.3),
            invalid_cog=config.get("preprocessing", {}).get("invalid_cog", 360.0),
            invalid_heading=config.get("preprocessing", {}).get("invalid_heading", 511),
            timestamp_cutoff=config.get("preprocessing", {}).get("timestamp_cutoff", "2010-01-01"),
        )
    except Exception:
        logger.warning(f"Could not load config from {config_path}, using defaults")
        return CleanerConfig()


class AISCleaner:
    INVALID_LAT = 91.0
    INVALID_LON = 181.0
    INVALID_SOG = 102.3
    INVALID_COG = 360.0
    INVALID_HEADING = 511
    INVALID_IMO = "IMO0000000"

    SPECIAL_MMSI = {
        "coastal_station": (0, 99_999_999),
        "group_ship": (970_000_000, 979_999_999),
        "sar_aircraft": (111_000_000, 111_999_999),
        "mob_device": (972_000_000, 972_999_999),
        "aton": (990_000_000, 999_999_999),
    }

    DTYPE_MAP = {
        "MMSI": "int64",
        "LAT": "float32",
        "LON": "float32",
        "SOG": "float32",
        "COG": "float32",
        "Heading": "float32",
        "VesselType": "float32",
        "Status": "float32",
        "Length": "float32",
        "Width": "float32",
        "Draft": "float32",
        "Cargo": "float32",
        "TransceiverClass": "category",
    }

    def __init__(self, input_path: str, output_path: str, config: CleanerConfig = None):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.report: dict = {}
        self.config = config or CleanerConfig()
        
        # Use config values as defaults
        self.INVALID_LAT = self.config.invalid_lat
        self.INVALID_LON = self.config.invalid_lon
        self.INVALID_SOG = self.config.invalid_sog
        self.INVALID_COG = self.config.invalid_cog
        self.INVALID_HEADING = self.config.invalid_heading
        self.INVALID_IMO = self.config.invalid_imo

    def load(self) -> pd.DataFrame:
        if self.input_path.suffix == ".parquet":
            return pd.read_parquet(self.input_path)
        return pd.read_csv(
            self.input_path,
            dtype=self.DTYPE_MAP,
            parse_dates=["BaseDateTime"],
            low_memory=False,
        )

    def clean_mmsi(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        df["mmsi_special_type"] = None
        for name, (lo, hi) in self.SPECIAL_MMSI.items():
            mask = df["MMSI"].between(lo, hi)
            df.loc[mask, "mmsi_special_type"] = name
        
        invalid_mask = ~df["MMSI"].between(200_000_000, 799_999_999)
        self.report["mmsi_removed"] = invalid_mask.sum()
        
        return df[~invalid_mask].copy()

    def clean_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        df = df[
            df["LAT"].between(-90.0, 90.0)
            & df["LON"].between(-180.0, 180.0)
            & (df["LAT"] != self.INVALID_LAT)
            & (df["LON"] != self.INVALID_LON)
            & df["LAT"].notna()
            & df["LON"].notna()
        ].copy()
        self.report["coord_removed"] = n - len(df)
        logger.info(f"Coordinate filter: removed {n - len(df):,} records")
        return df

    def clean_kinematics(self, df: pd.DataFrame) -> pd.DataFrame:
        df["SOG"] = df["SOG"].where(df["SOG"] < self.INVALID_SOG, np.nan)
        df["COG"] = df["COG"].where(df["COG"] < self.INVALID_COG, np.nan)
        df["Heading"] = df["Heading"].where(df["Heading"] < self.INVALID_HEADING, np.nan)
        df["sog_implausible_flag"] = (df["SOG"] > 50.0).astype("int8")
        return df

    def clean_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"], utc=True, errors="coerce")
        now = pd.Timestamp.now(tz="UTC")
        cutoff = pd.Timestamp("2010-01-01", tz="UTC")
        df = df[df["BaseDateTime"].between(cutoff, now)].copy()
        return df.sort_values(["MMSI", "BaseDateTime"]).reset_index(drop=True)

    def clean_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        df = df.drop_duplicates(subset=["MMSI", "BaseDateTime"], keep="first")
        self.report["duplicates_removed"] = n - len(df)
        if n - len(df) > 0:
            logger.info(f"Duplicate removal: removed {n - len(df):,} records")
        return df

    def impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ["VesselName", "IMO", "CallSign"]:
            df[col] = df[col].fillna("UNKNOWN").replace("", "UNKNOWN")
        df.loc[df["IMO"] == self.INVALID_IMO, "IMO"] = "UNKNOWN"

        type_mode = df.groupby("MMSI")["VesselType"].transform(
            lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else 0)
        )
        df["VesselType"] = df["VesselType"].fillna(type_mode)

        for col in ["Length", "Width", "Draft"]:
            df[col] = df.groupby("MMSI")[col].transform(lambda x: x.fillna(x.median()))
            df[col] = df.groupby("VesselType")[col].transform("median")
            df[col] = df[col].fillna(df[col].median())

        df["Status"] = df["Status"].fillna(0).astype("int8")
        return df

    def run(self) -> pd.DataFrame:
        logger.info(f"Loading raw AIS data from {self.input_path}...")
        df = self.load()
        logger.info(f"Raw records: {len(df):,}")

        df = self.clean_mmsi(df)
        df = self.clean_coordinates(df)
        df = self.clean_kinematics(df)
        df = self.clean_timestamps(df)
        df = self.clean_duplicates(df)
        df = self.impute_missing(df)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.output_path, index=False, compression="snappy")
        logger.info(f"Clean records: {len(df):,} → {self.output_path}")
        logger.info(f"Report: {self.report}")
        return df


def main():
    parser = argparse.ArgumentParser(description="AIS Raw Data Cleaner")
    parser.add_argument("--input", required=True, help="Input CSV/Parquet path")
    parser.add_argument("--output", required=True, help="Output Parquet path")
    parser.add_argument("--config", default="./config/settings.yaml", help="Config YAML path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config = load_config(args.config)
    cleaner = AISCleaner(args.input, args.output, config)
    cleaner.run()
    cleaner.run()


if __name__ == "__main__":
    main()