"""
AIS Raw Data Cleaner
====================
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
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)


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
        pp = config.get("preprocessing", {})
        return CleanerConfig(
            min_mmsi=pp.get("min_mmsi", 200_000_000),
            max_mmsi=pp.get("max_mmsi", 799_999_999),
            invalid_lat=pp.get("invalid_lat", 91.0),
            invalid_lon=pp.get("invalid_lon", 181.0),
            invalid_sog=pp.get("invalid_sog", 102.3),
            invalid_cog=pp.get("invalid_cog", 360.0),
            invalid_heading=pp.get("invalid_heading", 511),
            timestamp_cutoff=pp.get("timestamp_cutoff", "2010-01-01"),
        )
    except Exception:
        logger.warning(f"Could not load config from {config_path}, using defaults")
        return CleanerConfig()


class AISCleaner:
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

    SPECIAL_MMSI_RANGES = [
        ("coastal_station", 0, 99_999_999),
        ("group_ship", 970_000_000, 979_999_999),
        ("sar_aircraft", 111_000_000, 111_999_999),
        ("mob_device", 972_000_000, 972_999_999),
        ("aton", 990_000_000, 999_999_999),
    ]

    def __init__(
        self,
        input_path: str,
        output_path: str,
        config: Optional[CleanerConfig] = None,
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.report: Dict[str, int] = {}
        self.config = config or CleanerConfig()

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

        mmsi = df["MMSI"].values
        special_types = np.empty(len(mmsi), dtype=object)
        special_types[:] = None

        for name, lo, hi in self.SPECIAL_MMSI_RANGES:
            mask = (mmsi >= lo) & (mmsi <= hi)
            special_types[mask] = name

        df["mmsi_special_type"] = special_types

        valid_mask = (mmsi >= self.config.min_mmsi) & (mmsi <= self.config.max_mmsi)
        df = df[valid_mask].copy()
        self.report["mmsi_removed"] = n - len(df)

        return df

    def clean_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)

        lat = df["LAT"].values
        lon = df["LON"].values
        valid = (
            (lat >= -90.0) & (lat <= 90.0) &
            (lon >= -180.0) & (lon <= 180.0) &
            (lat != self.config.invalid_lat) &
            (lon != self.config.invalid_lon) &
            (~np.isnan(lat)) &
            (~np.isnan(lon))
        )

        df = df[valid].copy()
        self.report["coord_removed"] = n - len(df)
        logger.info(f"Coordinate filter: removed {n - len(df):,} records")
        return df

    def clean_kinematics(self, df: pd.DataFrame) -> pd.DataFrame:
        sog = df["SOG"].values
        cog = df["COG"].values
        heading = df["Heading"].values

        df["SOG"] = np.where(sog < self.config.invalid_sog, sog, np.nan)
        df["COG"] = np.where(cog < self.config.invalid_cog, cog, np.nan)
        df["Heading"] = np.where(
            heading < self.config.invalid_heading, heading, np.nan
        )
        df["sog_implausible_flag"] = (df["SOG"] > 50.0).astype("int8")
        return df

    def clean_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        df["BaseDateTime"] = pd.to_datetime(
            df["BaseDateTime"], utc=True, errors="coerce"
        )
        now = pd.Timestamp.now(tz="UTC")
        cutoff = pd.Timestamp(self.config.timestamp_cutoff, tz="UTC")

        valid = df["BaseDateTime"].between(cutoff, now)
        df = df[valid].copy()

        df.sort_values(["MMSI", "BaseDateTime"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def clean_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        df = df.drop_duplicates(
            subset=["MMSI", "BaseDateTime"], keep="first"
        )
        self.report["duplicates_removed"] = n - len(df)
        if n - len(df) > 0:
            logger.info(f"Duplicate removal: removed {n - len(df):,} records")
        return df

    def impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ["VesselName", "IMO", "CallSign"]:
            df[col] = df[col].fillna("UNKNOWN").replace("", "UNKNOWN")

        df.loc[df["IMO"] == self.config.invalid_imo, "IMO"] = "UNKNOWN"

        df["VesselType"] = df.groupby("MMSI")["VesselType"].transform(
            lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else 0)
        )

        for col in ["Length", "Width", "Draft"]:
            df[col] = df.groupby("MMSI")[col].transform(
                lambda x: x.fillna(x.median())
            )
            df[col] = df.groupby("VesselType")[col].transform(
                lambda x: x.fillna(x.median())
            )
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
    parser.add_argument(
        "--config",
        default="./config/settings.yaml",
        help="Config YAML path",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config = load_config(args.config)
    cleaner = AISCleaner(args.input, args.output, config)
    cleaner.run()


if __name__ == "__main__":
    main()