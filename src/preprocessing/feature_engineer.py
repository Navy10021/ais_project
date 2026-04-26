"""
AIS Feature Engineer
====================
Five feature categories for conflict detection and prediction.

A. Kinematic    — motion & maneuver characteristics
B. Geospatial  — spatial context & chokepoint proximity
C. Behavioral  — rolling-window irregularity metrics
D. Aggregation — grid-cell × time bucket traffic statistics
E. Labels       — binary conflict label + regression target

Coding Conventions:
  - Type hints
  - Logging via logger
  - Config via YAML
  - Reproducibility with SEED
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import logging
import argparse
import yaml
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)


@dataclass
class FeatureConfig:
    grid_resolution: float = 0.5
    time_bucket: str = "6h"
    rolling_window: str = "12h"
    dark_ship_threshold_seconds: int = 21600


def load_config(config_path: str = "./config/settings.yaml") -> FeatureConfig:
    """Load feature configuration from YAML"""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        ft = config.get("features", {})
        return FeatureConfig(
            grid_resolution=ft.get("grid_resolution", 0.5),
            time_bucket=ft.get("time_bucket", "6h"),
            rolling_window=ft.get("rolling_window", "12h"),
            dark_ship_threshold_seconds=ft.get(
                "dark_ship_threshold_seconds", 21600
            ),
        )
    except Exception:
        logger.warning(
            f"Could not load config from {config_path}, using defaults"
        )
        return FeatureConfig()


class AISFeatureEngineer:
    CONFLICT_ZONES = {
        "black_sea": {"bbox": [27.0, 40.5, 41.0, 46.8], "conflict": "ukraine_war"},
        "azov_sea": {"bbox": [33.5, 45.0, 39.5, 47.5], "conflict": "ukraine_war"},
        "kerch_strait": {"bbox": [36.4, 45.1, 36.8, 45.5], "conflict": "ukraine_war"},
        "red_sea": {"bbox": [32.0, 12.0, 43.5, 30.0], "conflict": "houthi_crisis"},
        "bab_el_mandeb": {"bbox": [43.0, 11.5, 45.0, 12.5], "conflict": "houthi_crisis"},
        "taiwan_strait": {"bbox": [119.0, 22.0, 122.0, 26.0], "conflict": "taiwan_tension"},
        "south_china_sea": {"bbox": [109.0, 3.0, 121.0, 22.0], "conflict": "scs_dispute"},
        "strait_hormuz": {"bbox": [56.0, 25.5, 59.5, 27.0], "conflict": "iran_tension"},
    }

    CHOKEPOINTS = {
        "hormuz": (56.5, 26.5),
        "malacca": (103.8, 1.2),
        "bab_mandeb": (43.4, 12.5),
        "suez": (32.5, 30.7),
        "panama": (-79.9, 9.0),
        "gibraltar": (-5.4, 36.0),
        "dover": (1.3, 51.0),
    }

    VESSEL_TYPE_MILITARY = 35
    VESSEL_TYPE_SAR = 51
    VESSEL_TYPE_CARGO = list(range(70, 80))
    VESSEL_TYPE_TANKER = list(range(80, 90))

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or load_config()

    def add_kinematic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["MMSI", "BaseDateTime"]).copy()

        df["speed_category"] = pd.cut(
            df["SOG"],
            bins=[-0.1, 0.5, 3.0, 8.0, 15.0, 102.2],
            labels=["anchored", "drifting", "slow", "cruising", "fast"],
        )

        grp = df.groupby("MMSI", sort=False)

        df["delta_sog"] = grp["SOG"].diff().abs()

        cog_diff = grp["COG"].diff().abs()
        df["delta_cog"] = np.minimum(cog_diff, 360.0 - cog_diff)

        df["time_diff_sec"] = grp["BaseDateTime"].diff().dt.total_seconds()
        df["turning_rate"] = df["delta_cog"] / df["time_diff_sec"].replace(0, np.nan)

        threshold = self.config.dark_ship_threshold_seconds
        df["is_dark_ship"] = (
            df["time_diff_sec"] > threshold
        ).astype("int8")

        df["moored_vs_drifting"] = (
            (df["SOG"] < 0.3) & df["Heading"].notna()
        ).astype("int8")

        df["sog_z_score"] = grp["SOG"].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-6)
        )

        return df

    def add_geospatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        res = self.config.grid_resolution

        df["grid_lat"] = (df["LAT"] // res) * res
        df["grid_lon"] = (df["LON"] // res) * res
        df["grid_cell"] = (
            df["grid_lat"].astype(str) + "_" + df["grid_lon"].astype(str)
        )

        in_zone = np.zeros(len(df), dtype=bool)
        zone_names = np.full(len(df), "none", dtype=object)

        lon_vals = df["LON"].values
        lat_vals = df["LAT"].values

        for zone, info in self.CONFLICT_ZONES.items():
            b = info["bbox"]
            mask = (
                (lon_vals >= b[0]) & (lon_vals <= b[2]) &
                (lat_vals >= b[1]) & (lat_vals <= b[3])
            )
            in_zone[mask] = True
            zone_names[mask] = zone

        df["in_conflict_zone"] = in_zone
        df["conflict_zone_name"] = zone_names

        for name, (cp_lon, cp_lat) in self.CHOKEPOINTS.items():
            df[f"dist_{name}_km"] = self._haversine(
                df["LAT"], df["LON"], cp_lat, cp_lon
            )

        return df

    def add_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["MMSI", "BaseDateTime"]).copy()
        df = df.set_index("BaseDateTime")

        roll_window = self.config.rolling_window
        roll = df.groupby("MMSI")["SOG"].rolling(
            roll_window, min_periods=3
        )

        df["rolling_sog_mean_12h"] = roll.mean().reset_index(level=0, drop=True)
        df["rolling_sog_std_12h"] = roll.std().reset_index(level=0, drop=True)
        df = df.reset_index()

        df["_date"] = df["BaseDateTime"].dt.date

        def _entropy(s: pd.Series) -> float:
            valid = s.dropna()
            if len(valid) < 2:
                return np.nan
            bins = pd.cut(valid, bins=36, labels=False)
            if len(bins) < 2:
                return np.nan
            p = bins.value_counts(normalize=True) + 1e-10
            return float(stats.entropy(p))

        ent = (
            df.groupby(["MMSI", "_date"])["COG"]
            .apply(_entropy)
            .reset_index()
            .rename(columns={"COG": "route_entropy"})
        )
        df = df.merge(ent, on=["MMSI", "_date"], how="left").drop(columns="_date")

        df["loitering_flag"] = (
            (df["SOG"] < 3.0) &
            (df["delta_cog"] > 45.0) &
            df["in_conflict_zone"]
        ).astype("int8")

        df["zig_zag_index"] = (
            df.groupby("MMSI")["delta_cog"]
            .transform(
                lambda x: (
                    np.sign(x.fillna(0).values) != np.sign(x.shift().fillna(0).values)
                ).astype(int)
            )
            .groupby(df["MMSI"])
            .rolling(10, min_periods=3)
            .sum()
            .reset_index(level=0, drop=True)
        )

        return df

    def add_temporal_aggregation(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        time_bucket = self.config.time_bucket
        df["time_bucket"] = df["BaseDateTime"].dt.floor(time_bucket)

        agg_dict = {
            "traffic_count": ("MMSI", "nunique"),
            "dark_ship_count": ("is_dark_ship", "sum"),
            "mean_sog": ("SOG", "mean"),
            "std_sog": ("SOG", "std"),
            "loitering_density": ("loitering_flag", "sum"),
            "military_count": ("VesselType", lambda x: (x == self.VESSEL_TYPE_MILITARY).sum()),
            "cargo_count": ("VesselType", lambda x: x.isin(self.VESSEL_TYPE_CARGO).sum()),
            "tanker_count": ("VesselType", lambda x: x.isin(self.VESSEL_TYPE_TANKER).sum()),
            "sar_count": ("VesselType", lambda x: (x == self.VESSEL_TYPE_SAR).sum()),
        }

        agg = df.groupby(["grid_cell", "time_bucket"]).agg(**agg_dict).reset_index()

        denom = agg["traffic_count"].clip(lower=1).values
        agg["dark_ship_ratio"] = agg["dark_ship_count"].values / denom
        agg["military_ratio"] = agg["military_count"].values / denom
        agg["tanker_ratio"] = agg["tanker_count"].values / denom

        return df, agg

    def add_conflict_labels(
        self, df: pd.DataFrame, conflict_events_path: str
    ) -> pd.DataFrame:
        df["conflict_label"] = 0
        df["days_to_conflict"] = np.nan
        df["conflict_intensity"] = 0.0

        if not conflict_events_path:
            logger.info("No conflict events path provided, using default labels")
            return df

        if not Path(conflict_events_path).exists():
            logger.warning(f"Conflict events file not found: {conflict_events_path}")
            return df

        try:
            events = pd.read_csv(conflict_events_path, parse_dates=["event_date"])

            for _, ev in events.iterrows():
                zone = ev.get("zone", "")
                if not zone or zone == "none":
                    continue

                zone_mask = df["conflict_zone_name"] == zone
                if not zone_mask.any():
                    continue

                event_date = pd.Timestamp(ev["event_date"])
                if event_date.tzinfo is None:
                    event_date = event_date.tz_localize("UTC")

                day_diff = (event_date - df["BaseDateTime"]).dt.days

                match = zone_mask & (day_diff >= -7) & (day_diff <= 30)
                df.loc[match, "conflict_label"] = 1
                df.loc[match, "days_to_conflict"] = day_diff[match]
                df.loc[match, "conflict_intensity"] = ev.get("fatalities", 0)

        except Exception as e:
            logger.warning(f"Could not load conflict events: {e}")

        return df

    @staticmethod
    def _haversine(lat1, lon1, lat2: float, lon2: float) -> pd.Series:
        R = 6_371.0
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)

        a = (
            np.sin(dlat / 2) ** 2 +
            np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        )
        return R * 2 * np.arcsin(np.sqrt(a))

    def run(
        self, df: pd.DataFrame, conflict_events_path: Optional[str] = None
    ) -> pd.DataFrame:
        logger.info("A. Kinematic features...")
        df = self.add_kinematic_features(df)

        logger.info("B. Geospatial features...")
        df = self.add_geospatial_features(df)

        logger.info("C. Behavioral features...")
        df = self.add_behavioral_features(df)

        logger.info("D. Temporal aggregation...")
        df, agg = self.add_temporal_aggregation(df)
        self.agg_df = agg

        logger.info("E. Conflict labels...")
        df = self.add_conflict_labels(df, conflict_events_path)

        return df


def main():
    parser = argparse.ArgumentParser(description="AIS Feature Engineer")
    parser.add_argument("--input", required=True, help="Input Parquet path")
    parser.add_argument("--output", required=True, help="Output Parquet path")
    parser.add_argument(
        "--conflict-events",
        default=None,
        help="Conflict events CSV path",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df):,} records")

    config = load_config()
    engineer = AISFeatureEngineer(config)
    df = engineer.run(df, args.conflict_events)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False, compression="snappy")
    logger.info(f"Features saved: {len(df):,} → {args.output}")


if __name__ == "__main__":
    main()
