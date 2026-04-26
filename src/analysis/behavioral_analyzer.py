"""
Behavioral Analyzer
==================
Analyze vessel behavior patterns: speed, direction changes, loitering, dark ships.
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import logging
import argparse
import yaml
from dataclasses import dataclass
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)


@dataclass
class BehavioralConfig:
    """Configuration for behavioral analysis"""
    loitering_speed_threshold: float = 3.0
    loitering_turn_threshold: float = 45.0
    dark_ship_threshold_hours: int = 6


def load_config(config_path: str = "./config/settings.yaml") -> BehavioralConfig:
    """Load behavioral analysis config"""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        ft = config.get("features", {})
        dark_threshold = ft.get("dark_ship_threshold_seconds", 21600) // 3600
        return BehavioralConfig(
            loitering_speed_threshold=ft.get("loitering_speed_threshold", 3.0),
            loitering_turn_threshold=ft.get("loitering_turn_threshold", 45.0),
            dark_ship_threshold_hours=dark_threshold,
        )
    except Exception:
        return BehavioralConfig()


class BehavioralAnalyzer:
    def __init__(
        self,
        output_dir: str = "./outputs/tables",
        config: Optional[BehavioralConfig] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or BehavioralConfig()

    def analyze_speed_behavior(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Analyze speed behavior patterns"""
        if "SOG" not in df.columns or "MMSI" not in df.columns:
            return pd.DataFrame()

        speed_stats = (
            df.groupby("MMSI")["SOG"]
            .agg(["mean", "std", "min", "max", "count"])
            .reset_index()
        )
        speed_stats.columns = [
            "MMSI",
            "mean_speed",
            "std_speed",
            "min_speed",
            "max_speed",
            "n_fixes",
        ]

        return speed_stats

    def analyze_direction_changes(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Analyze course over ground changes"""
        if "delta_cog" not in df.columns or "MMSI" not in df.columns:
            return pd.DataFrame()

        cog_stats = (
            df.groupby("MMSI")["delta_cog"]
            .agg(["mean", "std", "max"])
            .reset_index()
        )
        cog_stats.columns = ["MMSI", "mean_delta_cog", "std_delta_cog", "max_delta_cog"]

        return cog_stats

    def analyze_loitering(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Analyze loitering behavior"""
        if "loitering_flag" not in df.columns:
            return pd.DataFrame()

        loitering_vessels = df[df["loitering_flag"] == 1]["MMSI"].unique()

        if len(loitering_vessels) == 0:
            return pd.DataFrame()

        loitering_df = df[df["MMSI"].isin(loitering_vessels)]

        loitering_stats = (
            loitering_df.groupby("MMSI")
            .agg(
                loitering_events=("loitering_flag", "sum"),
                mean_speed=("SOG", "mean"),
            )
            .reset_index()
        )

        return loitering_stats

    def analyze_dark_ships(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """Analyze dark ship (AIS gap) patterns"""
        if "is_dark_ship" not in df.columns:
            return {}

        dark_ships = df[df["is_dark_ship"] == 1]

        if len(dark_ships) == 0:
            return {}

        dark_summary = (
            dark_ships.groupby("MMSI")
            .agg(
                dark_events=("is_dark_ship", "sum"),
                last_position=("BaseDateTime", "max"),
            )
            .reset_index()
        )

        dark_by_zone = (
            dark_ships.groupby("conflict_zone_name")
            .agg(
                n_dark_ships=("MMSI", "nunique"),
                n_events=("is_dark_ship", "sum"),
            )
            .reset_index()
        )
        dark_by_zone = dark_by_zone[dark_by_zone["conflict_zone_name"] != "none"]

        return {
            "dark_ship_summary": dark_summary,
            "dark_by_zone": dark_by_zone,
        }

    def analyze_route_entropy(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Analyze route unpredictability via entropy"""
        if "route_entropy" not in df.columns or "MMSI" not in df.columns:
            return pd.DataFrame()

        entropy_stats = (
            df.groupby("MMSI")["route_entropy"]
            .agg(["mean", "std", "max"])
            .reset_index()
        )
        entropy_stats.columns = ["MMSI", "mean_entropy", "std_entropy", "max_entropy"]

        return entropy_stats

    def analyze_turning_behavior(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Analyze turning/maneuvering patterns"""
        if "turning_rate" not in df.columns or "MMSI" not in df.columns:
            return pd.DataFrame()

        turning_stats = (
            df.groupby("MMSI")["turning_rate"]
            .agg(["mean", "std", "max"])
            .reset_index()
        )
        turning_stats.columns = ["MMSI", "mean_turn_rate", "std_turn_rate", "max_turn_rate"]

        return turning_stats

    def analyze_zigzag_patterns(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Analyze zigzag navigation patterns"""
        if "zig_zag_index" not in df.columns or "MMSI" not in df.columns:
            return pd.DataFrame()

        zigzag_vessels = df[df["zig_zag_index"] > 3]["MMSI"].unique()

        if len(zigzag_vessels) == 0:
            return pd.DataFrame()

        zigzag_stats = (
            df[df["MMSI"].isin(zigzag_vessels)]
            .groupby("MMSI")["zig_zag_index"]
            .max()
            .reset_index()
        )
        zigzag_stats.columns = ["MMSI", "zigzag_score"]

        return zigzag_stats

    def run_all(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Run all behavioral analyses"""
        logger.info("Running behavioral analysis...")

        results = {}

        speed_stats = self.analyze_speed_behavior(df)
        if not speed_stats.empty:
            results["speed_behavior"] = speed_stats
            speed_stats.to_csv(
                self.output_dir / "speed_behavior.csv",
                index=False,
            )
            logger.info(f"Saved: {self.output_dir / 'speed_behavior.csv'}")

        direction_stats = self.analyze_direction_changes(df)
        if not direction_stats.empty:
            results["direction_changes"] = direction_stats

        loitering_stats = self.analyze_loitering(df)
        if not loitering_stats.empty:
            results["loitering"] = loitering_stats
            loitering_stats.to_csv(
                self.output_dir / "loitering.csv",
                index=False,
            )
            logger.info(f"Saved: {self.output_dir / 'loitering.csv'}")

        dark_ship_results = self.analyze_dark_ships(df)
        if dark_ship_results:
            for key, val in dark_ship_results.items():
                if not val.empty:
                    results[key] = val
                    val.to_csv(
                        self.output_dir / f"{key}.csv",
                        index=False,
                    )
                    logger.info(f"Saved: {self.output_dir / f'{key}.csv'}")

        zigzag_stats = self.analyze_zigzag_patterns(df)
        if not zigzag_stats.empty:
            results["zigzag_patterns"] = zigzag_stats

        logger.info("Behavioral analysis complete!")
        return results


def main():
    parser = argparse.ArgumentParser(description="Behavioral Analyzer")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="./outputs/tables")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df):,} records")

    config = load_config()
    analyzer = BehavioralAnalyzer(args.output_dir, config)
    analyzer.run_all(df)


if __name__ == "__main__":
    main()