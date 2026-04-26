"""
Traffic Analyzer
================
Analyze maritime traffic volume, density, and flow patterns.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
import yaml
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)


@dataclass
class TrafficConfig:
    """Configuration for traffic analysis"""
    time_bucket: str = "6h"
    grid_resolution: float = 0.5


def load_config(config_path: str = "./config/settings.yaml") -> TrafficConfig:
    """Load traffic analysis config"""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        ft = config.get("features", {})
        tb = config.get("analysis", {}).get("time_bucket", "6h")
        gr = ft.get("grid_resolution", 0.5)
        return TrafficConfig(time_bucket=tb, grid_resolution=gr)
    except Exception:
        return TrafficConfig()


class TrafficAnalyzer:
    def __init__(
        self,
        output_dir: str = "./outputs/tables",
        config: Optional[TrafficConfig] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or TrafficConfig()

    def compute_hourly_traffic(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute hourly traffic counts"""
        if "BaseDateTime" not in df.columns:
            return pd.DataFrame()

        hourly = (
            df.groupby(df["BaseDateTime"].dt.hour)["MMSI"]
            .nunique()
            .reset_index()
        )
        hourly.columns = ["hour", "vessel_count"]
        return hourly

    def compute_daily_traffic(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute daily traffic counts"""
        if "BaseDateTime" not in df.columns:
            return pd.DataFrame()

        df = df.copy()
        df["date"] = df["BaseDateTime"].dt.date

        daily = (
            df.groupby("date")["MMSI"]
            .nunique()
            .reset_index()
        )
        daily.columns = ["date", "vessel_count"]
        return daily

    def compute_traffic_by_zone(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute traffic counts by conflict zone"""
        if "conflict_zone_name" not in df.columns:
            return pd.DataFrame()

        zones = df["conflict_zone_name"].unique()
        zones = [z for z in zones if z != "none"]

        if not zones:
            return pd.DataFrame()

        zone_traffic = (
            df[df["conflict_zone_name"].isin(zones)]
            .groupby("conflict_zone_name")["MMSI"]
            .nunique()
            .reset_index()
        )
        zone_traffic.columns = ["zone", "vessel_count"]

        return zone_traffic

    def compute_grid_density(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute traffic density by grid cell"""
        if "grid_cell" not in df.columns:
            return pd.DataFrame()

        grid_density = (
            df.groupby("grid_cell")
            .agg(
                vessel_count=("MMSI", "nunique"),
                record_count=("MMSI", "count"),
            )
            .reset_index()
        )
        grid_density = grid_density.sort_values(
            "vessel_count", ascending=False
        ).head(100)

        return grid_density

    def compute_traffic_trends(
        self,
        df: pd.DataFrame,
        freq: str = "D",
    ) -> pd.DataFrame:
        """Compute traffic trends over time"""
        if "BaseDateTime" not in df.columns:
            return pd.DataFrame()

        df = df.copy()
        df["period"] = df["BaseDateTime"].dt.floor(freq)

        trends = (
            df.groupby("period")["MMSI"]
            .nunique()
            .reset_index()
        )
        trends.columns = ["period", "vessel_count"]

        if len(trends) > 7:
            trends["trend"] = (
                trends["vessel_count"]
                .rolling(7, min_periods=3)
                .mean()
            )

        return trends

    def detect_traffic_anomalies(
        self,
        df: pd.DataFrame,
        threshold_std: float = 2.0,
    ) -> pd.DataFrame:
        """Detect traffic volume anomalies using z-score"""
        daily = self.compute_daily_traffic(df)

        if daily.empty or len(daily) < 14:
            return pd.DataFrame()

        mean = daily["vessel_count"].mean()
        std = daily["vessel_count"].std()

        daily["z_score"] = (daily["vessel_count"] - mean) / (std + 1e-6)
        daily["is_anomaly"] = daily["z_score"].abs() > threshold_std

        return daily[daily["is_anomaly"]]

    def run_all(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Run all traffic analyses"""
        logger.info("Running traffic analysis...")

        results = {}

        hourly = self.compute_hourly_traffic(df)
        if not hourly.empty:
            results["hourly_traffic"] = hourly

        daily = self.compute_daily_traffic(df)
        if not daily.empty:
            results["daily_traffic"] = daily

        zone_traffic = self.compute_traffic_by_zone(df)
        if not zone_traffic.empty:
            results["zone_traffic"] = zone_traffic
            zone_traffic.to_csv(
                self.output_dir / "zone_traffic.csv",
                index=False,
            )
            logger.info(f"Saved: {self.output_dir / 'zone_traffic.csv'}")

        grid_density = self.compute_grid_density(df)
        if not grid_density.empty:
            results["grid_density"] = grid_density
            grid_density.to_csv(
                self.output_dir / "grid_density.csv",
                index=False,
            )
            logger.info(f"Saved: {self.output_dir / 'grid_density.csv'}")

        anomalies = self.detect_traffic_anomalies(df)
        if not anomalies.empty:
            results["traffic_anomalies"] = anomalies
            anomalies.to_csv(
                self.output_dir / "traffic_anomalies.csv",
                index=False,
            )
            logger.info(f"Saved: {self.output_dir / 'traffic_anomalies.csv'}")

        logger.info("Traffic analysis complete!")
        return results


def main():
    parser = argparse.ArgumentParser(description="Traffic Analyzer")
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
    analyzer = TrafficAnalyzer(args.output_dir, config)
    analyzer.run_all(df)


if __name__ == "__main__":
    main()