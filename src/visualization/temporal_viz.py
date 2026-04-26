"""
Temporal Visualization Module
=====================
Time-series visualizations for AIS data with conflict event overlays.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import argparse
from typing import Optional, List, Dict

from .base import VizConfig, load_viz_config, setup_matplotlib, ensure_output_dir

logger = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)


class ConflictEvent:
    """Represents a conflict event for overlay plotting"""

    def __init__(self, date: str, name: str, zone: str = None):
        self.date = pd.to_datetime(date)
        self.name = name
        self.zone = zone


class TemporalVisualizer:
    CONFLICT_EVENTS: Dict[str, ConflictEvent] = {
        "ukraine_war": ConflictEvent("2022-02-24", "Russia-Ukraine War"),
        "houthi_crisis": ConflictEvent("2023-11-19", "Red Sea Crisis"),
        "taiwan_drill": ConflictEvent("2022-08-04", "Taiwan Strait Drill"),
    }

    def __init__(
        self,
        output_dir: str = "./outputs/figures/temporal",
        config: Optional[VizConfig] = None,
    ):
        self.output_dir = ensure_output_dir(output_dir)
        self.config = config or load_viz_config()
        setup_matplotlib(self.config)

    def plot_traffic_volume_timeseries(
        self,
        df: pd.DataFrame,
        freq: str = "D",
        conflict_overlay: bool = True,
    ) -> None:
        """Plot daily traffic volume with optional conflict event markers"""
        df = df.copy()
        df["date"] = df["BaseDateTime"].dt.floor(freq)
        daily = df.groupby("date")["MMSI"].nunique()

        fig, ax = plt.subplots(figsize=(12, 5))

        daily.plot(ax=ax, marker="o", color="steelblue", linewidth=1.5)
        ax.fill_between(daily.index, daily.values, alpha=0.3, color="steelblue")

        if conflict_overlay:
            for key, event in self.CONFLICT_EVENTS.items():
                daily_min = daily.index.min()
                daily_max = daily.index.max()
                # Make event date timezone-aware if needed
                event_date = event.date
                if daily_min.tzinfo is not None and event_date.tzinfo is None:
                    event_date = event_date.tz_localize('UTC')
                elif daily_min.tzinfo is None and event_date.tzinfo is not None:
                    event_date = event_date.tz_convert(None)
                if event_date >= daily_min and event_date <= daily_max:
                    ax.axvline(
                        event.date,
                        color="red",
                        linestyle="--",
                        alpha=0.7,
                        linewidth=1.5,
                    )
                    ax.text(
                        event.date,
                        ax.get_ylim()[1] * 0.95,
                        event.name,
                        rotation=90,
                        va="top",
                        fontsize=8,
                        color="red",
                    )

        ax.set_xlabel("Date")
        ax.set_ylabel("Unique Vessels")
        ax.set_title("Daily Traffic Volume")
        ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        path = self.output_dir / "traffic_volume.png"
        plt.savefig(path, dpi=self.config.dpi)
        plt.close()
        logger.info(f"Saved: {path}")

    def plot_vessel_type_composition(
        self,
        df: pd.DataFrame,
        zone: str = None,
        top_n: int = 10,
    ) -> None:
        """Plot stacked area chart of vessel type composition"""
        if zone:
            df = df[df["conflict_zone_name"] == zone]

        df = df.copy()
        df["date"] = df["BaseDateTime"].dt.date

        type_by_date = (
            df.groupby(["date", "VesselType"])["MMSI"]
            .nunique()
            .unstack(fill_value=0)
        )

        type_by_date = type_by_date.loc[:, type_by_date.sum().nlargest(top_n).index]

        fig, ax = plt.subplots(figsize=(12, 6))

        type_by_date.plot.area(ax=ax, alpha=0.7, colormap="tab10")

        ax.set_xlabel("Date")
        ax.set_ylabel("Unique Vessels")
        ax.set_title(
            "Vessel Type Composition Over Time"
            + (f" - {zone}" if zone else "")
        )
        ax.legend(
            title="Vessel Type",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )

        plt.tight_layout()
        path = self.output_dir / "vessel_type_composition.png"
        plt.savefig(path, dpi=self.config.dpi)
        plt.close()
        logger.info(f"Saved: {path}")

    def plot_speed_distribution_timeseries(
        self,
        df: pd.DataFrame,
    ) -> None:
        """Plot speed distribution changes over time"""
        df = df.copy()
        df["month"] = df["BaseDateTime"].dt.month
        df["year_month"] = df["BaseDateTime"].dt.to_period("M")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        daily_mean = df.groupby(df["BaseDateTime"].dt.date)["SOG"].mean()
        daily_mean.plot(ax=axes[0], color="teal", linewidth=1.5)

        axes[0].set_xlabel("Date")
        axes[0].set_ylabel("Mean Speed (knots)")
        axes[0].set_title("Mean Speed Over Time")
        axes[0].tick_params(axis="x", rotation=45)

        monthly_speeds = df.groupby("year_month")["SOG"].apply(list)

        if len(monthly_speeds) > 0:
            bp = axes[1].boxplot(
                monthly_speeds.values,
                labels=[str(m) for m in monthly_speeds.index],
                patch_artist=True,
            )

            for patch in bp["boxes"]:
                patch.set_facecolor("lightblue")

            axes[1].set_xlabel("Month")
            axes[1].set_ylabel("Speed (knots)")
            axes[1].set_title("Monthly Speed Distribution")
            axes[1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        path = self.output_dir / "speed_timeseries.png"
        plt.savefig(path, dpi=self.config.dpi)
        plt.close()
        logger.info(f"Saved: {path}")

    def plot_dark_ship_ratio(
        self,
        df: pd.DataFrame,
        conflict_overlay: bool = True,
    ) -> None:
        """Plot dark ship ratio over time with conflict markers"""
        if "is_dark_ship" not in df.columns:
            logger.warning("No is_dark_ship column")
            return

        df = df.copy()
        df["date"] = df["BaseDateTime"].dt.date

        daily_dark = df.groupby("date").agg(
            {"MMSI": "nunique", "is_dark_ship": "sum"}
        )
        daily_dark["dark_ratio"] = (
            daily_dark["is_dark_ship"] / daily_dark["MMSI"].clip(lower=1)
        )

        fig, ax = plt.subplots(figsize=(12, 5))

        daily_dark["dark_ratio"].plot(ax=ax, color="coral", marker="o", linewidth=1.5)

        if conflict_overlay:
            for key, event in self.CONFLICT_EVENTS.items():
                if event.date >= daily_dark.index.min() and event.date <= daily_dark.index.max():
                    ax.axvline(event.date, color="red", linestyle="--", alpha=0.7)

        ax.set_xlabel("Date")
        ax.set_ylabel("Dark Ship Ratio")
        ax.set_title("Daily Dark Ship Ratio (AIS Gap > 6h)")
        ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        path = self.output_dir / "dark_ship_ratio.png"
        plt.savefig(path, dpi=self.config.dpi)
        plt.close()
        logger.info(f"Saved: {path}")

    def plot_hourly_activity(self, df: pd.DataFrame) -> None:
        """Plot hourly activity pattern"""
        df = df.copy()
        df["hour"] = df["BaseDateTime"].dt.hour

        hourly = df.groupby("hour")["MMSI"].nunique()

        fig, ax = plt.subplots(figsize=(10, 5))

        bars = ax.bar(
            hourly.index,
            hourly.values,
            color="coral",
            edgecolor="white",
        )

        ax.set_xlabel("Hour of Day (UTC)")
        ax.set_ylabel("Unique Vessels")
        ax.set_title("Hourly Vessel Activity Pattern")
        ax.set_xticks(range(24))

        plt.tight_layout()
        path = self.output_dir / "hourly_activity.png"
        plt.savefig(path, dpi=self.config.dpi)
        plt.close()
        logger.info(f"Saved: {path}")

    def plot_weekly_pattern(self, df: pd.DataFrame) -> None:
        """Plot weekly activity pattern"""
        df = df.copy()
        df["dayofweek"] = df["BaseDateTime"].dt.dayofweek
        df["day_name"] = df["BaseDateTime"].dt.day_name()

        weekly = df.groupby("dayofweek")["MMSI"].nunique()
        weekly.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        fig, ax = plt.subplots(figsize=(10, 5))

        bars = ax.bar(weekly.index, weekly.values, color="teal", edgecolor="white")

        ax.set_xlabel("Day of Week")
        ax.set_ylabel("Unique Vessels")
        ax.set_title("Weekly Vessel Activity Pattern")

        plt.tight_layout()
        path = self.output_dir / "weekly_pattern.png"
        plt.savefig(path, dpi=self.config.dpi)
        plt.close()
        logger.info(f"Saved: {path}")

    def plot_traffic_by_zone(
        self,
        df: pd.DataFrame,
        conflict_overlay: bool = True,
    ) -> None:
        """Plot traffic volume by conflict zone over time"""
        if "conflict_zone_name" not in df.columns:
            logger.warning("No conflict_zone_name column")
            return

        df = df.copy()
        df["date"] = df["BaseDateTime"].dt.date

        zones = df["conflict_zone_name"].unique()
        zones = [z for z in zones if z != "none"]

        if not zones:
            logger.warning("No conflict zones found")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        for zone in zones[:5]:
            zone_df = df[df["conflict_zone_name"] == zone]
            zone_daily = zone_df.groupby("date")["MMSI"].nunique()
            zone_daily.plot(ax=ax, marker="o", label=zone, linewidth=1.5)

        if conflict_overlay:
            for key, event in self.CONFLICT_EVENTS.items():
                ax.axvline(event.date, color="red", linestyle="--", alpha=0.5)

        ax.set_xlabel("Date")
        ax.set_ylabel("Unique Vessels")
        ax.set_title("Traffic by Conflict Zone")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        path = self.output_dir / "traffic_by_zone.png"
        plt.savefig(path, dpi=self.config.dpi)
        plt.close()
        logger.info(f"Saved: {path}")

    def run(self, df: pd.DataFrame) -> None:
        """Run all temporal visualizations"""
        logger.info("Running temporal visualizations...")

        self.plot_traffic_volume_timeseries(df, conflict_overlay=False)

        self.plot_vessel_type_composition(df)

        if "SOG" in df.columns:
            self.plot_speed_distribution_timeseries(df)

        if "is_dark_ship" in df.columns:
            self.plot_dark_ship_ratio(df, conflict_overlay=False)

        self.plot_hourly_activity(df)

        if "conflict_zone_name" in df.columns:
            self.plot_traffic_by_zone(df, conflict_overlay=False)

        logger.info("Temporal visualizations complete!")


def main():
    parser = argparse.ArgumentParser(description="Temporal Visualizations")
    parser.add_argument("--input", required=True, help="Input Parquet path")
    parser.add_argument(
        "--output-dir",
        default="./outputs/figures/temporal",
        help="Output directory",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df):,} records")

    viz = TemporalVisualizer(args.output_dir)
    viz.run(df)


if __name__ == "__main__":
    main()