"""
Spatial Visualization Module
=========================
Geospatial visualizations for AIS data including folium maps.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import argparse
from typing import Optional, Dict, List, Tuple

from .base import VizConfig, load_viz_config, setup_matplotlib, ensure_output_dir

logger = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)


class SpatialVisualizer:
    CONFLICT_ZONES: Dict[str, List[float]] = {
        "black_sea": [27.0, 40.5, 41.0, 46.8],
        "azov_sea": [33.5, 45.0, 39.5, 47.5],
        "kerch_strait": [36.4, 45.1, 36.8, 45.5],
        "red_sea": [32.0, 12.0, 43.5, 30.0],
        "bab_el_mandeb": [43.0, 11.5, 45.0, 12.5],
        "taiwan_strait": [119.0, 22.0, 122.0, 26.0],
        "south_china_sea": [109.0, 3.0, 121.0, 22.0],
        "strait_hormuz": [56.0, 25.5, 59.5, 27.0],
    }

    CHOKEPOINTS: Dict[str, Tuple[float, float]] = {
        "hormuz": (56.5, 26.5),
        "malacca": (103.8, 1.2),
        "bab_mandeb": (43.4, 12.5),
        "suez": (32.5, 30.7),
        "panama": (-79.9, 9.0),
        "gibraltar": (-5.4, 36.0),
        "dover": (1.3, 51.0),
    }

    def __init__(
        self,
        output_dir: str = "./outputs/figures/spatial",
        config: Optional[VizConfig] = None,
    ):
        self.output_dir = ensure_output_dir(output_dir)
        self.config = config or load_viz_config()
        setup_matplotlib(self.config)

    def plot_global_density_heatmap(
        self,
        df: pd.DataFrame,
        time_col: str = "time_bucket",
        sample_size: int = 100000,
    ) -> None:
        """Plot vessel traffic density heatmap"""
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=SEED)

        fig, ax = plt.subplots(figsize=(12, 8))

        time_vals = (
            df[time_col].astype(int)
            if time_col in df.columns
            else np.arange(len(df))
        )

        scatter = ax.scatter(
            df["LON"],
            df["LAT"],
            c=time_vals,
            cmap="viridis",
            alpha=0.6,
            s=30,
            edgecolor="white",
            linewidth=0.3,
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label("Time Bucket")

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Vessel Traffic Density Heatmap")

        for zone, bbox in self.CONFLICT_ZONES.items():
            ax.axvline(x=bbox[0], color="red", linestyle="--", alpha=0.5)
            ax.axvline(x=bbox[2], color="red", linestyle="--", alpha=0.5)
            ax.axhline(y=bbox[1], color="red", linestyle="--", alpha=0.5)
            ax.axhline(y=bbox[3], color="red", linestyle="--", alpha=0.5)

        plt.tight_layout()
        path = self.output_dir / "density_heatmap.png"
        plt.savefig(path, dpi=self.config.dpi)
        plt.close()
        logger.info(f"Saved: {path}")

    def plot_vessel_trajectory(
        self,
        df: pd.DataFrame,
        mmsi: int,
        max_points: int = 5000,
    ) -> None:
        """Plot individual vessel trajectory"""
        vessel_df = df[df["MMSI"] == mmsi].sort_values("BaseDateTime")

        if len(vessel_df) == 0:
            logger.warning(f"No data for MMSI {mmsi}")
            return

        if len(vessel_df) > max_points:
            vessel_df = vessel_df.iloc[:: len(vessel_df) // max_points]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            vessel_df["LON"],
            vessel_df["LAT"],
            "b-",
            alpha=0.5,
            linewidth=1,
        )

        scatter = ax.scatter(
            vessel_df["LON"],
            vessel_df["LAT"],
            c=vessel_df["SOG"],
            cmap="plasma",
            s=50,
            edgecolor="black",
            linewidth=0.5,
        )

        ax.plot(
            vessel_df["LON"].iloc[0],
            vessel_df["LAT"].iloc[0],
            "go",
            markersize=10,
            label="Start",
        )
        ax.plot(
            vessel_df["LON"].iloc[-1],
            vessel_df["LAT"].iloc[-1],
            "r*",
            markersize=15,
            label="End",
        )

        cbar = plt.colorbar(scatter)
        cbar.set_label("Speed (knots)")

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Vessel Trajectory - MMSI {mmsi}")
        ax.legend()

        plt.tight_layout()
        path = self.output_dir / f"trajectory_{mmsi}.png"
        plt.savefig(path, dpi=self.config.dpi)
        plt.close()
        logger.info(f"Saved: {path}")

    def plot_dark_ship_clusters(
        self,
        df: pd.DataFrame,
        sample_size: int = 50000,
    ) -> None:
        """Plot dark ship locations in conflict zones"""
        if "is_dark_ship" not in df.columns:
            logger.warning("No is_dark_ship column found")
            return

        dark_ships = df[df["is_dark_ship"] == 1]

        if len(dark_ships) == 0:
            logger.warning("No dark ship data found")
            return

        if len(dark_ships) > sample_size:
            dark_ships = dark_ships.sample(n=sample_size, random_state=SEED)

        fig, ax = plt.subplots(figsize=(12, 7))

        ax.scatter(
            dark_ships["LON"],
            dark_ships["LAT"],
            c="red",
            alpha=0.6,
            s=30,
            label="Dark Ship",
        )

        for zone, bbox in self.CONFLICT_ZONES.items():
            rect = plt.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                fill=False,
                edgecolor="orange",
                linestyle="--",
                linewidth=1.5,
            )
            ax.add_patch(rect)
            ax.text(
                bbox[0] + 0.5,
                bbox[3] - 0.5,
                zone,
                color="orange",
                fontsize=8,
            )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Dark Ship Locations (AIS Gap > 6h)")
        ax.legend()
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)

        plt.tight_layout()
        path = self.output_dir / "dark_ships.png"
        plt.savefig(path, dpi=self.config.dpi)
        plt.close()
        logger.info(f"Saved: {path}")

    def plot_chokepoint_traffic(
        self,
        df: pd.DataFrame,
        radius_deg: float = 2.0,
    ) -> None:
        """Plot traffic near strategic chokepoints"""
        fig, ax = plt.subplots(figsize=(14, 7))

        ax.scatter(
            df["LON"],
            df["LAT"],
            c="lightblue",
            alpha=0.3,
            s=10,
            label="All Traffic",
        )

        for name, (lon, lat) in self.CHOKEPOINTS.items():
            nearby = df[
                (df["LON"].between(lon - radius_deg, lon + radius_deg))
                & (df["LAT"].between(lat - radius_deg, lat + radius_deg))
            ]

            if len(nearby) > 0:
                ax.scatter(
                    nearby["LON"],
                    nearby["LAT"],
                    c="red",
                    alpha=0.7,
                    s=20,
                    label=f"{name} ({len(nearby):,})",
                )

            ax.plot(lon, lat, "b*", markersize=15)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Chokepoint Traffic Flow")
        ax.legend()

        plt.tight_layout()
        path = self.output_dir / "chokepoint_traffic.png"
        plt.savefig(path, dpi=self.config.dpi)
        plt.close()
        logger.info(f"Saved: {path}")

    def plot_conflict_zones_with_traffic(
        self,
        df: pd.DataFrame,
        agg_df: pd.DataFrame = None,
    ) -> None:
        """Plot traffic distribution across conflict zones"""
        zone_stats = df.groupby("conflict_zone_name").agg(
            {"MMSI": "nunique", "is_dark_ship": "sum" if "is_dark_ship" in df.columns else "count"}
        ).rename(columns={"MMSI": "vessel_count"})

        zone_stats = zone_stats[zone_stats.index != "none"].sort_values("vessel_count", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(zone_stats)))
        ax.barh(zone_stats.index, zone_stats["vessel_count"], color=colors)

        ax.set_xlabel("Unique Vessels")
        ax.set_ylabel("Conflict Zone")
        ax.set_title("Traffic by Conflict Zone")

        plt.tight_layout()
        path = self.output_dir / "conflict_zones_traffic.png"
        plt.savefig(path, dpi=self.config.dpi)
        plt.close()
        logger.info(f"Saved: {path}")

    def run(self, df: pd.DataFrame, mmsi_list: List[int] = None) -> None:
        """Run all spatial visualizations"""
        logger.info("Running spatial visualizations...")

        self.plot_global_density_heatmap(df)

        if "is_dark_ship" in df.columns:
            self.plot_dark_ship_clusters(df)

        self.plot_chokepoint_traffic(df)

        if "conflict_zone_name" in df.columns:
            self.plot_conflict_zones_with_traffic(df)

        if mmsi_list:
            for mmsi in mmsi_list[:5]:
                self.plot_vessel_trajectory(df, mmsi)

        logger.info("Spatial visualizations complete!")


def main():
    parser = argparse.ArgumentParser(description="Spatial Visualizations")
    parser.add_argument("--input", required=True, help="Input Parquet path")
    parser.add_argument(
        "--output-dir",
        default="./outputs/figures/spatial",
        help="Output directory",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df):,} records")

    viz = SpatialVisualizer(args.output_dir)
    viz.run(df)


if __name__ == "__main__":
    main()