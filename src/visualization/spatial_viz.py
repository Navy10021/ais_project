"""
Spatial Visualization Module
========================
Geospatial visualizations for AIS data.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import argparse
import yaml
from dataclasses import dataclass

logger = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)

import random
random.seed(SEED)


@dataclass
class VizConfig:
    """Configuration for Visualizations"""
    dpi: int = 300
    figsize: tuple = (10, 6)
    font_family: str = "DejaVu Sans"
    font_size: int = 12
    style: str = "whitegrid"


def load_viz_config(config_path: str = "./config/settings.yaml") -> VizConfig:
    """Load visualization configuration"""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        v = config.get("visualization", {})
        return VizConfig(
            dpi=v.get("dpi", 300),
            figsize=tuple(v.get("figsize", [10, 6])),
            font_family=v.get("font_family", "DejaVu Sans"),
            font_size=v.get("font_size", 12),
            style=v.get("style", "whitegrid"),
        )
    except Exception:
        return VizConfig()


def setup_matplotlib(config: VizConfig = None):
    """Setup matplotlib with publication-quality defaults"""
    config = config or VizConfig()
    plt.rcParams.update({
        'figure.dpi': config.dpi,
        'figure.figsize': config.figsize,
        'font.family': config.font_family,
        'font.size': config.font_size,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })
    sns.set_style(config.style)


class SpatialVisualizer:
    CONFLICT_ZONES = {
        "black_sea": [27.0, 40.5, 41.0, 46.8],
        "azov_sea": [33.5, 45.0, 39.5, 47.5],
        "red_sea": [32.0, 12.0, 43.5, 30.0],
        "taiwan_strait": [119.0, 22.0, 122.0, 26.0],
        "strait_hormuz": [56.0, 25.5, 59.5, 27.0],
    }

    def __init__(self, output_dir: str = "./outputs/figures/spatial", config: VizConfig = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or load_viz_config()
        setup_matplotlib(self.config)

    def plot_global_density_heatmap(self, df: pd.DataFrame, time_col: str = "time_bucket"):
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(
            df['LON'], df['LAT'],
            c=df[time_col].astype(int) if time_col in df.columns else range(len(df)),
            cmap='viridis', alpha=0.6, s=30, edgecolor='white', linewidth=0.3
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label('Time Bucket')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Vessel Traffic Density Heatmap')
        
        for zone, bbox in self.CONFLICT_ZONES.items():
            ax.axvline(x=bbox[0], color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=bbox[2], color='red', linestyle='--', alpha=0.5)
            ax.axhline(y=bbox[1], color='red', linestyle='--', alpha=0.5)
            ax.axhline(y=bbox[3], color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'density_heatmap.png', dpi=300)
        plt.close()
        logger.info(f"Saved: {self.output_dir / 'density_heatmap.png'}")

    def plot_vessel_trajectory(self, df: pd.DataFrame, mmsi: int):
        vessel_df = df[df['MMSI'] == mmsi].sort_values('BaseDateTime')
        if len(vessel_df) == 0:
            logger.warning(f"No data for MMSI {mmsi}")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(vessel_df['LON'], vessel_df['LAT'], 'b-', alpha=0.5, linewidth=1)
        scatter = ax.scatter(
            vessel_df['LON'], vessel_df['LAT'],
            c=vessel_df['SOG'], cmap='plasma',
            s=50, edgecolor='black', linewidth=0.5
        )
        ax.plot(vessel_df['LON'].iloc[0], vessel_df['LAT'].iloc[0], 'go', label='Start')
        ax.plot(vessel_df['LON'].iloc[-1], vessel_df['LAT'].iloc[-1], 'r*', label='End')
        cbar = plt.colorbar(scatter)
        cbar.set_label('Speed (knots)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Vessel Trajectory - MMSI {mmsi}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / f'trajectory_{mmsi}.png', dpi=300)
        plt.close()
        logger.info(f"Saved: {self.output_dir / f'trajectory_{mmsi}.png'}")

    def plot_dark_ship_clusters(self, df: pd.DataFrame):
        dark_ships = df[df['is_dark_ship'] == 1]
        if len(dark_ships) == 0:
            logger.warning("No dark ship data found")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(
            dark_ships['LON'], dark_ships['LAT'],
            c='red', alpha=0.6, s=30, label='Dark Ship'
        )
        
        for zone, bbox in self.CONFLICT_ZONES.items():
            rect = plt.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0], bbox[3] - bbox[1],
                fill=False, edgecolor='orange', linestyle='--', linewidth=1.5
            )
            ax.add_patch(rect)
            ax.text(bbox[0] + 0.5, bbox[3] - 0.5, zone, color='orange', fontsize=8)
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Dark Ship Locations (AIS Gap > 6h)')
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dark_ships.png', dpi=300)
        plt.close()
        logger.info(f"Saved: {self.output_dir / 'dark_ships.png'}")

    def plot_chokepoint_traffic(self, df: pd.DataFrame):
        chokepoints = {
            'hormuz': (56.5, 26.5),
            'suez': (32.5, 30.7),
            'malacca': (103.8, 1.2),
        }
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(df['LON'], df['LAT'], c='lightblue', alpha=0.3, s=10, label='All Traffic')
        
        for name, (lon, lat) in chokepoints.items():
            nearby = df[
                (df['LON'].between(lon - 2, lon + 2)) &
                (df['LAT'].between(lat - 2, lat + 2))
            ]
            ax.scatter(nearby['LON'], nearby['LAT'], c='red', alpha=0.7, s=20, label=f'{name} Traffic')
            ax.plot(lon, lat, 'b*', markersize=15, label=f'{name} Point')
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Chokepoint Traffic Flow')
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'chokepoint_traffic.png', dpi=300)
        plt.close()
        logger.info(f"Saved: {self.output_dir / 'chokepoint_traffic.png'}")

    def run(self, df: pd.DataFrame):
        logger.info("Running spatial visualizations...")
        self.plot_global_density_heatmap(df)
        self.plot_dark_ship_clusters(df)
        self.plot_chokepoint_traffic(df)
        logger.info("Spatial visualizations complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="./outputs/figures/spatial")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df):,} records")

    viz = SpatialVisualizer(args.output_dir)
    viz.run(df)


if __name__ == "__main__":
    main()