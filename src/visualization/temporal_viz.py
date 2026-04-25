"""
Temporal Visualization Module
=============================
Time-series visualizations for AIS data.
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
    dpi: int = 300
    figsize: tuple = (10, 6)
    font_family: str = "DejaVu Sans"
    font_size: int = 12
    style: str = "whitegrid"


def load_viz_config(config_path: str = "./config/settings.yaml") -> VizConfig:
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


class TemporalVisualizer:
    def __init__(self, output_dir: str = "./outputs/figures/temporal", config: VizConfig = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or load_viz_config()
        setup_matplotlib(self.config)

    def plot_traffic_volume_timeseries(self, df: pd.DataFrame):
        df = df.copy()
        df['date'] = df['BaseDateTime'].dt.date
        daily = df.groupby('date')['MMSI'].nunique()
        
        fig, ax = plt.subplots(figsize=(12, 5))
        daily.plot(ax=ax, marker='o', color='steelblue', linewidth=1.5)
        ax.fill_between(daily.index, daily.values, alpha=0.3, color='steelblue')
        ax.set_xlabel('Date')
        ax.set_ylabel('Unique Vessels')
        ax.set_title('Daily Traffic Volume')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'traffic_volume.png', dpi=300)
        plt.close()
        logger.info(f"Saved: {self.output_dir / 'traffic_volume.png'}")

    def plot_vessel_type_composition(self, df: pd.DataFrame, zone: str = None):
        if zone:
            df = df[df['conflict_zone_name'] == zone]
        
        df = df.copy()
        df['date'] = df['BaseDateTime'].dt.date
        type_by_date = df.groupby(['date', 'VesselType'])['MMSI'].nunique().unstack(fill_value=0)
        
        type_by_date = type_by_date.loc[:, type_by_date.sum().nlargest(10).index]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        type_by_date.plot.area(ax=ax, alpha=0.7, colormap='tab10')
        ax.set_xlabel('Date')
        ax.set_ylabel('Unique Vessels')
        ax.set_title('Vessel Type Composition Over Time' + (f' - {zone}' if zone else ''))
        ax.legend(title='Vessel Type', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'vessel_type_composition.png', dpi=300)
        plt.close()
        logger.info(f"Saved: {self.output_dir / 'vessel_type_composition.png'}")

    def plot_speed_distribution_timeseries(self, df: pd.DataFrame):
        df = df.copy()
        df['date'] = df['BaseDateTime'].dt.date
        df['month'] = df['BaseDateTime'].dt.month
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        daily_mean = df.groupby('date')['SOG'].mean()
        daily_mean.plot(ax=axes[0], color='teal', linewidth=1.5)
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Mean Speed (knots)')
        axes[0].set_title('Mean Speed Over Time')
        axes[0].tick_params(axis='x', rotation=45)
        
        monthly_speeds = [df[df['month'] == m]['SOG'].dropna() for m in sorted(df['month'].unique())]
        axes[1].boxplot(monthly_speeds, labels=sorted(df['month'].unique()))
        axes[1].set_xlabel('Month')
        axes[1].set_ylabel('Speed (knots)')
        axes[1].set_title('Monthly Speed Distribution')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'speed_timeseries.png', dpi=300)
        plt.close()
        logger.info(f"Saved: {self.output_dir / 'speed_timeseries.png'}")

    def plot_dark_ship_ratio(self, df: pd.DataFrame):
        df = df.copy()
        df['date'] = df['BaseDateTime'].dt.date
        
        daily_dark = df.groupby('date').agg({
            'MMSI': 'nunique',
            'is_dark_ship': 'sum'
        })
        daily_dark['dark_ratio'] = daily_dark['is_dark_ship'] / daily_dark['MMSI']
        
        fig, ax = plt.subplots(figsize=(12, 5))
        daily_dark['dark_ratio'].plot(ax=ax, color='coral', marker='o', linewidth=1.5)
        ax.set_xlabel('Date')
        ax.set_ylabel('Dark Ship Ratio')
        ax.set_title('Daily Dark Ship Ratio (AIS Gap > 6h)')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dark_ship_ratio.png', dpi=300)
        plt.close()
        logger.info(f"Saved: {self.output_dir / 'dark_ship_ratio.png'}")

    def plot_hourly_activity(self, df: pd.DataFrame):
        df = df.copy()
        df['hour'] = df['BaseDateTime'].dt.hour
        
        hourly = df.groupby('hour')['MMSI'].nunique()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(hourly.index, hourly.values, color='coral', edgecolor='white')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Unique Vessels')
        ax.set_title('Hourly Vessel Activity Pattern')
        ax.set_xticks(range(24))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'hourly_activity.png', dpi=300)
        plt.close()
        logger.info(f"Saved: {self.output_dir / 'hourly_activity.png'}")

    def run(self, df: pd.DataFrame):
        logger.info("Running temporal visualizations...")
        self.plot_traffic_volume_timeseries(df)
        self.plot_vessel_type_composition(df)
        self.plot_speed_distribution_timeseries(df)
        self.plot_dark_ship_ratio(df)
        self.plot_hourly_activity(df)
        logger.info("Temporal visualizations complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="./outputs/figures/temporal")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df):,} records")

    viz = TemporalVisualizer(args.output_dir)
    viz.run(df)


if __name__ == "__main__":
    main()