"""
Statistical Visualization Module
============================
Statistical visualizations for model results and feature analysis.
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


class StatisticalVisualizer:
    def __init__(self, output_dir: str = "./outputs/figures/statistical", config: VizConfig = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or load_viz_config()
        setup_matplotlib(self.config)

    def plot_correlation_heatmap(self, df: pd.DataFrame, features: list = None):
        if features is None:
            features = ['SOG', 'COG', 'Heading', 'VesselType', 'Length', 'Width', 
                      'delta_sog', 'delta_cog', 'time_diff_sec']
            features = [f for f in features if f in df.columns]
        
        corr = df[features].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                   ax=ax, mask=mask, square=True, linewidths=0.5,
                   annot_kws={'size': 8})
        ax.set_title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_heatmap.png', dpi=300)
        plt.close()
        logger.info(f"Saved: {self.output_dir / 'correlation_heatmap.png'}")

    def plot_feature_distributions(self, df: pd.DataFrame, features: list = None):
        if features is None:
            features = ['SOG', 'COG', 'Heading', 'Length', 'Width']
            features = [f for f in features if f in df.columns]
        
        n_features = len(features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_features == 1:
            axes = [axes]
        axes = axes.flatten()
        
        for i, feat in enumerate(features):
            data = df[feat].dropna()
            axes[i].hist(data, bins=30, color='teal', edgecolor='white', alpha=0.7)
            axes[i].set_xlabel(feat)
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'{feat} Distribution')
        
        for i in range(len(features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_distributions.png', dpi=300)
        plt.close()
        logger.info(f"Saved: {self.output_dir / 'feature_distributions.png'}")

    def plot_vessel_type_bar(self, df: pd.DataFrame):
        type_counts = df['VesselType'].value_counts().head(15)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(type_counts)))
        ax.barh(range(len(type_counts)), type_counts.values, color=colors)
        ax.set_yticks(range(len(type_counts)))
        ax.set_yticklabels(type_counts.index.astype(str))
        ax.set_xlabel('Count')
        ax.set_ylabel('Vessel Type')
        ax.set_title('Vessel Type Distribution')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'vessel_type_bar.png', dpi=300)
        plt.close()
        logger.info(f"Saved: {self.output_dir / 'vessel_type_bar.png'}")

    def plot_speed_category_distribution(self, df: pd.DataFrame):
        if 'speed_category' not in df.columns:
            df['speed_category'] = pd.cut(
                df['SOG'],
                bins=[-0.1, 0.5, 3.0, 8.0, 15.0, 102.2],
                labels=['anchored', 'drifting', 'slow', 'cruising', 'fast']
            )
        
        cat_counts = df['speed_category'].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=cat_counts.index, y=cat_counts.values, ax=ax, color='coral')
        ax.set_xlabel('Speed Category')
        ax.set_ylabel('Count')
        ax.set_title('Speed Category Distribution')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'speed_category.png', dpi=300)
        plt.close()
        logger.info(f"Saved: {self.output_dir / 'speed_category.png'}")

    def plot_conflict_zone_distribution(self, df: pd.DataFrame):
        zone_counts = df['conflict_zone_name'].value_counts()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(zone_counts)))
        axes[0].pie(zone_counts.values, labels=zone_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        axes[0].set_title('Conflict Zone Proportions')
        
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(zone_counts)))
        axes[1].barh(range(len(zone_counts)), zone_counts.values, color=colors)
        axes[1].set_yticks(range(len(zone_counts)))
        axes[1].set_yticklabels(zone_counts.index)
        axes[1].set_xlabel('Count')
        axes[1].set_ylabel('Conflict Zone')
        axes[1].set_title('Records by Conflict Zone')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'conflict_zone_distribution.png', dpi=300)
        plt.close()
        logger.info(f"Saved: {self.output_dir / 'conflict_zone_distribution.png'}")

    def plot_pre_post_conflict_comparison(self, df: pd.DataFrame, event_date: str):
        if 'in_conflict_zone' not in df.columns or 'conflict_label' not in df.columns:
            logger.warning("Missing conflict columns for comparison")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        pre = df[df['conflict_label'] == 0]['SOG'].dropna()
        post = df[df['conflict_label'] == 1]['SOG'].dropna()
        
        axes[0].hist(pre, bins=30, alpha=0.6, label='Pre-Conflict', color='steelblue')
        axes[0].hist(post, bins=30, alpha=0.6, label='Post-Conflict', color='coral')
        axes[0].set_xlabel('Speed (knots)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Speed Distribution: Pre vs Post')
        axes[0].legend()
        
        data = df[['SOG', 'conflict_label']].dropna()
        data.boxplot(column='SOG', by='conflict_label', ax=axes[1])
        axes[1].set_xlabel('Conflict Label')
        axes[1].set_ylabel('Speed (knots)')
        axes[1].set_title('Speed by Conflict Label')
        plt.suptitle('')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pre_post_comparison.png', dpi=300)
        plt.close()
        logger.info(f"Saved: {self.output_dir / 'pre_post_comparison.png'}")

    def plot_model_evaluation(self, metrics: dict):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        if 'auroc' in metrics:
            axes[0].bar(['AUROC'], [metrics['auroc']], color='steelblue')
            axes[0].set_ylim([0, 1])
            axes[0].set_title('AUROC Score')
        
        if 'precision' in metrics and 'recall' in metrics:
            axes[1].bar(['Precision', 'Recall'], 
                      [metrics['precision'], metrics['recall']], 
                      color=['steelblue', 'coral'])
            axes[1].set_ylim([0, 1])
            axes[1].set_title('Precision & Recall')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_evaluation.png', dpi=300)
        plt.close()
        logger.info(f"Saved: {self.output_dir / 'model_evaluation.png'}")

    def run(self, df: pd.DataFrame):
        logger.info("Running statistical visualizations...")
        self.plot_correlation_heatmap(df)
        self.plot_feature_distributions(df)
        self.plot_vessel_type_bar(df)
        self.plot_speed_category_distribution(df)
        self.plot_conflict_zone_distribution(df)
        logger.info("Statistical visualizations complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="./outputs/figures/statistical")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df):,} records")

    viz = StatisticalVisualizer(args.output_dir)
    viz.run(df)


if __name__ == "__main__":
    main()