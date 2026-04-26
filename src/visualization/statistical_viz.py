"""
Statistical Visualization Module
=====================
Statistical visualizations for model results, feature analysis, and correlations.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import argparse
from typing import Optional, List, Dict, Tuple

from .base import VizConfig, load_viz_config, setup_matplotlib, ensure_output_dir

logger = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)


class StatisticalVisualizer:
    def __init__(
        self,
        output_dir: str = "./outputs/figures/statistical",
        config: Optional[VizConfig] = None,
    ):
        self.output_dir = ensure_output_dir(output_dir)
        self.config = config or load_viz_config()
        setup_matplotlib(self.config)

    def plot_correlation_heatmap(
        self,
        df: pd.DataFrame,
        features: List[str] = None,
    ) -> None:
        """Plot feature correlation matrix heatmap"""
        if features is None:
            features = [
                "SOG",
                "COG",
                "Heading",
                "VesselType",
                "Length",
                "Width",
                "delta_sog",
                "delta_cog",
                "time_diff_sec",
                "rolling_sog_mean_12h",
                "rolling_sog_std_12h",
            ]
            features = [f for f in features if f in df.columns]

        if len(features) < 2:
            logger.warning("Not enough features for correlation")
            return

        corr = df[features].corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))

        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            ax=ax,
            mask=mask,
            square=True,
            linewidths=0.5,
            annot_kws={"size": 8},
        )

        ax.set_title("Feature Correlation Matrix")

        plt.tight_layout()
        path = self.output_dir / "correlation_heatmap.png"
        plt.savefig(path, dpi=self.config.dpi)
        plt.close()
        logger.info(f"Saved: {path}")

    def plot_feature_distributions(
        self,
        df: pd.DataFrame,
        features: List[str] = None,
        n_cols: int = 3,
    ) -> None:
        """Plot histograms for multiple features"""
        if features is None:
            features = ["SOG", "COG", "Heading", "Length", "Width", "Draft"]
            features = [f for f in features if f in df.columns]

        if not features:
            logger.warning("No features found for distribution plots")
            return

        n_features = len(features)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]

        for i, feat in enumerate(features):
            data = df[feat].dropna()
            if len(data) == 0:
                continue

            axes[i].hist(data, bins=30, color="teal", edgecolor="white", alpha=0.7)
            axes[i].set_xlabel(feat)
            axes[i].set_ylabel("Frequency")
            axes[i].set_title(f"{feat} Distribution")

        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        path = self.output_dir / "feature_distributions.png"
        plt.savefig(path, dpi=self.config.dpi)
        plt.close()
        logger.info(f"Saved: {path}")

    def plot_vessel_type_bar(
        self,
        df: pd.DataFrame,
        top_n: int = 15,
    ) -> None:
        """Plot vessel type distribution bar chart"""
        if "VesselType" not in df.columns:
            logger.warning("No VesselType column")
            return

        type_counts = df["VesselType"].value_counts().head(top_n)

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.viridis(np.linspace(0, 1, len(type_counts)))
        ax.barh(range(len(type_counts)), type_counts.values, color=colors)

        ax.set_yticks(range(len(type_counts)))
        ax.set_yticklabels(type_counts.index.astype(int))
        ax.set_xlabel("Count")
        ax.set_ylabel("Vessel Type Code")
        ax.set_title("Vessel Type Distribution")

        plt.tight_layout()
        path = self.output_dir / "vessel_type_bar.png"
        plt.savefig(path, dpi=self.config.dpi)
        plt.close()
        logger.info(f"Saved: {path}")

    def plot_speed_category_distribution(self, df: pd.DataFrame) -> None:
        """Plot speed category distribution"""
        if "speed_category" not in df.columns:
            df["speed_category"] = pd.cut(
                df["SOG"],
                bins=[-0.1, 0.5, 3.0, 8.0, 15.0, 102.2],
                labels=["anchored", "drifting", "slow", "cruising", "fast"],
            )

        cat_counts = df["speed_category"].value_counts()

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.barplot(x=cat_counts.index, y=cat_counts.values, ax=ax, color="coral")

        ax.set_xlabel("Speed Category")
        ax.set_ylabel("Count")
        ax.set_title("Speed Category Distribution")

        plt.tight_layout()
        path = self.output_dir / "speed_category.png"
        plt.savefig(path, dpi=self.config.dpi)
        plt.close()
        logger.info(f"Saved: {path}")

    def plot_conflict_zone_distribution(self, df: pd.DataFrame) -> None:
        """Plot conflict zone distribution pie and bar charts"""
        if "conflict_zone_name" not in df.columns:
            logger.warning("No conflict_zone_name column")
            return

        zone_counts = df["conflict_zone_name"].value_counts()
        zone_counts = zone_counts[zone_counts.index != "none"]

        if len(zone_counts) == 0:
            logger.warning("No conflict zone data")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        colors = plt.cm.Pastel1(np.linspace(0, 1, len(zone_counts)))
        axes[0].pie(
            zone_counts.values,
            labels=zone_counts.index,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
        axes[0].set_title("Conflict Zone Proportions")

        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(zone_counts)))
        axes[1].barh(range(len(zone_counts)), zone_counts.values, color=colors)
        axes[1].set_yticks(range(len(zone_counts)))
        axes[1].set_yticklabels(zone_counts.index)
        axes[1].set_xlabel("Count")
        axes[1].set_ylabel("Conflict Zone")
        axes[1].set_title("Records by Conflict Zone")

        plt.tight_layout()
        path = self.output_dir / "conflict_zone_distribution.png"
        plt.savefig(path, dpi=self.config.dpi)
        plt.close()
        logger.info(f"Saved: {path}")

    def plot_pre_post_conflict_comparison(
        self,
        df: pd.DataFrame,
        metric: str = "SOG",
    ) -> None:
        """Plot pre/post conflict comparison for a metric"""
        if "conflict_label" not in df.columns or metric not in df.columns:
            logger.warning(f"Missing {metric} or conflict_label column")
            return

        pre = df[df["conflict_label"] == 0][metric].dropna()
        post = df[df["conflict_label"] == 1][metric].dropna()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].hist(pre, bins=30, alpha=0.6, label="Pre-Conflict", color="steelblue")
        axes[0].hist(post, bins=30, alpha=0.6, label="Post-Conflict", color="coral")
        axes[0].set_xlabel(metric)
        axes[0].set_ylabel("Frequency")
        axes[0].set_title(f"{metric}: Pre vs Post Conflict")
        axes[0].legend()

        data = df[[metric, "conflict_label"]].dropna()
        data.boxplot(column=metric, by="conflict_label", ax=axes[1])
        axes[1].set_xlabel("Conflict Label")
        axes[1].set_ylabel(metric)
        axes[1].set_title(f"{metric} by Conflict Label")
        plt.suptitle("")

        plt.tight_layout()
        path = self.output_dir / "pre_post_comparison.png"
        plt.savefig(path, dpi=self.config.dpi)
        plt.close()
        logger.info(f"Saved: {path}")

    def plot_model_evaluation(
        self,
        metrics: Dict[str, float],
        title: str = "Model Evaluation",
    ) -> None:
        """Plot model evaluation metrics"""
        if not metrics:
            logger.warning("No metrics provided")
            return

        metric_names = []
        metric_values = []

        for key in ["auroc", "auprc", "precision", "recall", "f1", "f2"]:
            if key in metrics:
                metric_names.append(key.upper())
                metric_values.append(metrics[key])

        if not metric_values:
            logger.warning("No recognized metrics")
            return

        fig, ax = plt.subplots(figsize=(10, 5))

        colors = ["steelblue" if v >= 0.7 else "coral" for v in metric_values]
        bars = ax.bar(metric_names, metric_values, color=colors)

        ax.set_ylim([0, 1.1])
        ax.set_ylabel("Score")
        ax.set_title(title)

        for bar, val in zip(bars, metric_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}",
                ha="center",
                fontsize=10,
            )

        plt.tight_layout()
        path = self.output_dir / "model_evaluation.png"
        plt.savefig(path, dpi=self.config.dpi)
        plt.close()
        logger.info(f"Saved: {path}")

    def plot_roc_curves(
        self,
        results: Dict[str, Dict[str, float]],
    ) -> None:
        """Plot ROC curves for multiple models"""
        fig, ax = plt.subplots(figsize=(8, 8))

        for model_name, metrics in results.items():
            if "fpr" in metrics and "tpr" in metrics:
                ax.plot(
                    metrics["fpr"],
                    metrics["tpr"],
                    label=f"{model_name} (AUC={metrics.get('auroc', 0):.2f})",
                    linewidth=2,
                )

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves")
        ax.legend()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()
        path = self.output_dir / "roc_curves.png"
        plt.savefig(path, dpi=self.config.dpi)
        plt.close()
        logger.info(f"Saved: {path}")

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        labels: List[str] = None,
    ) -> None:
        """Plot confusion matrix heatmap"""
        if labels is None:
            labels = ["No Conflict", "Conflict"]

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")

        plt.tight_layout()
        path = self.output_dir / "confusion_matrix.png"
        plt.savefig(path, dpi=self.config.dpi)
        plt.close()
        logger.info(f"Saved: {path}")

    def run(self, df: pd.DataFrame) -> None:
        """Run all statistical visualizations"""
        logger.info("Running statistical visualizations...")

        self.plot_correlation_heatmap(df)

        self.plot_feature_distributions(df)

        if "VesselType" in df.columns:
            self.plot_vessel_type_bar(df)

        if "speed_category" in df.columns or "SOG" in df.columns:
            self.plot_speed_category_distribution(df)

        if "conflict_zone_name" in df.columns:
            self.plot_conflict_zone_distribution(df)

        if "conflict_label" in df.columns:
            self.plot_pre_post_conflict_comparison(df)

        logger.info("Statistical visualizations complete!")


def main():
    parser = argparse.ArgumentParser(description="Statistical Visualizations")
    parser.add_argument("--input", required=True, help="Input Parquet path")
    parser.add_argument(
        "--output-dir",
        default="./outputs/figures/statistical",
        help="Output directory",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df):,} records")

    viz = StatisticalVisualizer(args.output_dir)
    viz.run(df)


if __name__ == "__main__":
    main()