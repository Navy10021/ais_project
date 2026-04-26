"""
Report Generator
================
Compiles all outputs into a final report.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import argparse
from datetime import datetime

logger = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.dpi": 150,
    "figure.figsize": (10, 6),
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


class ReportGenerator:
    """Generate final analysis reports"""

    def __init__(self, output_dir: str = "./outputs/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_summary(self) -> str:
        report = []
        report.append("=" * 60)
        report.append("Maritime Conflict Intelligence System (MCIS)")
        report.append("Final Analysis Report")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        return "\n".join(report)

    def compile_data_summary(self) -> str:
        lines = []
        lines.append("\n## Data Summary")
        lines.append("-" * 40)

        clean_path = Path("./data/processed/ais_clean.parquet")
        if clean_path.exists():
            df = pd.read_parquet(clean_path)
            lines.append(f"Cleaned records: {len(df):,}")
            lines.append(f"Cleaned columns: {len(df.columns)}")

        features_path = Path("./data/processed/ais_features.parquet")
        if features_path.exists():
            df = pd.read_parquet(features_path)
            lines.append(f"Feature records: {len(df):,}")
            lines.append(f"Feature columns: {len(df.columns)}")

        return "\n".join(lines)

    def compile_visualization_summary(self) -> str:
        lines = []
        lines.append("\n## Visualizations")
        lines.append("-" * 40)

        fig_dirs = ["eda", "spatial", "temporal", "statistical"]
        for fig_dir in fig_dirs:
            path = Path(f"./outputs/figures/{fig_dir}")
            if path.exists():
                files = list(path.glob("*.png"))
                lines.append(f"{fig_dir}: {len(files)} figures")

        return "\n".join(lines)

    def compile_model_summary(self) -> str:
        lines = []
        lines.append("\n## Models")
        lines.append("-" * 40)

        model_dirs = ["anomaly", "predictor"]
        for model_dir in model_dirs:
            path = Path(f"./outputs/models/{model_dir}")
            if path.exists():
                files = list(path.glob("*.joblib"))
                lines.append(f"{model_dir}: {len(files)} models")

        pred_path = Path("./outputs/models/predictor/evaluation_results.csv")
        if pred_path.exists():
            df = pd.read_csv(pred_path)
            if "auroc" in df.columns:
                lines.append("\nPrediction Results:")
                for idx, row in df.iterrows():
                    lines.append(f"  {row.get('model', idx)}: AUROC={row.get('auroc', 0):.3f}, F1={row.get('f1', 0):.3f}")

        return "\n".join(lines)

    def compile_analysis_summary(self) -> str:
        lines = []
        lines.append("\n## Analysis Results")
        lines.append("-" * 40)

        tables_path = Path("./outputs/tables")
        if tables_path.exists():
            tables = list(tables_path.glob("*.csv"))
            lines.append(f"Analysis tables: {len(tables)}")

        return "\n".join(lines)

    def create_html_report(self, output_file: Path):
        content = [self.generate_summary()]
        content.append(self.compile_data_summary())
        content.append(self.compile_visualization_summary())
        content.append(self.compile_model_summary())
        content.append(self.compile_analysis_summary())

        report_text = "\n".join(content)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>MCIS Final Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        pre {{ background: #f8f9fa; padding: 15px; border-radius: 5px; }}
        .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>Maritime Conflict Intelligence System</h1>
    <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Data Summary</h2>
    <pre>{self.compile_data_summary()}</pre>
    
    <h2>Visualizations</h2>
    <pre>{self.compile_visualization_summary()}</pre>
    
    <h2>Models</h2>
    <pre>{self.compile_model_summary()}</pre>
    
    <h2>Analysis Results</h2>
    <pre>{self.compile_analysis_summary()}</pre>
    
    <h2>Pipeline Summary</h2>
    <ol>
        <li>Data Cleaning: ais_raw.csv → ais_clean.parquet</li>
        <li>Feature Engineering: ais_clean.parquet → ais_features.parquet</li>
        <li>EDA: outputs/figures/eda/</li>
        <li>Spatial Visualization: outputs/figures/spatial/</li>
        <li>Temporal Visualization: outputs/figures/temporal/</li>
        <li>Statistical Visualization: outputs/figures/statistical/</li>
        <li>Anomaly Detection: outputs/models/anomaly/</li>
        <li>Conflict Prediction: outputs/models/predictor/</li>
    </ol>
</body>
</html>"""

        output_file.write_text(html, encoding="utf-8")
        logger.info(f"HTML Report saved: {output_file}")

        txt_file = output_file.with_suffix(".txt")
        txt_file.write_text(report_text, encoding="utf-8")
        logger.info(f"Text Report saved: {txt_file}")

    def create_summary_figure(self):
        """Create summary dashboard figure"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        data_path = Path("./data/processed/ais_features.parquet")
        if data_path.exists():
            df = pd.read_parquet(data_path)

            axes[0, 0].hist(df["SOG"].dropna(), bins=30, color="teal", alpha=0.7)
            axes[0, 0].set_xlabel("Speed (knots)")
            axes[0, 0].set_title("Speed Distribution")

            type_counts = df["VesselType"].value_counts().head(10)
            y_pos = range(len(type_counts))
            axes[0, 1].barh(y_pos, type_counts.values)
            axes[0, 1].set_yticks(y_pos)
            axes[0, 1].set_yticklabels(type_counts.index)
            axes[0, 1].set_title("Vessel Types")

            sample = df.sample(n=min(10000, len(df)), random_state=42)
            axes[1, 0].scatter(sample["LON"], sample["LAT"], c=sample["SOG"], cmap="viridis", alpha=0.5, s=10)
            axes[1, 0].set_xlabel("Longitude")
            axes[1, 0].set_ylabel("Latitude")
            axes[1, 0].set_title("Geographic Distribution")

            df["date"] = df["BaseDateTime"].dt.date
            daily = df.groupby("date")["MMSI"].nunique()
            axes[1, 1].plot(daily.index, daily.values, marker="o", color="steelblue")
            axes[1, 1].set_xlabel("Date")
            axes[1, 1].set_ylabel("Unique Vessels")
            axes[1, 1].set_title("Daily Traffic")
            axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        fig.savefig(self.output_dir / "summary_figure.png", dpi=300)
        plt.close()
        logger.info(f"Summary figure saved: {self.output_dir / 'summary_figure.png'}")

    def run(self):
        """Generate full report"""
        logger.info("Generating final report...")

        self.create_html_report(self.output_dir / "mcis_final_report.html")
        self.create_summary_figure()

        logger.info(f"Report complete! Output: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="MCIS Report Generator")
    parser.add_argument("--output-dir", default="./outputs/reports")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    generator = ReportGenerator(args.output_dir)
    generator.run()


if __name__ == "__main__":
    main()