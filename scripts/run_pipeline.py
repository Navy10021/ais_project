"""
Run MCIS Pipeline
================
Executes the full MCIS analysis pipeline.
"""
import argparse
import logging
import sys
from pathlib import Path


def run_preprocessing():
    """Run preprocessing pipeline"""
    from src.preprocessing.cleaner import AISCleaner
    from src.preprocessing.feature_engineer import AISFeatureEngineer

    logging.info("Step 1: Data Cleaning...")
    cleaner = AISCleaner(
        "./data/raw/ais_raw.csv",
        "./data/processed/ais_clean.parquet"
    )
    cleaner.run()

    logging.info("Step 2: Feature Engineering...")
    engineer = AISFeatureEngineer()
    df = cleaner.run() if hasattr(cleaner, 'run') else None
    if df is None:
        df = {"output_path": "./data/processed/ais_clean.parquet"}

    return df


def run_analysis():
    """Run analysis modules"""
    from src.analysis.correlation_analyzer import ConflictCorrelationAnalyzer
    from src.analysis.traffic_analyzer import TrafficAnalyzer
    from src.analysis.behavioral_analyzer import BehavioralAnalyzer

    df = None  # Load from processed

    logging.info("Step 3: Traffic Analysis...")
    traffic = TrafficAnalyzer("./outputs/tables")
    if df is not None:
        traffic.run_all(df)

    logging.info("Step 4: Behavioral Analysis...")
    behavioral = BehavioralAnalyzer("./outputs/tables")
    if df is not None:
        behavioral.run_all(df)

    logging.info("Step 5: Correlation Analysis...")
    correlation = ConflictCorrelationAnalyzer("./outputs/tables")
    if df is not None:
        correlation.run_all(df)


def run_models():
    """Run model training"""
    from src.models.anomaly_model import AnomalyDetector
    from src.models.conflict_predictor import ConflictPredictor

    df = None  # Load from processed

    logging.info("Step 6: Anomaly Detection...")
    anomaly = AnomalyDetector("./outputs/models/anomaly")
    if df is not None:
        anomaly.run(df)

    logging.info("Step 7: Conflict Prediction...")
    predictor = ConflictPredictor("./outputs/models/predictor")
    if df is not None:
        predictor.run(df)


def run_visualization():
    """Run visualizations"""
    from src.visualization.spatial_viz import SpatialVisualizer
    from src.visualization.temporal_viz import TemporalVisualizer
    from src.visualization.statistical_viz import StatisticalVisualizer

    df = None  # Load from processed

    logging.info("Step 8: Spatial Visualization...")
    spatial = SpatialVisualizer("./outputs/figures/spatial")
    if df is not None:
        spatial.run(df)

    logging.info("Step 9: Temporal Visualization...")
    temporal = TemporalVisualizer("./outputs/figures/temporal")
    if df is not None:
        temporal.run(df)

    logging.info("Step 10: Statistical Visualization...")
    statistical = StatisticalVisualizer("./outputs/figures/statistical")
    if df is not None:
        statistical.run(df)


def run_eda():
    """Run EDA"""
    import pandas as pd

    logging.info("Running EDA...")

    df_path = Path("./data/processed/ais_features.parquet")
    if df_path.exists():
        df = pd.read_parquet(df_path)
        logging.info(f"Records: {len(df):,}")
        logging.info(f"Columns: {len(df.columns)}")
    else:
        logging.warning("No processed data found")


def main():
    parser = argparse.ArgumentParser(description="MCIS Pipeline Runner")
    parser.add_argument(
        "--step",
        choices=["preprocessing", "analysis", "models", "viz", "eda", "full"],
        default="full",
        help="Pipeline step to run",
    )
    parser.add_argument(
        "--input",
        default="./data/processed/ais_features.parquet",
        help="Input data path",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    logging.info("=" * 50)
    logging.info("MCIS Pipeline Runner")
    logging.info("=" * 50)

    if args.step == "full":
        run_preprocessing()
        run_analysis()
        run_models()
        run_visualization()
    elif args.step == "preprocessing":
        run_preprocessing()
    elif args.step == "analysis":
        run_analysis()
    elif args.step == "models":
        run_models()
    elif args.step == "viz":
        run_visualization()
    elif args.step == "eda":
        run_eda()

    logging.info("=" * 50)
    logging.info("Pipeline complete!")
    logging.info("=" * 50)


if __name__ == "__main__":
    main()