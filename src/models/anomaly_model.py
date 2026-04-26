"""
Maritime Anomaly Detection
=======================
Unsupervised detection of behavioral outliers in AIS data.

Models:
  - Isolation Forest
  - Local Outlier Factor
  - DBSCAN
  - Ensemble voting

Coding Conventions:
  - Type hints
  - Logging via logger
  - Config via YAML
  - Reproducibility with SEED
  - tqdm progress bars
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from pathlib import Path
import logging
import argparse
import joblib
import yaml
from dataclasses import dataclass
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)


@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection"""
    contamination: float = 0.05
    n_estimators: int = 100
    n_neighbors: int = 20
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5


def load_config(config_path: str = "./config/settings.yaml") -> AnomalyConfig:
    """Load anomaly config from YAML"""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        m = config.get("models", {}).get("anomaly", {})
        return AnomalyConfig(
            contamination=m.get("contamination", 0.05),
            n_estimators=m.get("n_estimators", 100),
            n_neighbors=m.get("n_neighbors", 20),
            dbscan_eps=m.get("dbscan_eps", 0.5),
            dbscan_min_samples=m.get("dbscan_min_samples", 5),
        )
    except Exception:
        logger.warning(f"Could not load config from {config_path}, using defaults")
        return AnomalyConfig()


class AnomalyDetector:
    """Unsupervised anomaly detection for maritime behavior"""

    FEATURE_COLS: List[str] = [
        "SOG",
        "delta_sog",
        "delta_cog",
        "turning_rate",
        "rolling_sog_mean_12h",
        "rolling_sog_std_12h",
        "route_entropy",
        "zig_zag_index",
        "loitering_flag",
        "is_dark_ship",
    ]

    def __init__(
        self,
        output_dir: str = "./outputs/models/anomaly",
        config: Optional[AnomalyConfig] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.models: Dict[str, any] = {}
        self.config = config or AnomalyConfig()

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix for anomaly detection"""
        features = [col for col in self.FEATURE_COLS if col in df.columns]

        if not features:
            logger.warning("No features found for anomaly detection")
            return np.zeros((len(df), 1))

        X = df[features].copy()
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)

        return X.values

    def fit_isolation_forest(
        self,
        X: np.ndarray,
        contamination: Optional[float] = None,
    ) -> "IsolationForest":
        """Fit Isolation Forest model"""
        contamination = contamination or self.config.contamination
        logger.info(f"Fitting Isolation Forest (contamination={contamination})...")

        model = IsolationForest(
            n_estimators=self.config.n_estimators,
            contamination=contamination,
            random_state=SEED,
            n_jobs=-1,
        )
        model.fit(X)
        self.models["isolation_forest"] = model
        return model

    def fit_local_outlier_factor(
        self,
        X: np.ndarray,
        n_neighbors: Optional[int] = None,
    ) -> "LocalOutlierFactor":
        """Fit Local Outlier Factor model"""
        n_neighbors = n_neighbors or self.config.n_neighbors
        logger.info(f"Fitting Local Outlier Factor (n_neighbors={n_neighbors})...")

        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination="auto",
            novelty=True,
            n_jobs=-1,
        )
        model.fit(X)
        self.models["lof"] = model
        return model

    def fit_dbscan(
        self,
        X: np.ndarray,
        eps: Optional[float] = None,
        min_samples: Optional[int] = None,
    ) -> np.ndarray:
        """Fit DBSCAN clustering"""
        eps = eps or self.config.dbscan_eps
        min_samples = min_samples or self.config.dbscan_min_samples
        logger.info(f"Fitting DBSCAN (eps={eps}, min_samples={min_samples})...")

        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
        self.models["dbscan"] = model
        return labels

    def predict(self, X: np.ndarray, method: str) -> np.ndarray:
        """Get predictions from a specific model"""
        if method not in self.models:
            logger.warning(f"Model {method} not fitted")
            return np.zeros(len(X))

        model = self.models[method]

        if method == "isolation_forest":
            return (model.predict(X) == -1).astype(int)
        elif method == "lof":
            return (model.predict(X) == -1).astype(int)
        elif method == "dbscan":
            return (model.labels_ == -1).astype(int)

        return np.zeros(len(X))

    def get_anomaly_scores(self, X: np.ndarray) -> pd.DataFrame:
        """Get ensemble anomaly scores from all models"""
        results = pd.DataFrame(index=range(len(X)))

        if "isolation_forest" in self.models:
            results["if_score"] = self.models["isolation_forest"].decision_function(X)
            results["if_label"] = self.predict(X, "isolation_forest")

        if "lof" in self.models:
            results["lof_score"] = self.models["lof"].decision_function(X)
            results["lof_label"] = self.predict(X, "lof")

        if "if_score" in results.columns and "lof_score" in results.columns:
            results["ensemble_score"] = results[["if_score", "lof_score"]].mean(axis=1)
            results["ensemble_label"] = (results["ensemble_score"] < 0).astype(int)
        elif "if_label" in results.columns:
            results["ensemble_label"] = results["if_label"]
        else:
            results["ensemble_label"] = 0

        return results

    def classify_anomaly_type(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
    ) -> pd.Series:
        """Classify the type of detected anomaly"""
        types = pd.Series("normal", index=df.index, dtype=object)

        if "is_dark_ship" in df.columns:
            mask = (df["is_dark_ship"] == 1) & (predictions == 1)
            types[mask] = "dark_ship"

        if "loitering_flag" in df.columns:
            mask = (df["loitering_flag"] == 1) & (predictions == 1)
            types[mask] = "loitering"

        if "zig_zag_index" in df.columns:
            mask = (df["zig_zag_index"] > 3) & (predictions == 1)
            types[mask] = "zig_zag"

        if "SOG" in df.columns:
            mask = (df["SOG"] > 20) & (predictions == 1)
            types[mask] = "speed_spike"

        return types

    def run(
        self,
        df: pd.DataFrame,
        contamination: Optional[float] = None,
    ) -> pd.DataFrame:
        """Run full anomaly detection pipeline"""
        logger.info("Running anomaly detection...")

        X = self.prepare_features(df)
        X_scaled = self.scaler.fit_transform(X)

        self.fit_isolation_forest(X_scaled, contamination)
        self.fit_local_outlier_factor(X_scaled)

        predictions = self.get_anomaly_scores(X_scaled)

        df = df.copy()
        df["anomaly_score"] = predictions.get("ensemble_score", 0)
        df["anomaly_label"] = predictions["ensemble_label"]
        df["anomaly_type"] = self.classify_anomaly_type(
            df, predictions["ensemble_label"].values
        )

        n_anomalies = df["anomaly_label"].sum()
        pct = n_anomalies / len(df) * 100 if len(df) > 0 else 0
        logger.info(f"Anomalies detected: {n_anomalies} ({pct:.1f}%)")

        self.save_models()

        output_path = self.output_dir / "anomaly_results.parquet"
        df.to_parquet(output_path, index=False, compression="snappy")
        logger.info(f"Saved: {output_path}")

        return df

    def save_models(self) -> None:
        """Save trained models to disk"""
        for name, model in self.models.items():
            path = self.output_dir / f"{name}.joblib"
            joblib.dump(model, path)
            logger.info(f"Model saved: {path}")

        scaler_path = self.output_dir / "scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Scaler saved: {scaler_path}")

    def load_models(self) -> None:
        """Load trained models from disk"""
        for path in self.output_dir.glob("*.joblib"):
            name = path.stem
            if name == "scaler":
                self.scaler = joblib.load(path)
            else:
                self.models[name] = joblib.load(path)
            logger.info(f"Model loaded: {path}")


def main():
    parser = argparse.ArgumentParser(description="Maritime Anomaly Detection")
    parser.add_argument("--input", required=True, help="Input Parquet path")
    parser.add_argument(
        "--output-dir",
        default="./outputs/models/anomaly",
        help="Output directory",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Anomaly contamination rate",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df):,} records")

    config = load_config()
    detector = AnomalyDetector(args.output_dir, config)
    detector.run(df, args.contamination)


if __name__ == "__main__":
    main()