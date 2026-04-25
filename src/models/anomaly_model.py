"""
Maritime Anomaly Detection
======================
Unsupervised detection of behavioral outliers in AIS data.

Coding Conventions:
  - Type hints
  - Logging via logger
  - Config via YAML
  - Reproducibility with SEED
  - tqdm progress bars for long loops
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

logger = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)

import random
random.seed(SEED)


@dataclass
class AnomalyConfig:
    contamination: float = 0.05
    n_estimators: int = 100
    n_neighbors: int = 20


def load_config(config_path: str = "./config/settings.yaml") -> AnomalyConfig:
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        m = config.get("models", {}).get("anomaly", {})
        return AnomalyConfig(
            contamination=m.get("contamination", 0.05),
            n_estimators=m.get("n_estimators", 100),
            n_neighbors=m.get("n_neighbors", 20),
        )
    except Exception:
        return AnomalyConfig()


class AnomalyDetector:
    FEATURE_COLS = [
        'SOG', 'delta_sog', 'delta_cog', 'turning_rate',
        'rolling_sog_mean_12h', 'rolling_sog_std_12h', 'route_entropy',
        'zig_zag_index', 'loitering_flag', 'is_dark_ship'
    ]

    def __init__(self, output_dir: str = "./outputs/models/anomaly", config: AnomalyConfig = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.models = {}
        self.config = config or load_config()

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = []
        for col in self.FEATURE_COLS:
            if col in df.columns:
                features.append(col)
        
        X = df[features].copy()
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        return X

    def fit_isolation_forest(self, X: np.ndarray, contamination: float = 0.05):
        logger.info(f"Fitting Isolation Forest (contamination={contamination})...")
        model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=SEED,
            n_jobs=-1
        )
        model.fit(X)
        self.models['isolation_forest'] = model
        return model

    def fit_local_outlier_factor(self, X: np.ndarray, n_neighbors: int = 20):
        logger.info(f"Fitting Local Outlier Factor (n_neighbors={n_neighbors})...")
        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination='auto',
            novelty=True,
            n_jobs=-1
        )
        model.fit(X)
        self.models['lof'] = model
        return model

    def fit_dbscan(self, X: np.ndarray, eps: float = 0.5, min_samples: int = 5):
        logger.info(f"Fitting DBSCAN (eps={eps}, min_samples={min_samples})...")
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
        self.models['dbscan'] = model
        return labels

    def predict(self, X: np.ndarray, method: str = 'isolation_forest') -> np.ndarray:
        if method not in self.models:
            logger.warning(f"Model {method} not fitted")
            return np.zeros(len(X))

        model = self.models[method]
        
        if method == 'isolation_forest':
            scores = model.decision_function(X)
            return (model.predict(X) == -1).astype(int)
        elif method == 'lof':
            return (model.predict(X) == -1).astype(int)
        elif method == 'dbscan':
            return (model.labels_ == -1).astype(int)
        
        return np.zeros(len(X))

    def get_anomaly_scores(self, X: np.ndarray) -> pd.DataFrame:
        results = pd.DataFrame(index=range(len(X)))

        if 'isolation_forest' in self.models:
            results['if_score'] = self.models['isolation_forest'].decision_function(X)
            results['if_label'] = self.predict(X, 'isolation_forest')
        
        if 'lof' in self.models:
            results['lof_score'] = self.models['lof'].decision_function(X)
            results['lof_label'] = self.predict(X, 'lof')

        results['ensemble_score'] = results.filter(like='score').mean(axis=1)
        results['ensemble_label'] = (results['ensemble_score'] < 0).astype(int)
        
        return results

    def classify_anomaly_type(self, df: pd.DataFrame, predictions: np.ndarray) -> pd.Series:
        types = pd.Series('normal', index=df.index)

        if 'is_dark_ship' in df.columns:
            types[(df['is_dark_ship'] == 1) & (predictions == 1)] = 'dark_ship'

        if 'loitering_flag' in df.columns:
            types[(df['loitering_flag'] == 1) & (predictions == 1)] = 'loitering'

        if 'zig_zag_index' in df.columns:
            types[(df['zig_zag_index'] > 3) & (predictions == 1)] = 'zig_zag'

        if 'SOG' in df.columns:
            types[(df['SOG'] > 20) & (predictions == 1)] = 'speed_spike'

        return types

    def run(self, df: pd.DataFrame, contamination: float = 0.05):
        logger.info("Running anomaly detection...")
        
        X = self.prepare_features(df)
        X_scaled = self.scaler.fit_transform(X)

        self.fit_isolation_forest(X_scaled, contamination)
        self.fit_local_outlier_factor(X_scaled)

        predictions = self.get_anomaly_scores(X_scaled)
        
        df['anomaly_score'] = predictions['ensemble_score']
        df['anomaly_label'] = predictions['ensemble_label']
        df['anomaly_type'] = self.classify_anomaly_type(df, predictions['ensemble_label'])

        logger.info(f"Anomalies detected: {df['anomaly_label'].sum()} ({df['anomaly_label'].mean()*100:.1f}%)")
        
        self.save_models()
        
        df.to_parquet(self.output_dir / 'anomaly_results.parquet', index=False)
        logger.info(f"Saved: {self.output_dir / 'anomaly_results.parquet'}")
        
        return df

    def save_models(self):
        for name, model in self.models.items():
            path = self.output_dir / f'{name}.joblib'
            joblib.dump(model, path)
            logger.info(f"Model saved: {path}")

    def load_models(self):
        for path in self.output_dir.glob('*.joblib'):
            name = path.stem
            self.models[name] = joblib.load(path)
            logger.info(f"Model loaded: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="./outputs/models/anomaly")
    parser.add_argument("--contamination", type=float, default=0.05)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df):,} records")

    detector = AnomalyDetector(args.output_dir)
    detector.run(df, args.contamination)


if __name__ == "__main__":
    main()