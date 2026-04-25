"""
Conflict Prediction Model
=======================
Supervised learning for conflict prediction.

Coding Conventions:
  - Type hints
  - Logging via logger
  - Config via YAML
  - Reproducibility with SEED
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    f1_score, classification_report, confusion_matrix
)
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
class PredictionConfig:
    test_size: float = 0.2
    n_estimators: int = 100
    max_depth: int = 10


def load_config(config_path: str = "./config/settings.yaml") -> PredictionConfig:
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        m = config.get("models", {}).get("prediction", {})
        return PredictionConfig(
            test_size=m.get("test_size", 0.2),
            n_estimators=m.get("n_estimators", 100),
            max_depth=m.get("max_depth", 10),
        )
    except Exception:
        return PredictionConfig()


class ConflictPredictor:
    FEATURE_COLS = [
        'SOG', 'delta_sog', 'delta_cog', 'turning_rate',
        'rolling_sog_mean_12h', 'rolling_sog_std_12h',
        'is_dark_ship', 'loitering_flag', 'zig_zag_index',
        'in_conflict_zone', 'traffic_count', 'dist_hormuz_km', 'dist_malacca_km'
    ]

    def __init__(self, output_dir: str = "./outputs/models/predictor", config: PredictionConfig = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.config = config or load_config()

    def prepare_data(self, df: pd.DataFrame):
        features = []
        for col in self.FEATURE_COLS:
            if col in df.columns:
                features.append(col)

        X = df[features].copy()
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)

        y = df['conflict_label'].copy() if 'conflict_label' in df.columns else pd.Series(0, index=df.index)

        return X, y

    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        logger.info("Training Random Forest...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=SEED,
            n_jobs=-1,
            **kwargs
        )
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        return model

    def train_gradient_boosting(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        logger.info("Training Gradient Boosting...")
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=SEED,
            **kwargs
        )
        model.fit(X_train, y_train)
        self.models['gradient_boosting'] = model
        return model

    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray):
        logger.info("Training Logistic Regression...")
        model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=SEED
        )
        model.fit(X_train, y_train)
        self.models['logistic_regression'] = model
        return model

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        results = {}

        for name, model in self.models.items():
            try:
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

                results[name] = {
                    'auroc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0,
                    'auprc': average_precision_score(y_test, y_proba),
                    'f1': f1_score(y_test, y_pred, zero_division=0),
                    'accuracy': (y_pred == y_test).mean(),
                }

                cm = confusion_matrix(y_test, y_pred)
                results[name]['tn'] = cm[0, 0] if cm.shape == (2, 2) else 0
                results[name]['fp'] = cm[0, 1] if cm.shape == (2, 2) else 0
                results[name]['fn'] = cm[1, 0] if cm.shape == (2, 2) else 0
                results[name]['tp'] = cm[1, 1] if cm.shape == (2, 2) else 0

            except Exception as e:
                logger.warning(f"Evaluation failed for {name}: {e}")

        return results

    def get_feature_importance(self) -> pd.DataFrame:
        importance_df = pd.DataFrame()

        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_df[name] = model.feature_importances_

        if not importance_df.empty and len(self.FEATURE_COLS) == len(importance_df):
            importance_df.index = self.FEATURE_COLS
            importance_df = importance_df.mean(axis=1).sort_values(ascending=False)

        return importance_df

    def run(self, df: pd.DataFrame, test_size: float = 0.2):
        logger.info("Running conflict prediction...")

        X, y = self.prepare_data(df)

        if y.sum() == 0:
            logger.warning("No positive labels found. Using synthetic labels for demonstration.")
            y = (X['SOG'] > X['SOG'].quantile(0.9)).astype(int)
            y = y | (X['is_dark_ship'] > 0).astype(int)
            logger.info(f"Using synthetic labels: {y.sum()} positives")

        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=SEED, stratify=y
        )

        self.train_random_forest(X_train, y_train)
        self.train_gradient_boosting(X_train, y_train)
        self.train_logistic_regression(X_train, y_train)

        eval_results = self.evaluate(X_test, y_test)
        self.results = eval_results

        for name, metrics in eval_results.items():
            logger.info(f"{name}: AUROC={metrics['auroc']:.3f}, F1={metrics['f1']:.3f}")

        importance = self.get_feature_importance()
        if not importance.empty:
            importance.to_csv(self.output_dir / 'feature_importance.csv')
            logger.info(f"Saved: {self.output_dir / 'feature_importance.csv'}")

        pd.DataFrame(eval_results).T.to_csv(self.output_dir / 'evaluation_results.csv')
        logger.info(f"Saved: {self.output_dir / 'evaluation_results.csv'}")

        self.save_models()

        logger.info(f"Prediction complete! Results: {self.output_dir}")
        return self.results

    def save_models(self):
        for name, model in self.models.items():
            path = self.output_dir / f'{name}.joblib'
            joblib.dump(model, path)
            logger.info(f"Model saved: {path}")

        path = self.output_dir / 'scaler.joblib'
        joblib.dump(self.scaler, path)
        logger.info(f"Scaler saved: {path}")

    def load_models(self):
        for path in self.output_dir.glob('*.joblib'):
            name = path.stem
            if name == 'scaler':
                self.scaler = joblib.load(path)
            else:
                self.models[name] = joblib.load(path)
            logger.info(f"Loaded: {path}")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.models:
            self.load_models()

        X, _ = self.prepare_data(df)
        X_scaled = self.scaler.transform(X)

        results = pd.DataFrame(index=df.index)

        for name, model in self.models.items():
            results[f'{name}_proba'] = model.predict_proba(X_scaled)[:, 1]
            results[f'{name}_pred'] = model.predict(X_scaled)

        results['ensemble_proba'] = results.filter(like='proba').mean(axis=1)
        results['ensemble_pred'] = (results['ensemble_proba'] > 0.5).astype(int)

        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="./outputs/models/predictor")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df):,} records")

    predictor = ConflictPredictor(args.output_dir)
    predictor.run(df, args.test_size)


if __name__ == "__main__":
    main()