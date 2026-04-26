"""
Baseline Models
==============
Simple statistical baseline models for comparison.

Models:
  - ARIMA for time series forecasting
  - Simple moving average
  - Exponential smoothing
  - Naive predictors

Coding Conventions:
  - Type hints
  - Logging via logger
  - Reproducibility with SEED
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from pathlib import Path
import logging
import argparse
import yaml
from dataclasses import dataclass
from typing import Optional, List

logger = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)


@dataclass
class BaselineConfig:
    """Configuration for baselines"""
    window_size: int = 7
    smoothing_factor: float = 0.3


def load_config(config_path: str = "./config/settings.yaml") -> BaselineConfig:
    """Load baseline config"""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return BaselineConfig(
            window_size=config.get("models", {}).get("baseline", {}).get("window_size", 7),
            smoothing_factor=config.get("models", {}).get("baseline", {}).get("smoothing_factor", 0.3),
        )
    except Exception:
        return BaselineConfig()


class SimpleMovingAverage(BaseEstimator, ClassifierMixin):
    """Simple moving average baseline"""

    def __init__(self, window: int = 7):
        self.window = window

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.mean_ = X[: self.window].mean() if len(X) >= self.window else X.mean()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.mean_)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pred = self.predict(X)
        proba = np.zeros((len(X), 2))
        proba[:, 1] = np.clip(pred, 0, 1)
        proba[:, 0] = 1 - proba[:, 1]
        return proba


class ExponentialSmoothing(BaseEstimator, ClassifierMixin):
    """Exponential smoothing baseline"""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.last_ = X[0] if len(X) > 0 else 0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        result = np.zeros(len(X))
        result[0] = self.last_
        for i in range(1, len(X)):
            result[i] = self.alpha * result[i - 1] + (1 - self.alpha) * X[i]
        return result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pred = self.predict(X)
        proba = np.zeros((len(X), 2))
        proba[:, 1] = np.clip(pred, 0, 1)
        proba[:, 0] = 1 - proba[:, 1]
        return proba


class NaiveRegressor(BaseEstimator, ClassifierMixin):
    """Naive baseline - predict last value"""

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.last_value_ = X[-1] if len(X) > 0 else 0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.last_value_)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pred = self.predict(X)
        proba = np.zeros((len(X), 2))
        proba[:, 1] = np.clip(pred, 0, 1)
        proba[:, 0] = 1 - proba[:, 1]
        return proba


class TrendBaseline(BaseEstimator, ClassifierMixin):
    """Linear trend baseline"""

    def __init__(self, lag: int = 7):
        self.lag = lag

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        if len(X) >= self.lag:
            self.coef_ = (X[-1] - X[0]) / len(X)
            self.intercept_ = X[0]
        else:
            self.coef_ = 0
            self.intercept_ = X.mean() if len(X) > 0 else 0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        t = np.arange(len(X))
        return self.intercept_ + self.coef_ * t

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pred = self.predict(X)
        proba = np.zeros((len(X), 2))
        proba[:, 1] = np.clip(pred, 0, 1)
        proba[:, 0] = 1 - proba[:, 1]
        return proba


class BaselineModel:
    """Collection of baseline models"""

    def __init__(
        self,
        output_dir: str = "./outputs/models/baseline",
        config: Optional[BaselineConfig] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or BaselineConfig()
        self.models = {}

    def fit_all(self, X: np.ndarray, y: np.ndarray) -> "BaselineModel":
        """Fit all baseline models"""
        logger.info("Fitting baseline models...")

        self.models["sma"] = SimpleMovingAverage(self.config.window_size)
        self.models["sma"].fit(X)

        self.models["exp"] = ExponentialSmoothing(self.config.smoothing_factor)
        self.models["exp"].fit(X)

        self.models["naive"] = NaiveRegressor()
        self.models["naive"].fit(X)

        self.models["trend"] = TrendBaseline(self.config.window_size)
        self.models["trend"].fit(X)

        return self

    def predict_all(self, X: np.ndarray) -> pd.DataFrame:
        """Get predictions from all baselines"""
        results = pd.DataFrame()

        for name, model in self.models.items():
            results[name] = model.predict(X)

        return results

    def evaluate_baselines(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
    ) -> pd.DataFrame:
        """Evaluate baseline models"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error

        predictions = self.predict_all(X)
        metrics = []

        for name in predictions.columns:
            y_pred = predictions[name]
            metrics.append({
                "model": name,
                "mse": mean_squared_error(y_true, y_pred),
                "mae": mean_absolute_error(y_true, y_pred),
            })

        return pd.DataFrame(metrics)


def main():
    parser = argparse.ArgumentParser(description="Baseline Models")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="./outputs/models/baseline")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df):,} records")

    X = df[["SOG"]].fillna(0).values.flatten()
    y = df[["conflict_label"]].fillna(0).values.flatten() if "conflict_label" in df.columns else X

    config = load_config()
    baseline = BaselineModel(args.output_dir, config)
    baseline.fit_all(X, y)

    results = baseline.evaluate_baselines(X, y[:len(X)])
    print(results)

    results.to_csv(args.output_dir + "/baseline_results.csv", index=False)


if __name__ == "__main__":
    main()