"""
Conflict Prediction Model
========================
Supervised learning for conflict prediction.

Models:
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - Logistic Regression

Evaluation:
  - AUROC, AUPRC, F1, F2
  - Cross-validation

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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
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
class PredictionConfig:
    """Configuration for prediction models"""
    test_size: float = 0.2
    n_estimators: int = 100
    max_depth: int = 10
    cv_folds: int = 5


def load_config(config_path: str = "./config/settings.yaml") -> PredictionConfig:
    """Load prediction config from YAML"""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        m = config.get("models", {}).get("prediction", {})
        return PredictionConfig(
            test_size=m.get("test_size", 0.2),
            n_estimators=m.get("n_estimators", 100),
            max_depth=m.get("max_depth", 10),
            cv_folds=m.get("cv_folds", 5),
        )
    except Exception:
        logger.warning(
            f"Could not load config from {config_path}, using defaults"
        )
        return PredictionConfig()


class ConflictPredictor:
    """Supervised conflict prediction model"""

    FEATURE_COLS: List[str] = [
        "SOG",
        "delta_sog",
        "delta_cog",
        "turning_rate",
        "rolling_sog_mean_12h",
        "rolling_sog_std_12h",
        "is_dark_ship",
        "loitering_flag",
        "zig_zag_index",
        "in_conflict_zone",
        "traffic_count",
        "dist_hormuz_km",
        "dist_malacca_km",
    ]

    def __init__(
        self,
        output_dir: str = "./outputs/models/predictor",
        config: Optional[PredictionConfig] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.models: Dict[str, any] = {}
        self.results: Dict[str, any] = {}
        self.config = config or PredictionConfig()

    def prepare_data(
        self,
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for training"""
        features = [col for col in self.FEATURE_COLS if col in df.columns]

        X = df[features].copy()
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)

        y = (
            df["conflict_label"].copy()
            if "conflict_label" in df.columns
            else pd.Series(0, index=df.index)
        )
        y = y.fillna(0).astype(int)

        return X.values, y.values

    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> "RandomForestClassifier":
        """Train Random Forest classifier"""
        logger.info("Training Random Forest...")

        model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        self.models["random_forest"] = model
        return model

    def train_gradient_boosting(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> "GradientBoostingClassifier":
        """Train Gradient Boosting classifier"""
        logger.info("Training Gradient Boosting...")

        model = GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth // 2,
            learning_rate=0.1,
            random_state=SEED,
        )
        model.fit(X_train, y_train)
        self.models["gradient_boosting"] = model
        return model

    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Optional[any]:
        """Train XGBoost classifier if available"""
        try:
            from xgboost import XGBClassifier

            logger.info("Training XGBoost...")

            model = XGBClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=0.1,
                scale_pos_weight=sum(y_train == 0) / max(1, sum(y_train == 1)),
                random_state=SEED,
                use_label_encoder=False,
                eval_metric="logloss",
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            self.models["xgboost"] = model
            return model
        except ImportError:
            logger.warning("XGBoost not available, skipping...")
            return None

    def train_logistic_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> "LogisticRegression":
        """Train Logistic Regression"""
        logger.info("Training Logistic Regression...")

        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=SEED,
        )
        model.fit(X_train, y_train)
        self.models["logistic_regression"] = model
        return model

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate all models"""
        results = {}

        for name, model in self.models.items():
            try:
                y_pred = model.predict(X_test)
                y_proba = (
                    model.predict_proba(X_test)[:, 1]
                    if hasattr(model, "predict_proba")
                    else y_pred
                )

                has_positives = len(np.unique(y_test)) > 1
                auroc = (
                    roc_auc_score(y_test, y_proba) if has_positives else 0.0
                )
                auprc = average_precision_score(y_test, y_proba)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                f2 = (
                    5 * recall_score(y_test, y_pred, zero_division=0) * precision_score(y_test, y_pred, zero_division=0)
                    / (4 * precision_score(y_test, y_pred, zero_division=0) + recall_score(y_test, y_pred, zero_division=0) + 1e-10)
                )

                results[name] = {
                    "auroc": auroc,
                    "auprc": auprc,
                    "f1": f1,
                    "f2": f2,
                    "precision": precision_score(y_test, y_pred, zero_division=0),
                    "recall": recall_score(y_test, y_pred, zero_division=0),
                    "accuracy": (y_pred == y_test).mean(),
                }

                cm = confusion_matrix(y_test, y_pred)
                if cm.shape == (2, 2):
                    results[name]["tn"] = cm[0, 0]
                    results[name]["fp"] = cm[0, 1]
                    results[name]["fn"] = cm[1, 0]
                    results[name]["tp"] = cm[1, 1]

            except Exception as e:
                logger.warning(f"Evaluation failed for {name}: {e}")

        return results

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
    ) -> Dict[str, float]:
        """Perform cross-validation"""
        cv_results = {}
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)

        for name, model in self.models.items():
            if not hasattr(model, "predict_proba"):
                continue

            try:
                scores = cross_val_score(
                    model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1
                )
                cv_results[name] = {
                    "cv_mean": scores.mean(),
                    "cv_std": scores.std(),
                }
                logger.info(f"{name} CV AUROC: {scores.mean():.3f} (+/- {scores.std():.3f})")
            except Exception as e:
                logger.warning(f"CV failed for {name}: {e}")

        return cv_results

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from all models"""
        importances = {}

        for name, model in self.models.items():
            if hasattr(model, "feature_importances_"):
                importances[name] = model.feature_importances_

        if not importances:
            return pd.DataFrame()

        df = pd.DataFrame(importances)
        features = [col for col in self.FEATURE_COLS if col in df.index or col in df.columns]
        if len(df) == len(features):
            df.index = features

        if not df.empty:
            df["mean"] = df.mean(axis=1)
            df = df.sort_values("mean", ascending=False)

        return df

    def run(
        self,
        df: pd.DataFrame,
        test_size: Optional[float] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Run full prediction pipeline"""
        logger.info("Running conflict prediction...")

        X, y = self.prepare_data(df)

        if y.sum() == 0:
            logger.warning("No positive labels found. Using synthetic labels.")
            X_df = pd.DataFrame(X, columns=[c for c in self.FEATURE_COLS if c in df.columns])
            y = (X_df["SOG"] > X_df["SOG"].quantile(0.9)).astype(int).values
            y = y | (X_df["is_dark_ship"] > 0).astype(int).values
            logger.info(f"Synthetic labels: {y.sum()} positives")

        X_scaled = self.scaler.fit_transform(X)

        test_size = test_size or self.config.test_size
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=test_size,
            random_state=SEED,
            stratify=y if y.sum() > 1 else None,
        )

        self.train_random_forest(X_train, y_train)
        self.train_gradient_boosting(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        self.train_logistic_regression(X_train, y_train)

        eval_results = self.evaluate(X_test, y_test)
        self.results = eval_results

        for name, metrics in eval_results.items():
            logger.info(
                f"{name}: AUROC={metrics['auroc']:.3f}, "
                f"F1={metrics['f1']:.3f}, F2={metrics['f2']:.3f}"
            )

        cv_results = self.cross_validate(X_scaled, y, cv=self.config.cv_folds)
        self.results["cv"] = cv_results

        importance = self.get_feature_importance()
        if not importance.empty:
            importance.to_csv(self.output_dir / "feature_importance.csv")
            logger.info(f"Saved: {self.output_dir / 'feature_importance.csv'}")

        pd.DataFrame(eval_results).T.to_csv(
            self.output_dir / "evaluation_results.csv"
        )
        logger.info(f"Saved: {self.output_dir / 'evaluation_results.csv'}")

        self.save_models()

        logger.info(f"Prediction complete! Results: {self.output_dir}")
        return self.results

    def save_models(self) -> None:
        """Save trained models"""
        for name, model in self.models.items():
            path = self.output_dir / f"{name}.joblib"
            joblib.dump(model, path)
            logger.info(f"Model saved: {path}")

        scaler_path = self.output_dir / "scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Scaler saved: {scaler_path}")

    def load_models(self) -> None:
        """Load trained models"""
        for path in self.output_dir.glob("*.joblib"):
            name = path.stem
            if name == "scaler":
                self.scaler = joblib.load(path)
            else:
                self.models[name] = joblib.load(path)
            logger.info(f"Loaded: {path}")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for new data"""
        if not self.models:
            self.load_models()

        X, _ = self.prepare_data(df)
        X_scaled = self.scaler.transform(X)

        results = pd.DataFrame(index=df.index)

        for name, model in self.models.items():
            if hasattr(model, "predict_proba"):
                results[f"{name}_proba"] = model.predict_proba(X_scaled)[:, 1]
                results[f"{name}_pred"] = model.predict(X_scaled)

        proba_cols = results.filter(like="proba").columns
        if len(proba_cols) > 0:
            results["ensemble_proba"] = results[proba_cols].mean(axis=1)
            results["ensemble_pred"] = (results["ensemble_proba"] > 0.5).astype(int)

        return results


def main():
    parser = argparse.ArgumentParser(description="Conflict Prediction Model")
    parser.add_argument("--input", required=True, help="Input Parquet path")
    parser.add_argument(
        "--output-dir",
        default="./outputs/models/predictor",
        help="Output directory",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set proportion",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df):,} records")

    config = load_config()
    predictor = ConflictPredictor(args.output_dir, config)
    predictor.run(df, args.test_size)


if __name__ == "__main__":
    main()