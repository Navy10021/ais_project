"""
Model Evaluator
============
Comprehensive model evaluation and reporting utilities.

Metrics:
  - AUROC, AUPRC
  - F1, F2, F-beta
  - Precision, Recall
  - Confusion Matrix
  - Calibration curves

Coding Conventions:
  - Type hints
  - Logging via logger
  - Reproducibility with SEED
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from pathlib import Path
import logging
import yaml
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)


@dataclass
class EvalConfig:
    """Configuration for evaluation"""
    f_beta: float = 2.0
    threshold: float = 0.5


def load_config(config_path: str = "./config/settings.yaml") -> EvalConfig:
    """Load evaluation config"""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return EvalConfig(
            f_beta=config.get("models", {}).get("eval_f_beta", 2.0),
            threshold=config.get("models", {}).get("threshold", 0.5),
        )
    except Exception:
        return EvalConfig()


class ModelEvaluator:
    """Comprehensive model evaluation"""

    def __init__(
        self,
        output_dir: str = "./outputs/models/evaluator",
        config: Optional[EvalConfig] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or EvalConfig()

    def compute_auroc_auprc(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Compute AUROC and AUPRC with curves"""
        has_positives = len(np.unique(y_true)) > 1

        auroc = roc_auc_score(y_true, y_proba) if has_positives else 0.0
        auprc = average_precision_score(y_true, y_proba)

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        precision, recall, _ = precision_recall_curve(y_true, y_proba)

        return auroc, auprc, fpr, tpr, precision, recall

    def compute_f_scores(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Compute F-beta scores"""
        return {
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "f2": fbeta_score(y_true, y_pred, beta=self.config.f_beta, zero_division=0),
        }

    def compute_confusion(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, int]:
        """Compute confusion matrix elements"""
        cm = confusion_matrix(y_true, y_pred)

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
            if len(np.unique(y_true)) == 1:
                tn = len(y_true) - (y_true != y_pred[0]).sum()

        return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}

    def compute_sensitivity_specificity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Compute sensitivity and specificity"""
        cm = self.compute_confusion(y_true, y_pred)

        tp = cm["tp"]
        fn = cm["fn"]
        tn = cm["tn"]
        fp = cm["fp"]

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        return {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "ppv": ppv,
            "npv": npv,
        }

    def compute_calibration(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute calibration curve"""
        try:
            frac_pos, mean_pred = calibration_curve(
                y_true, y_proba, n_bins=n_bins, strategy="uniform"
            )
            brier = brier_score_loss(y_true, y_proba)
            return frac_pos, mean_pred, brier
        except Exception as e:
            logger.warning(f"Calibration computation failed: {e}")
            return np.array([]), np.array([]), 1.0

    def findOptimalThreshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        metric: str = "f2",
    ) -> float:
        """Find optimal classification threshold"""
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_score = -1
        best_thresh = 0.5

        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            if metric == "f2":
                score = fbeta_score(
                    y_true, y_pred, beta=2.0, zero_division=0
                )
            elif metric == "f1":
                score = f1_score(y_true, y_pred, zero_division=0)
            else:
                score = recall_score(y_true, y_pred, zero_division=0)

            if score > best_score:
                best_score = score
                best_thresh = thresh

        return best_thresh

    def evaluate_model(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str = "model",
    ) -> Dict[str, any]:
        """Comprehensive model evaluation"""
        y_pred = (y_proba >= self.config.threshold).astype(int)

        auroc, auprc, fpr, tpr, precision, recall = self.compute_auroc_auprc(
            y_true, y_proba
        )
        f_scores = self.compute_f_scores(y_true, y_pred)
        cm = self.compute_confusion(y_true, y_pred)
        sens_spec = self.compute_sensitivity_specificity(y_true, y_pred)
        frac_pos, mean_pred, brier = self.compute_calibration(y_true, y_proba)
        optimal_thresh = self.findOptimalThreshold(y_true, y_proba)

        results = {
            "model": model_name,
            "auroc": auroc,
            "auprc": auprc,
            "f1": f_scores["f1"],
            "f2": f_scores["f2"],
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "specificity": sens_spec["specificity"],
            "brier_score": brier,
            "optimal_threshold": optimal_thresh,
            **cm,
        }

        return results

    def compare_models(
        self,
        results: Dict[str, Dict[str, float]],
    ) -> pd.DataFrame:
        """Compare multiple model results"""
        rows = []

        for model_name, metrics in results.items():
            row = {"model": model_name}
            row.update(metrics)
            rows.append(row)

        df = pd.DataFrame(rows)

        for col in ["auroc", "auprc", "f1", "f2"]:
            if col in df.columns:
                df = df.sort_values(col, ascending=False)

        return df

    def generate_report(
        self,
        results: Dict[str, Dict[str, float]],
        output_path: Optional[Path] = None,
    ) -> str:
        """Generate text evaluation report"""
        output_path = output_path or (self.output_dir / "evaluation_report.txt")

        lines = ["=" * 60, "MODEL EVALUATION REPORT", "=" * 60, ""]

        for model_name, metrics in results.items():
            lines.append(f"Model: {model_name}")
            lines.append("-" * 40)
            lines.append(f"  AUROC:           {metrics.get('auroc', 0):.4f}")
            lines.append(f"  AUPRC:          {metrics.get('auprc', 0):.4f}")
            lines.append(f"  F1 Score:       {metrics.get('f1', 0):.4f}")
            lines.append(f"  F2 Score:       {metrics.get('f2', 0):.4f}")
            lines.append(f"  Precision:       {metrics.get('precision', 0):.4f}")
            lines.append(f"  Recall:          {metrics.get('recall', 0):.4f}")
            lines.append(f"  Specificity:    {metrics.get('specificity', 0):.4f}")
            lines.append(f"  Optimal Thresh:  {metrics.get('optimal_threshold', 0.5):.2f}")
            lines.append("")

            tp = metrics.get("tp", 0)
            fp = metrics.get("fp", 0)
            fn = metrics.get("fn", 0)
            tn = metrics.get("tn", 0)
            lines.append(f"  Conf. Matrix:   TP={tp}, FP={fp}, FN={fn}, TN={tn}")
            lines.append("")

        report = "\n".join(lines)

        with open(output_path, "w") as f:
            f.write(report)

        logger.info(f"Report saved: {output_path}")
        return report


class ThresholdOptimizer:
    """Optimize classification thresholds"""

    def __init__(self, config: Optional[EvalConfig] = None):
        self.config = config or EvalConfig()
        self.optimal_thresholds = {}

    def optimize_for_f2(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> float:
        """Find threshold that maximizes F2"""
        return self._find_optimal(y_true, y_proba, "f2")

    def optimize_for_precision(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> float:
        """Find threshold that maximizes precision"""
        return self._find_optimal(y_true, y_proba, "precision")

    def optimize_for_recall(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> float:
        """Find threshold that maximizes recall"""
        return self._find_optimal(y_true, y_proba, "recall")

    def optimize_all(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, float]:
        """Find optimal thresholds for all metrics"""
        self.optimal_thresholds = {
            "f2": self.optimize_for_f2(y_true, y_proba),
            "precision": self.optimize_for_precision(y_true, y_proba),
            "recall": self.optimize_for_recall(y_true, y_proba),
        }
        return self.optimal_thresholds

    def _find_optimal(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        metric: str,
    ) -> float:
        """Internal threshold optimization"""
        thresholds = np.arange(0.05, 0.95, 0.05)
        best_score = -1
        best_thresh = 0.5

        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)

            if metric == "f2":
                score = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)
            elif metric == "precision":
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == "recall":
                score = recall_score(y_true, y_pred, zero_division=0)
            else:
                score = f1_score(y_true, y_pred, zero_division=0)

            if score > best_score:
                best_score = score
                best_thresh = thresh

        return best_thresh


def evaluate_and_report(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "model",
    output_dir: str = "./outputs/models/evaluator",
) -> Dict[str, any]:
    """Convenience function for quick evaluation"""
    config = load_config()
    evaluator = ModelEvaluator(output_dir, config)
    results = evaluator.evaluate_model(y_true, y_proba, model_name)
    evaluator.generate_report({model_name: results})
    return results