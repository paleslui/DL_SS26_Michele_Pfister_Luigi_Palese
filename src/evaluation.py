"""Metrics and bootstrap confidence intervals for binary classifiers.

The whole project compares MSI-H vs MSS, so every model produces:
    y_true : (n,) array of 0/1 labels (1 = MSI-H)
    y_prob : (n,) array of predicted probabilities for the MSI-H class

Convention everywhere: positive class = MSI-H = label 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)


# -------- single-prediction metrics ------------------------------------------

@dataclass
class Metrics:
    """Holds metrics for one set of predictions."""
    auc: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    threshold: float

    def as_dict(self) -> dict:
        return {
            "auc": self.auc, "f1": self.f1,
            "precision": self.precision, "recall": self.recall,
            "accuracy": self.accuracy, "threshold": self.threshold,
        }


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Metrics:
    """Compute all metrics from probabilities + a decision threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    return Metrics(
        auc=roc_auc_score(y_true, y_prob),
        f1=f1_score(y_true, y_pred, zero_division=0),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        accuracy=accuracy_score(y_true, y_pred),
        threshold=threshold,
    )


# -------- bootstrap confidence intervals -------------------------------------

def bootstrap_metric(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap a metric: returns (point_estimate, low_95, high_95).

    metric_fn(y_true_b, y_prob_b) -> float
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    point = metric_fn(y_true, y_prob)
    samples = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        # Skip degenerate resamples (only one class)
        if len(np.unique(y_true[idx])) < 2:
            samples[i] = np.nan
            continue
        samples[i] = metric_fn(y_true[idx], y_prob[idx])

    samples = samples[~np.isnan(samples)]
    low = np.percentile(samples, 2.5)
    high = np.percentile(samples, 97.5)
    return point, low, high


# -------- pooled cross-validation evaluation ---------------------------------

def evaluate_cv(
    y_true_per_fold: list[np.ndarray],
    y_prob_per_fold: list[np.ndarray],
    threshold: float = 0.5,
    bootstrap_resamples: int = 1000,
    seed: int = 42,
) -> dict:
    """Evaluate cross-validation results.

    Two complementary views:
      - per-fold: list of fold-level metrics (shows variance across folds)
      - pooled  : metrics computed on the concatenation of all val predictions,
                  with bootstrap CIs (more stable, used as the headline number)
    """
    per_fold = []
    for y_t, y_p in zip(y_true_per_fold, y_prob_per_fold):
        per_fold.append(compute_metrics(y_t, y_p, threshold).as_dict())

    y_true_pooled = np.concatenate(y_true_per_fold)
    y_prob_pooled = np.concatenate(y_prob_per_fold)
    pooled_point = compute_metrics(y_true_pooled, y_prob_pooled, threshold).as_dict()

    auc_p, auc_lo, auc_hi = bootstrap_metric(
        y_true_pooled, y_prob_pooled, roc_auc_score,
        n_resamples=bootstrap_resamples, seed=seed,
    )
    f1_fn = lambda yt, yp: f1_score(yt, (yp >= threshold).astype(int), zero_division=0)
    f1_p, f1_lo, f1_hi = bootstrap_metric(
        y_true_pooled, y_prob_pooled, f1_fn,
        n_resamples=bootstrap_resamples, seed=seed,
    )

    cm = confusion_matrix(y_true_pooled, (y_prob_pooled >= threshold).astype(int))

    return {
        "per_fold": per_fold,
        "pooled": pooled_point,
        "ci": {
            "auc":  {"point": auc_p, "low": auc_lo, "high": auc_hi},
            "f1":   {"point": f1_p,  "low": f1_lo,  "high": f1_hi},
        },
        "confusion_matrix": cm.tolist(),
        "n_pooled": int(len(y_true_pooled)),
    }


def format_results(results: dict, model_name: str = "model") -> str:
    """Pretty-print a results dict from evaluate_cv."""
    auc = results["ci"]["auc"]
    f1 = results["ci"]["f1"]
    cm = results["confusion_matrix"]
    lines = [
        f"=== {model_name} ===",
        f"AUC  : {auc['point']:.3f}  [95% CI {auc['low']:.3f} – {auc['high']:.3f}]",
        f"F1   : {f1['point']:.3f}  [95% CI {f1['low']:.3f} – {f1['high']:.3f}]",
        f"Acc  : {results['pooled']['accuracy']:.3f}",
        f"Confusion matrix (rows=true [MSS, MSI-H], cols=pred):",
        f"  MSS  : {cm[0][0]:>4}  {cm[0][1]:>4}",
        f"  MSI-H: {cm[1][0]:>4}  {cm[1][1]:>4}",
        f"Per-fold AUC: " + ", ".join(f"{f['auc']:.3f}" for f in results['per_fold']),
    ]
    return "\n".join(lines)
