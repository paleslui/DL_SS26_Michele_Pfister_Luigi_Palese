"""Model 1: Logistic regression on EPIC cell fractions.

This is our LINEAR BASELINE. It uses Pfister's exact features (the 8 EPIC-derived
cell-type fractions) and asks "is there ANY signal in cell counts?"

Expected behaviour: barely above chance (AUC ~0.5–0.6). If true, this confirms
Pfister's negative finding from a different angle and motivates the move to
non-linear, activation-level features in subsequent models.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loading import load_aligned_dataset, REPO_ROOT
from src.splits import load_splits
from src.evaluation import evaluate_cv, format_results


MODEL_NAME = "01_logreg_epic"
RESULTS_DIR = REPO_ROOT / "results"


def train_one_fold(
    epic_train: np.ndarray,
    y_train: np.ndarray,
    epic_val: np.ndarray,
) -> np.ndarray:
    """Fit logistic regression on one fold; return predicted probabilities for val."""
    # Scale features (small effect for logreg with regularization, but standard practice)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(epic_train)
    X_val = scaler.transform(epic_val)

    # L2-regularized logistic regression with class weighting for the 2:1 imbalance
    model = LogisticRegression(
        # sklearn 1.8+ deprecated penalty="l2"; l1_ratio=0.0 is equivalent
        l1_ratio=0.0,
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_val)[:, 1]


def main() -> None:
    print("Loading data...")
    _, epic, labels = load_aligned_dataset()
    folds = load_splits()
    print(f"  {len(labels)} samples, {epic.shape[1]} EPIC features, {len(folds)} folds")

    y_all = (labels == "MSI-H").astype(int)

    y_true_per_fold = []
    y_prob_per_fold = []

    print(f"\nTraining {MODEL_NAME} across folds...")
    for fold in folds:
        train_ids, val_ids = fold["train"], fold["val"]
        X_train = epic.loc[train_ids].to_numpy()
        X_val = epic.loc[val_ids].to_numpy()
        y_train = y_all.loc[train_ids].to_numpy()
        y_val = y_all.loc[val_ids].to_numpy()

        y_prob = train_one_fold(X_train, y_train, X_val)
        y_true_per_fold.append(y_val)
        y_prob_per_fold.append(y_prob)

        fold_auc = float(np.mean(y_prob[y_val == 1]) > np.mean(y_prob[y_val == 0]))
        print(f"  fold {fold['fold']}: val n={len(y_val)}, MSI-H frac={y_val.mean():.3f}")

    print("\nEvaluating...")
    results = evaluate_cv(y_true_per_fold, y_prob_per_fold)
    print()
    print(format_results(results, MODEL_NAME))

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{MODEL_NAME}.json"
    # Convert numpy arrays in confusion matrix etc. to lists, already done in evaluate_cv
    results["model_name"] = MODEL_NAME
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)

    # Also save raw predictions (useful later for combining results, plotting ROC curves)
    np.savez(
        RESULTS_DIR / f"{MODEL_NAME}_predictions.npz",
        y_true_per_fold=np.array(y_true_per_fold, dtype=object),
        y_prob_per_fold=np.array(y_prob_per_fold, dtype=object),
        allow_pickle=True,
    )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
