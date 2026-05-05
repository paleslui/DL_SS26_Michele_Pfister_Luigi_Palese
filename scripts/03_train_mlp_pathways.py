"""Model 2: MLP on MSigDB Hallmark pathway scores.

Each tumor is summarized by 50 numbers — one mean z-scored expression value
per Hallmark pathway. A small MLP then predicts MSI-H vs MSS from these.

Tests Pfister's "activation, not abundance" hypothesis: pathway scores capture
biological activity states that EPIC's cell-fraction outputs cannot.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loading import load_aligned_dataset, REPO_ROOT
from src.splits import load_splits
from src.preprocessing import preprocess_for_fold
from src.pathways import load_gmt, compute_pathway_scores
from src.models.mlp import MLPClassifier
from src.training import TrainConfig, train_one_fold, best_device
from src.evaluation import evaluate_cv, format_results


MODEL_NAME = "02_mlp_pathways"
RESULTS_DIR = REPO_ROOT / "results"


def main() -> None:
    print("Loading data...")
    tpm, _, labels = load_aligned_dataset()
    folds = load_splits()
    gene_sets = load_gmt()
    print(f"  {len(labels)} samples, {tpm.shape[0]} genes, {len(gene_sets)} pathways, {len(folds)} folds")

    y_all = (labels == "MSI-H").astype(int)
    device = best_device()
    print(f"  device: {device}")

    config = TrainConfig(
        epochs=200,
        batch_size=64,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=30,
        verbose=False,
    )

    y_true_per_fold, y_prob_per_fold = [], []
    fold_histories = []

    print(f"\nTraining {MODEL_NAME} across folds...")
    for fold in folds:
        train_ids, val_ids = fold["train"], fold["val"]
        y_train = y_all.loc[train_ids].to_numpy()
        y_val = y_all.loc[val_ids].to_numpy()

        # Per-fold preprocessing: log + filter + z-score, fit on train only
        train_z, val_z = preprocess_for_fold(tpm, train_ids, val_ids)

        # Compute pathway scores for both train and val using the SAME z-scored gene panel
        # (means/stds were already fit on train inside preprocess_for_fold)
        train_path = compute_pathway_scores(train_z, gene_sets)  # (n_train, n_pathways)
        val_path = compute_pathway_scores(val_z, gene_sets)

        # Align column order between train and val (must match for the model)
        common_paths = [p for p in train_path.columns if p in val_path.columns]
        train_path = train_path[common_paths]
        val_path = val_path[common_paths]

        # Reorder rows to match the y arrays
        X_train = train_path.loc[train_ids].to_numpy()
        X_val = val_path.loc[val_ids].to_numpy()

        n_features = X_train.shape[1]
        model = MLPClassifier(input_dim=n_features, hidden_dims=(64, 32), dropout=0.5)

        y_prob, history = train_one_fold(
            model, X_train, y_train, X_val, y_val,
            config=config, device=device,
        )

        y_true_per_fold.append(y_val)
        y_prob_per_fold.append(y_prob)
        fold_histories.append(history)

        print(f"  fold {fold['fold']}: best epoch={history['best_epoch']:>3}, "
              f"best val AUC={history['best_val_auc']:.3f}, n_features={n_features}")

    print("\nEvaluating...")
    results = evaluate_cv(y_true_per_fold, y_prob_per_fold)
    print()
    print(format_results(results, MODEL_NAME))

    RESULTS_DIR.mkdir(exist_ok=True)
    results["model_name"] = MODEL_NAME
    results["config"] = config.__dict__
    with (RESULTS_DIR / f"{MODEL_NAME}.json").open("w") as f:
        json.dump(results, f, indent=2)
    np.savez(
        RESULTS_DIR / f"{MODEL_NAME}_predictions.npz",
        y_true_per_fold=np.array(y_true_per_fold, dtype=object),
        y_prob_per_fold=np.array(y_prob_per_fold, dtype=object),
        allow_pickle=True,
    )
    print(f"\nResults saved.")


if __name__ == "__main__":
    main()
