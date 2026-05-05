"""Model 4: Gene-attention transformer on a curated MSI/immune panel.

Each gene is a token; self-attention learns gene-to-gene relationships
without imposing any order. Attention weights from the [CLS] token are
directly interpretable as "which genes did the model use?"
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
from src.preprocessing import log_transform, fit_zscore_params, apply_zscore
from src.gene_panels import build_panel
from src.models.gene_transformer import GeneTransformer
from src.training import TrainConfig, train_one_fold, best_device
from src.evaluation import evaluate_cv, format_results


MODEL_NAME = "04_transformer_panel"
RESULTS_DIR = REPO_ROOT / "results"


def preprocess_for_fold_panel(
    tpm: pd.DataFrame,
    train_ids: list,
    val_ids: list,
    panel: list,
) -> tuple[np.ndarray, np.ndarray]:
    """log + train-only z-score, restricted to the panel genes."""
    log_all = log_transform(tpm)
    log_panel = log_all.loc[panel]

    log_train = log_panel[train_ids]
    means, stds = fit_zscore_params(log_train)
    train_z = apply_zscore(log_train, means, stds)

    log_val = log_panel[val_ids]
    val_z = apply_zscore(log_val, means, stds)

    # (n_samples, n_genes)
    return train_z[train_ids].to_numpy().T, val_z[val_ids].to_numpy().T


def main() -> None:
    print("Loading data...")
    tpm, _, labels = load_aligned_dataset()
    folds = load_splits()
    panel = build_panel(set(tpm.index))
    print(f"  {len(labels)} samples, {len(panel)}-gene panel, {len(folds)} folds")

    y_all = (labels == "MSI-H").astype(int)
    device = best_device()
    print(f"  device: {device}")

    config = TrainConfig(
        epochs=100,
        batch_size=16,
        learning_rate=3e-4,
        weight_decay=1e-3,
        patience=20,
        min_epochs=15,
        verbose=False,
    )

    y_true_per_fold, y_prob_per_fold, fold_histories = [], [], []

    print(f"\nTraining {MODEL_NAME} across folds...")
    for fold in folds:
        train_ids, val_ids = fold["train"], fold["val"]
        y_train = y_all.loc[train_ids].to_numpy()
        y_val = y_all.loc[val_ids].to_numpy()

        X_train, X_val = preprocess_for_fold_panel(tpm, train_ids, val_ids, panel)

        model = GeneTransformer(
            n_genes=len(panel),
            d_model=64, n_heads=4, n_layers=2,
            dim_feedforward=128, dropout=0.3,
        )

        y_prob, history = train_one_fold(
            model, X_train, y_train, X_val, y_val,
            config=config, device=device,
        )

        y_true_per_fold.append(y_val)
        y_prob_per_fold.append(y_prob)
        fold_histories.append(history)

        print(f"  fold {fold['fold']}: best epoch={history['best_epoch']:>3}, "
              f"best val AUC={history['best_val_auc']:.3f}, n_genes={len(panel)}")

    print("\nEvaluating...")
    results = evaluate_cv(y_true_per_fold, y_prob_per_fold)
    print()
    print(format_results(results, MODEL_NAME))

    RESULTS_DIR.mkdir(exist_ok=True)
    results["model_name"] = MODEL_NAME
    results["config"] = config.__dict__
    results["panel_size"] = len(panel)
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
