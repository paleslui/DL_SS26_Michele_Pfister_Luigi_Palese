"""Model 3: 1D CNN on chromosome-ordered gene expression.

Genes are ordered by (chromosome, start position). The top-N most-variable
genes (selected on training fold only) are kept, preserving genomic order.
A 1D convolutional network then looks for local patterns along the genome
axis — capturing co-expression of nearby genes, copy-number block effects,
and other spatial genomic structure.
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
from src.preprocessing import log_transform, filter_low_expression, fit_zscore_params, apply_zscore
from src.genome_ordering import order_genes_by_genome
from src.models.cnn1d import CNN1D
from src.training import TrainConfig, train_one_fold, best_device
from src.evaluation import evaluate_cv, format_results


MODEL_NAME = "03_cnn_chromorder"
RESULTS_DIR = REPO_ROOT / "results"
N_TOP_VARIABLE = 5000   # subset to top-N variable genes (chosen on train), preserving genomic order


def preprocess_for_fold_genome_ordered(
    tpm: pd.DataFrame,
    train_ids: list,
    val_ids: list,
    n_top: int,
    genome_order: list,
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """log + filter + train-only z-score + top-N variable, preserving genomic order.

    Returns (X_train, X_val, kept_genes) where the gene axis is in genome order.
    """
    log_all = log_transform(tpm)
    log_train = log_all[train_ids]

    # Filter low-expression on train
    log_train = filter_low_expression(log_train)

    # Restrict to genes that have a known genomic position (and apply that order)
    in_order = [g for g in genome_order if g in log_train.index]
    log_train = log_train.loc[in_order]

    # Pick top-N most variable on train, preserving order
    variances = log_train.var(axis=1)
    top_set = set(variances.sort_values(ascending=False).head(n_top).index)
    keep = [g for g in in_order if g in top_set]
    log_train = log_train.loc[keep]

    # Train-only z-score, then apply to val
    means, stds = fit_zscore_params(log_train)
    train_z = apply_zscore(log_train, means, stds)

    log_val = log_all.loc[keep, val_ids]
    val_z = apply_zscore(log_val, means, stds)

    return train_z, val_z, keep


def main() -> None:
    print("Loading data...")
    tpm, _, labels = load_aligned_dataset()
    folds = load_splits()
    genome_order = order_genes_by_genome(tpm.index.tolist())
    print(f"  {len(labels)} samples, {len(genome_order)} genes with genomic positions")
    print(f"  will subset to top-{N_TOP_VARIABLE} variable genes per fold")

    y_all = (labels == "MSI-H").astype(int)
    device = best_device()
    print(f"  device: {device}")

    config = TrainConfig(
        epochs=120,
        batch_size=32,
        learning_rate=5e-4,
        weight_decay=1e-3,
        patience=20,
        verbose=False,
    )

    y_true_per_fold, y_prob_per_fold, fold_histories = [], [], []

    print(f"\nTraining {MODEL_NAME} across folds...")
    for fold in folds:
        train_ids, val_ids = fold["train"], fold["val"]
        y_train = y_all.loc[train_ids].to_numpy()
        y_val = y_all.loc[val_ids].to_numpy()

        train_z, val_z, kept_genes = preprocess_for_fold_genome_ordered(
            tpm, train_ids, val_ids, n_top=N_TOP_VARIABLE, genome_order=genome_order,
        )

        # Convert to numpy: (n_samples, n_genes)
        X_train = train_z[train_ids].to_numpy().T
        X_val = val_z[val_ids].to_numpy().T

        n_genes = X_train.shape[1]
        model = CNN1D(n_genes=n_genes)

        y_prob, history = train_one_fold(
            model, X_train, y_train, X_val, y_val,
            config=config, device=device,
        )

        y_true_per_fold.append(y_val)
        y_prob_per_fold.append(y_prob)
        fold_histories.append(history)

        print(f"  fold {fold['fold']}: best epoch={history['best_epoch']:>3}, "
              f"best val AUC={history['best_val_auc']:.3f}, n_genes={n_genes}")

    print("\nEvaluating...")
    results = evaluate_cv(y_true_per_fold, y_prob_per_fold)
    print()
    print(format_results(results, MODEL_NAME))

    RESULTS_DIR.mkdir(exist_ok=True)
    results["model_name"] = MODEL_NAME
    results["config"] = config.__dict__
    results["n_top_variable"] = N_TOP_VARIABLE
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
