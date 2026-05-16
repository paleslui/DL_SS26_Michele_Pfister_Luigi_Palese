"""Final clean evaluation on the 20% held-out test set.

For each of the 5 models we use the best hyperparameters from the v2 search
(or from the original training scripts for the un-tunable logreg baseline),
train on the FULL 80% holdout-train portion, and evaluate ONCE on the 20%
holdout-test that has never been touched by Optuna.

These are the unbiased numbers that go on the talk's headline slide.

For each model we save:
    results/holdout_<model>.json   metrics + bootstrapped 95% CIs
    results/holdout_<model>.npz    raw y_true + y_prob for plots
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loading import load_aligned_dataset, REPO_ROOT
from src.preprocessing import (
    log_transform, filter_low_expression,
    fit_zscore_params, apply_zscore,
)
from src.genome_ordering import order_genes_by_genome
from src.pathways import load_gmt, compute_pathway_scores
from src.gene_panels import build_panel_sized
from src.models.mlp import MLPClassifier
from src.models.cnn1d import CNN1D
from src.models.gene_transformer import GeneTransformer
from src.models.lstm_chrom import LSTMChrom
from src.training import TrainConfig, train_one_fold, best_device
from src.evaluation import compute_metrics, bootstrap_metric

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

RESULTS_DIR = REPO_ROOT / "results"
DEVICE = best_device()
SEED = 42


# -------- splits -----------------------------------------------------------

def load_holdout_split() -> tuple[list, list]:
    p = REPO_ROOT / "data" / "holdout_split.json"
    with p.open() as f:
        d = json.load(f)
    return d["train"], d["test"]


# -------- per-model preprocessing (mirrors what each model's tuning used) --

def preprocess_logreg_epic(epic_df, train_ids, test_ids):
    sc = StandardScaler()
    X_tr = sc.fit_transform(epic_df.loc[train_ids].to_numpy())
    X_te = sc.transform(epic_df.loc[test_ids].to_numpy())
    return X_tr, X_te


def preprocess_pathway_mlp(tpm, train_ids, test_ids):
    log_train = log_transform(tpm[train_ids])
    log_train = filter_low_expression(log_train)
    means, stds = fit_zscore_params(log_train)
    train_z = apply_zscore(log_train, means, stds)
    log_test = log_transform(tpm[test_ids]).loc[train_z.index]
    test_z = apply_zscore(log_test, means, stds)
    sets = load_gmt()
    train_p = compute_pathway_scores(train_z, sets)
    test_p = compute_pathway_scores(test_z, sets)
    common = [c for c in train_p.columns if c in test_p.columns]
    X_tr = train_p.loc[train_ids][common].to_numpy().astype(np.float32)
    X_te = test_p.loc[test_ids][common].to_numpy().astype(np.float32)
    return X_tr, X_te


def preprocess_cnn(tpm, train_ids, test_ids, n_top_variable):
    log_all = log_transform(tpm)
    log_train = log_all[train_ids]
    log_train = filter_low_expression(log_train)
    order = order_genes_by_genome(tpm.index.tolist())
    in_order = [g for g in order if g in log_train.index]
    log_train = log_train.loc[in_order]

    if n_top_variable is not None and n_top_variable < len(log_train):
        var = log_train.var(axis=1)
        top_set = set(var.sort_values(ascending=False).head(n_top_variable).index)
        keep = [g for g in in_order if g in top_set]
        log_train = log_train.loc[keep]
    else:
        keep = in_order

    means, stds = fit_zscore_params(log_train)
    train_z = apply_zscore(log_train, means, stds)
    test_z = apply_zscore(log_all.loc[keep, test_ids], means, stds)
    return train_z[train_ids].to_numpy().T.astype(np.float32), test_z[test_ids].to_numpy().T.astype(np.float32)


def preprocess_transformer(tpm, train_ids, test_ids, panel):
    log_panel = log_transform(tpm.loc[panel])
    log_train = log_panel[train_ids]
    means, stds = fit_zscore_params(log_train)
    train_z = apply_zscore(log_train, means, stds)
    test_z = apply_zscore(log_panel[test_ids], means, stds)
    return train_z[train_ids].to_numpy().T.astype(np.float32), test_z[test_ids].to_numpy().T.astype(np.float32)


# -------- single train + holdout-evaluate runner ---------------------------

def evaluate_on_holdout(
    model_name: str,
    y_test_true: np.ndarray,
    y_test_prob: np.ndarray,
    n_bootstrap: int = 2000,
) -> dict:
    """Compute AUC + F1 + confusion matrix + bootstrapped CIs, return dict."""
    # Pick threshold = 0.5 for F1/confusion
    auc_p, auc_lo, auc_hi = bootstrap_metric(
        y_test_true, y_test_prob, roc_auc_score,
        n_resamples=n_bootstrap, seed=SEED,
    )
    f1_fn = lambda yt, yp: f1_score(yt, (yp >= 0.5).astype(int), zero_division=0)
    f1_p, f1_lo, f1_hi = bootstrap_metric(
        y_test_true, y_test_prob, f1_fn,
        n_resamples=n_bootstrap, seed=SEED,
    )
    metrics = compute_metrics(y_test_true, y_test_prob, threshold=0.5).as_dict()
    cm = confusion_matrix(y_test_true, (y_test_prob >= 0.5).astype(int))

    return {
        "model_name": model_name,
        "n_test": int(len(y_test_true)),
        "n_pos_test": int(y_test_true.sum()),
        "auc":   {"point": auc_p, "low": auc_lo, "high": auc_hi},
        "f1":    {"point": f1_p,  "low": f1_lo,  "high": f1_hi},
        "metrics_at_0.5": metrics,
        "confusion_matrix": cm.tolist(),
    }


def save_predictions(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> None:
    np.savez(
        RESULTS_DIR / f"holdout_{name}_predictions.npz",
        y_true=y_true, y_prob=y_prob,
    )


def save_metrics(name: str, results: dict) -> None:
    with (RESULTS_DIR / f"holdout_{name}.json").open("w") as f:
        json.dump(results, f, indent=2)


# -------- per-model training functions -------------------------------------

def run_logreg(epic, labels, train_ids, test_ids):
    """Model 1: logistic regression on EPIC fractions. No tuning — fixed config."""
    X_tr, X_te = preprocess_logreg_epic(epic, train_ids, test_ids)
    y_tr = (labels.loc[train_ids] == "MSI-H").astype(int).to_numpy()
    y_te = (labels.loc[test_ids] == "MSI-H").astype(int).to_numpy()

    model = LogisticRegression(
        l1_ratio=0.0, C=1.0, class_weight="balanced",
        max_iter=2000, solver="lbfgs", random_state=SEED,
    )
    model.fit(X_tr, y_tr)
    y_prob = model.predict_proba(X_te)[:, 1]
    return y_te, y_prob


def run_mlp(tpm, labels, train_ids, test_ids, params):
    X_tr, X_te = preprocess_pathway_mlp(tpm, train_ids, test_ids)
    y_tr = (labels.loc[train_ids] == "MSI-H").astype(int).to_numpy()
    y_te = (labels.loc[test_ids] == "MSI-H").astype(int).to_numpy()

    hidden = tuple(h for h in [params.get("h1"), params.get("h2"), params.get("h3")] if h is not None)
    model = MLPClassifier(input_dim=X_tr.shape[1], hidden_dims=hidden, dropout=params["dropout"])
    config = TrainConfig(
        epochs=300, batch_size=params["batch_size"],
        learning_rate=params["learning_rate"], weight_decay=params["weight_decay"],
        patience=30, min_epochs=30,
        scheduler_type=params["scheduler_type"], optimizer=params["optimizer"],
    )
    y_prob, _ = train_one_fold(model, X_tr, y_tr, X_te, y_te, config=config, device=DEVICE)
    return y_te, y_prob


def run_cnn(tpm, labels, train_ids, test_ids, params):
    n_top = params["n_top_variable"]
    X_tr, X_te = preprocess_cnn(tpm, train_ids, test_ids, n_top)
    y_tr = (labels.loc[train_ids] == "MSI-H").astype(int).to_numpy()
    y_te = (labels.loc[test_ids] == "MSI-H").astype(int).to_numpy()

    model = CNN1D(
        n_genes=X_tr.shape[1],
        base_channels=params["base_channels"],
        n_conv_blocks=params["n_conv_blocks"],
        kernel_size=params["kernel_size"],
        pool_size=params["pool_size"],
        pool_type=params["pool_type"],
        dropout_conv=params["dropout_conv"],
        dropout_head=params["dropout_head"],
        dense_dim=params["head_hidden"],
    )
    config = TrainConfig(
        epochs=200, batch_size=params["batch_size"],
        learning_rate=params["learning_rate"], weight_decay=params["weight_decay"],
        patience=30, min_epochs=30,
        scheduler_type=params["scheduler_type"], optimizer=params["optimizer"],
    )
    y_prob, _ = train_one_fold(model, X_tr, y_tr, X_te, y_te, config=config, device=DEVICE)
    return y_te, y_prob


def run_transformer(tpm, labels, train_ids, test_ids, params):
    panel = build_panel_sized(set(tpm.index), size=params["panel"])
    X_tr, X_te = preprocess_transformer(tpm, train_ids, test_ids, panel)
    y_tr = (labels.loc[train_ids] == "MSI-H").astype(int).to_numpy()
    y_te = (labels.loc[test_ids] == "MSI-H").astype(int).to_numpy()

    model = GeneTransformer(
        n_genes=len(panel),
        d_model=params["d_model"],
        n_heads=params["n_heads"],
        n_layers=params["n_layers"],
        dim_feedforward=params["dim_feedforward"],
        dropout=params["dropout"],
    )
    config = TrainConfig(
        epochs=200, batch_size=params["batch_size"],
        learning_rate=params["learning_rate"], weight_decay=params["weight_decay"],
        patience=30, min_epochs=30,
        scheduler_type=params["scheduler_type"], optimizer=params["optimizer"],
    )
    y_prob, _ = train_one_fold(model, X_tr, y_tr, X_te, y_te, config=config, device=DEVICE)
    return y_te, y_prob


def run_lstm(tpm, labels, train_ids, test_ids, params):
    X_tr, X_te = preprocess_cnn(tpm, train_ids, test_ids, params["n_top_variable"])
    y_tr = (labels.loc[train_ids] == "MSI-H").astype(int).to_numpy()
    y_te = (labels.loc[test_ids] == "MSI-H").astype(int).to_numpy()

    model = LSTMChrom(
        n_genes=X_tr.shape[1],
        chunk_size=params["chunk_size"],
        chunk_pool=params["chunk_pool"],
        rnn_type=params["rnn_type"],
        hidden_size=params["hidden_size"],
        n_layers=params["rnn_n_layers"],
        bidirectional=params["bidirectional"],
        dropout_rnn=params["dropout_rnn"],
        dropout_head=params["dropout_head"],
        dense_dim=params["dense_dim"],
        sequence_pool=params["sequence_pool"],
    )
    config = TrainConfig(
        epochs=200, batch_size=params["batch_size"],
        learning_rate=params["learning_rate"], weight_decay=params["weight_decay"],
        patience=30, min_epochs=30,
        scheduler_type=params["scheduler_type"], optimizer=params["optimizer"],
    )
    y_prob, _ = train_one_fold(model, X_tr, y_tr, X_te, y_te, config=config, device=DEVICE)
    return y_te, y_prob


# -------- main -------------------------------------------------------------

def main() -> None:
    print(f"Device: {DEVICE}", flush=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("\nLoading data...", flush=True)
    tpm, epic, labels = load_aligned_dataset()
    train_ids, test_ids = load_holdout_split()
    print(f"  holdout-train: {len(train_ids)} samples ({(labels.loc[train_ids] == 'MSI-H').mean():.3f} MSI-H)")
    print(f"  holdout-test : {len(test_ids)} samples ({(labels.loc[test_ids] == 'MSI-H').mean():.3f} MSI-H)")

    # Map model name → (runner fn, optional best.json file)
    runs = [
        ("logreg_epic",       lambda: run_logreg(epic, labels, train_ids, test_ids), None),
        ("mlp_pathways",      lambda p: run_mlp(tpm, labels, train_ids, test_ids, p), "mlp_tpe_v2_best.json"),
        ("cnn_chromorder",    lambda p: run_cnn(tpm, labels, train_ids, test_ids, p), "cnn_tpe_v2_best.json"),
        ("transformer_panel", lambda p: run_transformer(tpm, labels, train_ids, test_ids, p), "transformer_tpe_v2_best.json"),
        ("lstm_chromorder",   lambda p: run_lstm(tpm, labels, train_ids, test_ids, p), "lstm_tpe_v2_best.json"),
    ]

    summary_rows = []
    for name, fn, best_file in runs:
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        if best_file is None:
            y_true, y_prob = fn()
        else:
            with (RESULTS_DIR / best_file).open() as f:
                best_params = json.load(f)["best_params"]
            print(f"  using params from {best_file}")
            y_true, y_prob = fn(best_params)
        elapsed = time.time() - t0

        results = evaluate_on_holdout(name, y_true, y_prob)
        save_predictions(name, y_true, y_prob)
        save_metrics(name, results)

        auc = results["auc"]
        f1  = results["f1"]
        cm = results["confusion_matrix"]
        print(f"  AUC  : {auc['point']:.3f}  [95% CI {auc['low']:.3f}-{auc['high']:.3f}]")
        print(f"  F1   : {f1['point']:.3f}  [95% CI {f1['low']:.3f}-{f1['high']:.3f}]")
        print(f"  conf : MSS [{cm[0][0]:>3} {cm[0][1]:>3}]  MSI-H [{cm[1][0]:>3} {cm[1][1]:>3}]  ({elapsed:.1f}s)")
        summary_rows.append((name, auc["point"], auc["low"], auc["high"], f1["point"]))

    # Overall summary
    print("\n" + "=" * 72)
    print("HOLDOUT SUMMARY (n_test = {})".format(len(test_ids)))
    print("=" * 72)
    print(f"  {'model':<22s} {'AUC':>6s}  {'95% CI':>16s}     {'F1':>6s}")
    for n, ap, alo, ahi, fp in summary_rows:
        print(f"  {n:<22s} {ap:>6.3f}  [{alo:.3f} - {ahi:.3f}]  {fp:>6.3f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
