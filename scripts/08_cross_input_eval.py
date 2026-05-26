"""Cross-input evaluation: 3 architectures × 4 inputs grid.

Holds each architecture's tuned hyperparameters fixed (option α) and only
swaps the input format. Tests "input vs architecture" directly.

Grid:
    arch \\ input | EPIC(8) | Pathways(50) | Panel(38) | ChrOrdered(12k)
    -----------------------------------------------------------------
    Logreg       |  ✅    |     new      |   new     |    new
    MLP          |  new   |    ✅       |   new     |    new
    Transformer  |  new   |     new      |    ✅    | skipped (memory)

Cells marked ✅ are already in results/holdout_*.json from script 07 and
are NOT re-run here - we'll merge them into the final table.

Outputs:
    results/holdout_<arch>_on_<input>.json
    results/cross_input_summary.json   (combined grid + reused existing cells)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

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
from src.models.gene_transformer import GeneTransformer
from src.training import TrainConfig, train_one_fold, best_device
from src.evaluation import bootstrap_metric, compute_metrics

RESULTS_DIR = REPO_ROOT / "results"
DEVICE = best_device()
SEED = 42

# ---------- splits ---------------------------------------------------------

def load_holdout():
    p = REPO_ROOT / "data" / "holdout_split.json"
    with p.open() as f:
        d = json.load(f)
    return d["train"], d["test"]


# ---------- per-input preprocessing ----------------------------------------

def build_input_epic(epic, train_ids, test_ids):
    sc = StandardScaler()
    X_tr = sc.fit_transform(epic.loc[train_ids].to_numpy()).astype(np.float32)
    X_te = sc.transform(epic.loc[test_ids].to_numpy()).astype(np.float32)
    return X_tr, X_te


def build_input_pathways(tpm, train_ids, test_ids):
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
    return (train_p.loc[train_ids][common].to_numpy().astype(np.float32),
            test_p.loc[test_ids][common].to_numpy().astype(np.float32))


def build_input_panel(tpm, train_ids, test_ids, panel_size="small"):
    panel = build_panel_sized(set(tpm.index), size=panel_size)
    log_panel = log_transform(tpm.loc[panel])
    log_train = log_panel[train_ids]
    means, stds = fit_zscore_params(log_train)
    train_z = apply_zscore(log_train, means, stds)
    test_z = apply_zscore(log_panel[test_ids], means, stds)
    return (train_z[train_ids].to_numpy().T.astype(np.float32),
            test_z[test_ids].to_numpy().T.astype(np.float32),
            panel)


def build_input_chrordered(tpm, train_ids, test_ids, n_top=12000):
    log_all = log_transform(tpm)
    log_train = log_all[train_ids]
    log_train = filter_low_expression(log_train)
    order = order_genes_by_genome(tpm.index.tolist())
    in_order = [g for g in order if g in log_train.index]
    log_train = log_train.loc[in_order]
    var = log_train.var(axis=1)
    top_set = set(var.sort_values(ascending=False).head(n_top).index)
    keep = [g for g in in_order if g in top_set]
    log_train = log_train.loc[keep]
    means, stds = fit_zscore_params(log_train)
    train_z = apply_zscore(log_train, means, stds)
    test_z = apply_zscore(log_all.loc[keep, test_ids], means, stds)
    return (train_z[train_ids].to_numpy().T.astype(np.float32),
            test_z[test_ids].to_numpy().T.astype(np.float32))


# ---------- evaluation ----------------------------------------------------

def eval_holdout(y_true, y_prob, n_bootstrap=2000):
    auc_p, auc_lo, auc_hi = bootstrap_metric(y_true, y_prob, roc_auc_score,
                                              n_resamples=n_bootstrap, seed=SEED)
    f1_fn = lambda yt, yp: f1_score(yt, (yp >= 0.5).astype(int), zero_division=0)
    f1_p, f1_lo, f1_hi = bootstrap_metric(y_true, y_prob, f1_fn,
                                           n_resamples=n_bootstrap, seed=SEED)
    cm = confusion_matrix(y_true, (y_prob >= 0.5).astype(int))
    return {
        "auc": {"point": auc_p, "low": auc_lo, "high": auc_hi},
        "f1":  {"point": f1_p,  "low": f1_lo,  "high": f1_hi},
        "confusion_matrix": cm.tolist(),
        "metrics_at_0.5": compute_metrics(y_true, y_prob, 0.5).as_dict(),
    }


# ---------- architecture runners -------------------------------------------

def run_logreg(X_tr, y_tr, X_te):
    """Logreg has no input-specific hyperparameters; same config everywhere."""
    model = LogisticRegression(
        l1_ratio=0.0, C=1.0, class_weight="balanced",
        max_iter=2000, solver="lbfgs", random_state=SEED,
    )
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_te)[:, 1]


def run_mlp(X_tr, y_tr, X_te, y_te):
    """MLP with the v2-tuned hyperparameters."""
    with (RESULTS_DIR / "mlp_tpe_v2_best.json").open() as f:
        p = json.load(f)["best_params"]
    hidden = tuple(h for h in [p.get("h1"), p.get("h2"), p.get("h3")] if h is not None)
    model = MLPClassifier(input_dim=X_tr.shape[1], hidden_dims=hidden, dropout=p["dropout"])
    cfg = TrainConfig(
        epochs=300, batch_size=p["batch_size"],
        learning_rate=p["learning_rate"], weight_decay=p["weight_decay"],
        patience=30, min_epochs=30,
        scheduler_type=p["scheduler_type"], optimizer=p["optimizer"],
    )
    y_prob, _ = train_one_fold(model, X_tr, y_tr, X_te, y_te, config=cfg, device=DEVICE)
    return y_prob


def run_transformer(X_tr, y_tr, X_te, y_te):
    """Transformer with the v2-tuned hyperparameters. n_genes inferred from input."""
    with (RESULTS_DIR / "transformer_tpe_v2_best.json").open() as f:
        p = json.load(f)["best_params"]
    model = GeneTransformer(
        n_genes=X_tr.shape[1],
        d_model=p["d_model"], n_heads=p["n_heads"], n_layers=p["n_layers"],
        dim_feedforward=p["dim_feedforward"], dropout=p["dropout"],
    )
    cfg = TrainConfig(
        epochs=200, batch_size=p["batch_size"],
        learning_rate=p["learning_rate"], weight_decay=p["weight_decay"],
        patience=30, min_epochs=30,
        scheduler_type=p["scheduler_type"], optimizer=p["optimizer"],
    )
    y_prob, _ = train_one_fold(model, X_tr, y_tr, X_te, y_te, config=cfg, device=DEVICE)
    return y_prob


# ---------- main ----------------------------------------------------------

def main():
    print(f"Device: {DEVICE}", flush=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("\nLoading data...", flush=True)
    tpm, epic, labels = load_aligned_dataset()
    train_ids, test_ids = load_holdout()
    y_tr = (labels.loc[train_ids] == "MSI-H").astype(int).to_numpy()
    y_te = (labels.loc[test_ids] == "MSI-H").astype(int).to_numpy()
    print(f"  train: {len(train_ids)}, test: {len(test_ids)}")

    # Build all 4 input formats once
    print("\nBuilding inputs (once for all architectures)...", flush=True)
    t0 = time.time()
    XA_tr, XA_te = build_input_epic(epic, train_ids, test_ids)
    print(f"  EPIC:        train {XA_tr.shape}, test {XA_te.shape}  ({time.time()-t0:.1f}s)")
    t0 = time.time()
    XB_tr, XB_te = build_input_pathways(tpm, train_ids, test_ids)
    print(f"  Pathways:    train {XB_tr.shape}, test {XB_te.shape}  ({time.time()-t0:.1f}s)")
    t0 = time.time()
    XC_tr, XC_te, panel = build_input_panel(tpm, train_ids, test_ids)
    print(f"  Panel(38):   train {XC_tr.shape}, test {XC_te.shape}  ({time.time()-t0:.1f}s)")
    t0 = time.time()
    XD_tr, XD_te = build_input_chrordered(tpm, train_ids, test_ids, n_top=12000)
    print(f"  ChrOrdered:  train {XD_tr.shape}, test {XD_te.shape}  ({time.time()-t0:.1f}s)")

    inputs = {
        "epic":         (XA_tr, XA_te),
        "pathways":     (XB_tr, XB_te),
        "panel38":      (XC_tr, XC_te),
        "chrord12k":    (XD_tr, XD_te),
    }

    # ---- 8 new cells ----
    new_cells = [
        ("logreg",      "pathways"),
        ("logreg",      "panel38"),
        ("logreg",      "chrord12k"),
        ("mlp",         "epic"),
        ("mlp",         "panel38"),
        ("mlp",         "chrord12k"),
        ("transformer", "epic"),
        ("transformer", "pathways"),
    ]

    results = {}
    for arch, inp in new_cells:
        cell_name = f"{arch}_on_{inp}"
        print(f"\n=== {cell_name} ===", flush=True)
        X_tr, X_te = inputs[inp]
        t0 = time.time()
        if arch == "logreg":
            y_prob = run_logreg(X_tr, y_tr, X_te)
        elif arch == "mlp":
            y_prob = run_mlp(X_tr, y_tr, X_te, y_te)
        elif arch == "transformer":
            y_prob = run_transformer(X_tr, y_tr, X_te, y_te)
        else:
            raise ValueError(arch)
        ev = eval_holdout(y_te, y_prob)
        elapsed = time.time() - t0

        ev["cell"] = cell_name
        ev["arch"] = arch
        ev["input"] = inp
        ev["elapsed_seconds"] = elapsed
        with (RESULTS_DIR / f"holdout_{cell_name}.json").open("w") as f:
            json.dump(ev, f, indent=2)
        np.savez(RESULTS_DIR / f"holdout_{cell_name}_predictions.npz",
                 y_true=y_te, y_prob=y_prob)
        print(f"  AUC: {ev['auc']['point']:.3f}  [{ev['auc']['low']:.3f}-{ev['auc']['high']:.3f}]   "
              f"F1: {ev['f1']['point']:.3f}   ({elapsed:.1f}s)")
        results[cell_name] = ev

    # ---- merge with existing already-run cells from script 07 ----
    existing = [
        ("logreg",      "epic",     "holdout_logreg_epic.json"),
        ("mlp",         "pathways", "holdout_mlp_pathways.json"),
        ("transformer", "panel38",  "holdout_transformer_panel.json"),
    ]
    for arch, inp, fname in existing:
        cell_name = f"{arch}_on_{inp}"
        path = RESULTS_DIR / fname
        if not path.exists():
            print(f"warn: existing {fname} not found, skipping")
            continue
        with path.open() as f:
            r = json.load(f)
        # Standardise to our format
        results[cell_name] = {
            "cell": cell_name,
            "arch": arch,
            "input": inp,
            "auc": r["auc"],
            "f1":  r["f1"],
            "confusion_matrix": r["confusion_matrix"],
            "metrics_at_0.5": r.get("metrics_at_0.5"),
            "elapsed_seconds": None,
            "reused_from": fname,
        }

    # ---- summary table ----
    print("\n" + "=" * 88)
    print(f"{'CROSS-INPUT GRID - holdout AUC (n_test=' + str(len(test_ids)) + ')':^88s}")
    print("=" * 88)
    print(f"  {'Architecture':<13s} | {'EPIC(8)':>17s} | {'Pathways(50)':>17s} | "
          f"{'Panel38':>17s} | {'ChrOrd12k':>17s}")
    print("  " + "-" * 86)
    for arch in ["logreg", "mlp", "transformer"]:
        row = f"  {arch:<13s} |"
        for inp in ["epic", "pathways", "panel38", "chrord12k"]:
            cell = f"{arch}_on_{inp}"
            if cell in results:
                a = results[cell]["auc"]
                row += f" {a['point']:>4.3f} [{a['low']:.2f}-{a['high']:.2f}] |"
            elif arch == "transformer" and inp == "chrord12k":
                row += f" {'(skipped: OOM)':>17s} |"
            else:
                row += f" {'-':>17s} |"
        print(row)
    print("=" * 88)

    with (RESULTS_DIR / "cross_input_summary.json").open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to results/cross_input_summary.json")


if __name__ == "__main__":
    main()
