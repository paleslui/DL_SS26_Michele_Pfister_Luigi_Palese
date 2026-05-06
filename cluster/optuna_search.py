"""Optuna hyperparameter search worker.

Usage (one worker, one study):
    python optuna_search.py --model cnn        --sampler tpe    --n-trials 200
    python optuna_search.py --model cnn        --sampler random --n-trials  50
    python optuna_search.py --model transformer --sampler tpe   --n-trials 200
    python optuna_search.py --model transformer --sampler random --n-trials 50

All workers writing to the same SQLite database (default
$REPO/results/optuna_studies.db) coordinate automatically — multiple workers can
run in parallel against the same study and trials are not duplicated.

Robustness:
  * Uses load_if_exists=True, so the same job can be killed and resubmitted.
    The job picks up where it left off, no data loss.
  * After every completed trial, prints a summary line and saves the current
    best hyperparameters to results/<study_name>_best.json.
  * stdout/stderr flushed after every print, so SLURM .out files show progress
    in real time (useful for `tail -f`).
  * Per-trial pruning at fold granularity (Median pruner).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import torch
from sklearn.metrics import roc_auc_score

# -- Make repo root importable, regardless of where this script is invoked from
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data_loading import load_aligned_dataset
from src.splits import load_splits
from src.preprocessing import (
    log_transform, filter_low_expression,
    fit_zscore_params, apply_zscore,
)
from src.genome_ordering import order_genes_by_genome
from src.gene_panels import build_panel_sized
from src.models.cnn1d import CNN1D
from src.models.gene_transformer import GeneTransformer
from src.training import TrainConfig, train_one_fold, best_device


# ---------- shared data cache (load once per worker) ------------------------

class _Cache:
    """Loads expensive artifacts once and reuses across trials."""

    def __init__(self) -> None:
        self.tpm = None
        self.labels = None
        self.folds = None
        self.genome_order = None
        self.log_tpm = None  # log-transformed once; per-fold z-score still happens

    def load(self) -> None:
        print("Loading data (one-time per worker)...", flush=True)
        t0 = time.time()
        self.tpm, _, self.labels = load_aligned_dataset()
        self.folds = load_splits()
        self.genome_order = order_genes_by_genome(self.tpm.index.tolist())
        self.log_tpm = log_transform(self.tpm)
        print(f"  {len(self.labels)} samples, {self.tpm.shape[0]} genes, "
              f"{len(self.genome_order)} positioned, {len(self.folds)} folds  "
              f"({time.time() - t0:.1f}s)", flush=True)


CACHE = _Cache()


# ---------- preprocessing helpers (per-fold, no leakage) --------------------

def prep_fold_cnn(
    train_ids: list, val_ids: list, n_top_variable: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """log → filter → genome-order → top-N variable on train → z-score.
    n_top_variable=None means keep all positioned + filtered genes."""
    log_train = CACHE.log_tpm[train_ids]
    log_train = filter_low_expression(log_train)
    in_order = [g for g in CACHE.genome_order if g in log_train.index]
    log_train = log_train.loc[in_order]

    if n_top_variable is not None and n_top_variable < len(log_train):
        variances = log_train.var(axis=1)
        top_set = set(variances.sort_values(ascending=False).head(n_top_variable).index)
        keep = [g for g in in_order if g in top_set]
        log_train = log_train.loc[keep]
    else:
        keep = in_order

    means, stds = fit_zscore_params(log_train)
    train_z = apply_zscore(log_train, means, stds)
    val_z = apply_zscore(CACHE.log_tpm.loc[keep, val_ids], means, stds)
    return train_z[train_ids].to_numpy().T, val_z[val_ids].to_numpy().T


def prep_fold_transformer(
    train_ids: list, val_ids: list, panel: list,
) -> tuple[np.ndarray, np.ndarray]:
    """log + train-only z-score, restricted to the panel genes."""
    log_panel = CACHE.log_tpm.loc[panel]
    log_train = log_panel[train_ids]
    means, stds = fit_zscore_params(log_train)
    train_z = apply_zscore(log_train, means, stds)
    val_z = apply_zscore(log_panel[val_ids], means, stds)
    return train_z[train_ids].to_numpy().T, val_z[val_ids].to_numpy().T


# ---------- model builders --------------------------------------------------

def build_cnn(trial: optuna.Trial, n_genes: int) -> CNN1D:
    kernel = trial.params["kernel_size"]
    base = trial.params["base_channels"]
    pool = trial.params["pool_size"]
    dropout_conv = trial.params["dropout_conv"]
    dropout_head = trial.params["dropout_head"]
    head = trial.params["head_hidden"]
    return CNN1D(
        n_genes=n_genes,
        channels=(base, base * 2, base * 4),
        kernel_size=kernel,
        pool_size=pool,
        dropout_conv=dropout_conv,
        dropout_head=dropout_head,
        dense_dim=head,
    )


def build_transformer(trial: optuna.Trial, n_genes: int) -> GeneTransformer:
    return GeneTransformer(
        n_genes=n_genes,
        d_model=trial.params["d_model"],
        n_heads=trial.params["n_heads"],
        n_layers=trial.params["n_layers"],
        dim_feedforward=trial.params["dim_feedforward"],
        dropout=trial.params["dropout"],
    )


# ---------- objectives ------------------------------------------------------

def cnn_objective(trial: optuna.Trial, device: torch.device) -> float:
    # Search space
    n_top = trial.suggest_categorical("n_top_variable", [None, 15000, 18000, 20000])
    trial.suggest_categorical("kernel_size", [7, 11, 15, 21, 31])
    trial.suggest_categorical("base_channels", [16, 32, 64])
    trial.suggest_categorical("pool_size", [2, 4])
    trial.suggest_float("dropout_conv", 0.0, 0.4)
    trial.suggest_float("dropout_head", 0.2, 0.6)
    trial.suggest_categorical("head_hidden", [64, 128, 256])
    lr = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    bs = trial.suggest_categorical("batch_size", [16, 32, 64])
    sched = trial.suggest_categorical("scheduler_type", ["none", "cosine", "plateau"])

    config = TrainConfig(
        epochs=120, batch_size=bs, learning_rate=lr, weight_decay=wd,
        patience=20, min_epochs=15, scheduler_type=sched,
    )

    fold_aucs = []
    for fold_idx, fold in enumerate(CACHE.folds):
        X_tr, X_va = prep_fold_cnn(fold["train"], fold["val"], n_top)
        y_tr = (CACHE.labels.loc[fold["train"]] == "MSI-H").astype(int).to_numpy()
        y_va = (CACHE.labels.loc[fold["val"]] == "MSI-H").astype(int).to_numpy()

        model = build_cnn(trial, n_genes=X_tr.shape[1])
        y_prob, _ = train_one_fold(model, X_tr, y_tr, X_va, y_va, config=config, device=device)
        fold_aucs.append(roc_auc_score(y_va, y_prob))

        # Report intermediate (mean so far) for fold-granularity pruning
        running_mean = float(np.mean(fold_aucs))
        trial.report(running_mean, step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_aucs))


def transformer_objective(trial: optuna.Trial, device: torch.device) -> float:
    panel_size = trial.suggest_categorical("panel", ["small", "medium", "large", "xlarge"])
    panel = build_panel_sized(set(CACHE.tpm.index), size=panel_size)

    d_model = trial.suggest_categorical("d_model", [32, 64, 128])
    # n_heads must divide d_model — use only valid combos
    valid_heads = [h for h in (2, 4, 8) if d_model % h == 0]
    n_heads = trial.suggest_categorical("n_heads", valid_heads)
    trial.suggest_int("n_layers", 1, 3)
    trial.suggest_categorical("dim_feedforward", [64, 128, 256, 512])
    trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    bs = trial.suggest_categorical("batch_size", [16, 32, 64])
    sched = trial.suggest_categorical("scheduler_type", ["none", "cosine", "plateau"])

    # Avoid the n_heads suggestion warning if n_heads/d_model combos differ across trials
    _ = n_heads  # used implicitly via trial.params["n_heads"]

    config = TrainConfig(
        epochs=100, batch_size=bs, learning_rate=lr, weight_decay=wd,
        patience=20, min_epochs=15, scheduler_type=sched,
    )

    fold_aucs = []
    for fold_idx, fold in enumerate(CACHE.folds):
        X_tr, X_va = prep_fold_transformer(fold["train"], fold["val"], panel)
        y_tr = (CACHE.labels.loc[fold["train"]] == "MSI-H").astype(int).to_numpy()
        y_va = (CACHE.labels.loc[fold["val"]] == "MSI-H").astype(int).to_numpy()

        model = build_transformer(trial, n_genes=len(panel))
        y_prob, _ = train_one_fold(model, X_tr, y_tr, X_va, y_va, config=config, device=device)
        fold_aucs.append(roc_auc_score(y_va, y_prob))

        running_mean = float(np.mean(fold_aucs))
        trial.report(running_mean, step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_aucs))


# ---------- progress callback ----------------------------------------------

def make_progress_callback(study_name: str, results_dir: Path):
    """Build a callback that prints + persists best params after every trial."""
    state = {"start_time": time.time()}
    best_path = results_dir / f"{study_name}_best.json"

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        elapsed_s = time.time() - state["start_time"]
        elapsed = f"{int(elapsed_s // 60)}m{int(elapsed_s % 60):02d}s"

        # Trial outcome
        if trial.state == optuna.trial.TrialState.COMPLETE:
            outcome = f"AUC={trial.value:.4f}"
        elif trial.state == optuna.trial.TrialState.PRUNED:
            outcome = "PRUNED"
        elif trial.state == optuna.trial.TrialState.FAIL:
            outcome = "FAILED"
        else:
            outcome = str(trial.state)

        try:
            best = study.best_trial
            best_str = f"best={best.value:.4f} (trial {best.number})"
        except ValueError:
            best_str = "best=- (no completed trial yet)"

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
              f"Trial {trial.number:>4} | {outcome:>16s} | {best_str} | elapsed={elapsed}",
              flush=True)
        # Compact param printout
        if trial.params:
            params_brief = "  ".join(f"{k}={v}" for k, v in trial.params.items())
            print(f"    params: {params_brief}", flush=True)

        # Save best params after every trial so a kill mid-run doesn't lose progress
        try:
            best = study.best_trial
            payload = {
                "study_name": study_name,
                "best_trial_number": best.number,
                "best_value_auc": best.value,
                "best_params": best.params,
                "n_completed": sum(
                    1 for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE
                ),
                "n_pruned": sum(
                    1 for t in study.trials
                    if t.state == optuna.trial.TrialState.PRUNED
                ),
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            results_dir.mkdir(parents=True, exist_ok=True)
            with best_path.open("w") as f:
                json.dump(payload, f, indent=2)
        except ValueError:
            pass  # No completed trial yet

    return callback


# ---------- main ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "transformer"], required=True)
    parser.add_argument("--sampler", choices=["tpe", "random"], required=True)
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--storage", default=None,
                        help="Optuna storage URL (default: sqlite:///<repo>/results/optuna_studies.db)")
    parser.add_argument("--study-name", default=None,
                        help="Default: <model>_<sampler>")
    parser.add_argument("--results-dir", default=None,
                        help="Default: <repo>/results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    storage = args.storage or f"sqlite:///{results_dir / 'optuna_studies.db'}"
    study_name = args.study_name or f"{args.model}_{args.sampler}"

    print("=" * 70, flush=True)
    print(f"Optuna search: {study_name}", flush=True)
    print(f"  storage   : {storage}", flush=True)
    print(f"  n_trials  : {args.n_trials} (this worker)", flush=True)
    print(f"  results   : {results_dir}", flush=True)
    print("=" * 70, flush=True)

    # Two-step seeding so resumed workers don't re-explore the same params:
    #   1. base seed varies by worker_id (distinguishes parallel workers)
    #   2. additional offset = number of trials already in the study
    #      (distinguishes a fresh start from a resumed run)
    worker_id = int(os.environ.get("SLURM_PROCID", os.environ.get("WORKER_ID", "0")))

    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2)
    # n_warmup_steps=2 because we report after each fold (5 total); pruner can act after fold 2.

    # First create/load the study with a placeholder sampler, just to count existing trials
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=optuna.samplers.RandomSampler(seed=0),
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )
    n_existing_trials = len(study.trials)
    worker_seed = args.seed + worker_id * 1000 + n_existing_trials
    print(f"Worker seed: {worker_seed}  (worker_id={worker_id}, existing_trials={n_existing_trials})", flush=True)

    # Now install the real sampler
    if args.sampler == "tpe":
        study.sampler = optuna.samplers.TPESampler(seed=worker_seed, n_startup_trials=15)
    elif args.sampler == "random":
        study.sampler = optuna.samplers.RandomSampler(seed=worker_seed)
    else:
        raise ValueError(args.sampler)

    # Load data once
    CACHE.load()

    # Objective
    device = best_device()
    print(f"Device: {device}", flush=True)
    if args.model == "cnn":
        objective = lambda t: cnn_objective(t, device)
    else:
        objective = lambda t: transformer_objective(t, device)

    callback = make_progress_callback(study_name, results_dir)

    study.optimize(
        objective,
        n_trials=args.n_trials,
        callbacks=[callback],
        gc_after_trial=True,
        show_progress_bar=False,
    )

    # Final summary
    print("\n" + "=" * 70, flush=True)
    print("Search complete.", flush=True)
    try:
        best = study.best_trial
        print(f"Best AUC : {best.value:.4f}  (trial {best.number})", flush=True)
        print(f"Best params:", flush=True)
        for k, v in best.params.items():
            print(f"  {k}: {v}", flush=True)
    except ValueError:
        print("No completed trials.", flush=True)


if __name__ == "__main__":
    main()
