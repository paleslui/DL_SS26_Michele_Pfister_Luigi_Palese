"""Hyperparameter tuning for the pathway-MLP model (Model 2).

Standalone: runs locally (CPU or MPS), small search space, ~75 trials total
(50 TPE + 25 random for the methodology comparison). Writes best params to
results/mlp_tpe_best.json and results/mlp_random_best.json.

Why a separate script?
  - The pathway MLP is small enough that 75 trials run in <30 min on a laptop
  - The cluster is busy with CNN/transformer/LSTM searches
  - This makes the architecture comparison fair: every model now tuned

Run from repo root:
    python scripts/06_tune_pathway_mlp.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import torch

# Path setup
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loading import load_aligned_dataset, REPO_ROOT
from src.splits import load_splits
from src.preprocessing import preprocess_for_fold
from src.pathways import load_gmt, compute_pathway_scores
from src.models.mlp import MLPClassifier
from src.training import TrainConfig, train_one_fold, best_device
from sklearn.metrics import roc_auc_score

RESULTS_DIR = REPO_ROOT / "results"
DB_PATH = RESULTS_DIR / "optuna_studies.db"
DEVICE = best_device()


# ---------- shared cache (load once) ----------------------------------------

class _Cache:
    def __init__(self) -> None:
        self.tpm = None
        self.labels = None
        self.folds = None
        self.gene_sets = None
        # Pre-compute per-fold pathway scores so trials don't redo it
        self.fold_pathways: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    def load(self) -> None:
        print("Loading data + computing pathway scores per fold (one-time)...", flush=True)
        t0 = time.time()
        self.tpm, _, self.labels = load_aligned_dataset()
        self.folds = load_splits()
        self.gene_sets = load_gmt()

        for fold in self.folds:
            train_ids = fold["train"]
            val_ids = fold["val"]
            train_z, val_z = preprocess_for_fold(self.tpm, train_ids, val_ids)
            train_p = compute_pathway_scores(train_z, self.gene_sets)
            val_p = compute_pathway_scores(val_z, self.gene_sets)
            common = [p for p in train_p.columns if p in val_p.columns]
            X_tr = train_p.loc[train_ids][common].to_numpy().astype(np.float32)
            X_va = val_p.loc[val_ids][common].to_numpy().astype(np.float32)
            y_tr = (self.labels.loc[train_ids] == "MSI-H").astype(int).to_numpy()
            y_va = (self.labels.loc[val_ids] == "MSI-H").astype(int).to_numpy()
            self.fold_pathways.append((X_tr, y_tr, X_va, y_va))

        print(f"  {len(self.labels)} samples, {self.fold_pathways[0][0].shape[1]} pathways, "
              f"{len(self.folds)} folds  ({time.time() - t0:.1f}s)", flush=True)


CACHE = _Cache()


# ---------- objective -------------------------------------------------------

def objective(trial: optuna.Trial) -> float:
    # Architecture
    n_hidden_layers = trial.suggest_int("n_hidden_layers", 1, 3)
    h1 = trial.suggest_categorical("h1", [16, 32, 64, 128])
    h2 = trial.suggest_categorical("h2", [16, 32, 64]) if n_hidden_layers >= 2 else None
    h3 = trial.suggest_categorical("h3", [16, 32]) if n_hidden_layers >= 3 else None

    hidden_dims = tuple(h for h in (h1, h2, h3) if h is not None)
    dropout = trial.suggest_float("dropout", 0.1, 0.7)

    # Optimization
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    bs = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    sched = trial.suggest_categorical("scheduler_type", ["none", "cosine", "plateau"])
    opt = trial.suggest_categorical("optimizer", ["adam", "adamw"])

    config = TrainConfig(
        epochs=200, batch_size=bs, learning_rate=lr, weight_decay=wd,
        patience=25, min_epochs=20,
        scheduler_type=sched, optimizer=opt,
    )

    fold_aucs = []
    for fold_idx, (X_tr, y_tr, X_va, y_va) in enumerate(CACHE.fold_pathways):
        model = MLPClassifier(
            input_dim=X_tr.shape[1],
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        y_prob, _ = train_one_fold(
            model, X_tr, y_tr, X_va, y_va,
            config=config, device=DEVICE,
        )
        fold_aucs.append(roc_auc_score(y_va, y_prob))
        running_mean = float(np.mean(fold_aucs))
        trial.report(running_mean, step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_aucs))


# ---------- callback --------------------------------------------------------

def make_callback(study_name: str):
    state = {"start": time.time()}
    best_path = RESULTS_DIR / f"{study_name}_best.json"

    def cb(study, trial):
        elapsed = int(time.time() - state["start"])
        if trial.state == optuna.trial.TrialState.COMPLETE:
            outcome = f"AUC={trial.value:.4f}"
        elif trial.state == optuna.trial.TrialState.PRUNED:
            outcome = "PRUNED"
        else:
            outcome = str(trial.state)
        try:
            best = study.best_trial
            best_str = f"best={best.value:.4f} (#{best.number})"
        except ValueError:
            best_str = "best=-"
        print(f"  Trial {trial.number:>3} | {outcome:>13s} | {best_str} | {elapsed//60}m{elapsed%60:02d}s",
              flush=True)
        try:
            best = study.best_trial
            payload = {
                "study_name": study_name,
                "best_trial_number": best.number,
                "best_value_auc": best.value,
                "best_params": best.params,
                "n_completed": sum(1 for t in study.trials
                                    if t.state == optuna.trial.TrialState.COMPLETE),
                "n_pruned": sum(1 for t in study.trials
                                 if t.state == optuna.trial.TrialState.PRUNED),
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            with best_path.open("w") as f:
                json.dump(payload, f, indent=2)
        except ValueError:
            pass
    return cb


# ---------- main ------------------------------------------------------------

def run_study(study_name: str, sampler, n_trials: int) -> None:
    storage = f"sqlite:///{DB_PATH}"
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2)

    # Bootstrap study (count existing trials so resumed runs use a different seed)
    study = optuna.create_study(
        study_name=study_name, storage=storage,
        sampler=optuna.samplers.RandomSampler(seed=0),
        pruner=pruner, direction="maximize", load_if_exists=True,
    )
    n_existing = len(study.trials)
    if n_existing >= n_trials:
        print(f"\n[{study_name}] already has {n_existing} trials, skipping.")
        return

    study.sampler = sampler
    print(f"\n=== Running {study_name} (target: {n_trials} trials, existing: {n_existing}) ===", flush=True)
    study.optimize(
        objective,
        n_trials=n_trials - n_existing,
        callbacks=[make_callback(study_name)],
        gc_after_trial=True,
    )
    try:
        b = study.best_trial
        print(f"\n[{study_name}] DONE  best AUC={b.value:.4f}  (trial #{b.number})", flush=True)
    except ValueError:
        print(f"[{study_name}] no completed trials")


def main() -> None:
    print(f"Device: {DEVICE}", flush=True)
    CACHE.load()

    # 1) TPE — 50 trials
    tpe = optuna.samplers.TPESampler(
        seed=42, n_startup_trials=15, multivariate=True,
        n_ei_candidates=24, consider_endpoints=True,
    )
    run_study("mlp_tpe_v2", tpe, n_trials=50)

    # 2) Random — 25 trials baseline
    rnd = optuna.samplers.RandomSampler(seed=42)
    run_study("mlp_random_v2", rnd, n_trials=25)

    print("\nAll done.")


if __name__ == "__main__":
    main()
