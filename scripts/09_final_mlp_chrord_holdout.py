"""Final clean evaluation on the 20% holdout for the tuned MLP-on-chrord.

Uses hyperparameters from results/mlp_chrord_tpe_v2_best.json. Trains on the
full 80% holdout-train portion, evaluates ONCE on the 100-sample holdout-test.
Saves predictions + bootstrapped 95% CIs to results/holdout_mlp_chrord.json.

This is the missing companion to scripts/07_final_holdout_evaluation.py for
the newly-tuned MLP-on-chr-ordered configuration discovered after that script
was first run.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loading import load_aligned_dataset, REPO_ROOT
from src.preprocessing import (
    log_transform, filter_low_expression,
    fit_zscore_params, apply_zscore,
)
from src.genome_ordering import order_genes_by_genome
from src.models.mlp import MLPClassifier
from src.training import TrainConfig, train_one_fold, best_device
from src.evaluation import bootstrap_metric, compute_metrics

RESULTS_DIR = REPO_ROOT / 'results'
DEVICE = best_device()
SEED = 42


def load_holdout_split() -> tuple[list, list]:
    with (REPO_ROOT / 'data' / 'holdout_split.json').open() as f:
        d = json.load(f)
    return d['train'], d['test']


def preprocess_chrord(tpm, train_ids, test_ids, n_top_variable):
    """Same pipeline as Optuna search: log -> filter -> chromosome-order -> top-N variance -> z-score."""
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
    return (train_z[train_ids].to_numpy().T.astype(np.float32),
            test_z[test_ids].to_numpy().T.astype(np.float32))


def main():
    print(f'Device: {DEVICE}', flush=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load best params
    best_path = RESULTS_DIR / 'mlp_chrord_tpe_v2_best.json'
    with best_path.open() as f:
        best = json.load(f)
    p = best['best_params']
    print(f'\nLoaded best params from {best_path.name}:')
    print(f'  CV-AUC at search time: {best["best_value_auc"]:.4f}  (trial #{best["best_trial_number"]})')
    for k, v in p.items():
        print(f'  {k}: {v}')

    print('\nLoading data...', flush=True)
    tpm, _, labels = load_aligned_dataset()
    train_ids, test_ids = load_holdout_split()

    print(f'  holdout-train: {len(train_ids)}, holdout-test: {len(test_ids)}')

    t0 = time.time()
    X_tr, X_te = preprocess_chrord(tpm, train_ids, test_ids, p['n_top_variable'])
    print(f'  preprocessed: train {X_tr.shape}, test {X_te.shape}  ({time.time()-t0:.1f}s)')

    y_tr = (labels.loc[train_ids] == 'MSI-H').astype(int).to_numpy()
    y_te = (labels.loc[test_ids]  == 'MSI-H').astype(int).to_numpy()

    # Build the tuned model
    hidden = tuple(h for h in [p.get('h1'), p.get('h2'), p.get('h3')] if h is not None)
    model = MLPClassifier(
        input_dim=X_tr.shape[1],
        hidden_dims=hidden,
        dropout=p['dropout'],
    )
    cfg = TrainConfig(
        epochs=300, batch_size=p['batch_size'],
        learning_rate=p['learning_rate'], weight_decay=p['weight_decay'],
        patience=30, min_epochs=30,
        scheduler_type=p['scheduler_type'], optimizer=p['optimizer'],
    )

    print(f'\nTraining MLP({X_tr.shape[1]} -> {" -> ".join(map(str, hidden))} -> 1)')
    print(f'  params: {sum(pp.numel() for pp in model.parameters()):,}')
    t0 = time.time()
    y_prob, history = train_one_fold(model, X_tr, y_tr, X_te, y_te, config=cfg, device=DEVICE)
    elapsed = time.time() - t0
    print(f'  trained in {elapsed:.1f}s  (best epoch {history["best_epoch"]}, best val-AUC during training {history["best_val_auc"]:.4f})')

    # Evaluate
    auc_p, auc_lo, auc_hi = bootstrap_metric(
        y_te, y_prob, roc_auc_score, n_resamples=2000, seed=SEED,
    )
    f1_fn = lambda yt, yp: f1_score(yt, (yp >= 0.5).astype(int), zero_division=0)
    f1_p, f1_lo, f1_hi = bootstrap_metric(
        y_te, y_prob, f1_fn, n_resamples=2000, seed=SEED,
    )
    cm = confusion_matrix(y_te, (y_prob >= 0.5).astype(int))

    print(f'\n=== Holdout test results (n={len(y_te)}) ===')
    print(f'  AUC : {auc_p:.3f}  [95% CI {auc_lo:.3f} - {auc_hi:.3f}]')
    print(f'  F1  : {f1_p:.3f}  [95% CI {f1_lo:.3f} - {f1_hi:.3f}]')
    print(f'  Confusion: MSS [{cm[0,0]} {cm[0,1]}]  MSI-H [{cm[1,0]} {cm[1,1]}]')

    results = {
        'model_name': 'mlp_chrord',
        'best_params_file': str(best_path.name),
        'best_params': p,
        'cv_auc_at_search': best['best_value_auc'],
        'n_test': int(len(y_te)),
        'n_pos_test': int(y_te.sum()),
        'auc': {'point': auc_p, 'low': auc_lo, 'high': auc_hi},
        'f1':  {'point': f1_p,  'low': f1_lo,  'high': f1_hi},
        'metrics_at_0.5': compute_metrics(y_te, y_prob, 0.5).as_dict(),
        'confusion_matrix': cm.tolist(),
        'elapsed_seconds': elapsed,
        'best_train_epoch': history['best_epoch'],
    }
    with (RESULTS_DIR / 'holdout_mlp_chrord.json').open('w') as f:
        json.dump(results, f, indent=2)
    np.savez(RESULTS_DIR / 'holdout_mlp_chrord_predictions.npz',
             y_true=y_te, y_prob=y_prob)
    print(f'\nSaved to results/holdout_mlp_chrord.json')


if __name__ == '__main__':
    main()
