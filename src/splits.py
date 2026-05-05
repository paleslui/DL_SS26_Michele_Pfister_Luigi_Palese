"""Train/test split + stratified k-fold CV on the train portion.

Strategy
--------
We do an honest 80/20 holdout split FIRST, then perform stratified k-fold CV on
the 80% training portion only. The 20% holdout is never used during model
development or tuning — only for the final, unbiased report at the end.

This is the standard "outer holdout, inner CV" pattern. It prevents the very
common mistake of tuning hyperparameters against the same fold splits used to
report headline metrics.

File outputs
------------
data/holdout_split.json : {"train": [...ids...], "test": [...ids...]}
data/cv_splits.json     : list of {"fold": k, "train": [...], "val": [...]}
                          where train+val ⊆ holdout_split["train"]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .data_loading import REPO_ROOT


HOLDOUT_PATH = REPO_ROOT / "data" / "holdout_split.json"
SPLITS_PATH = REPO_ROOT / "data" / "cv_splits.json"
RANDOM_SEED = 42
N_FOLDS = 5
HOLDOUT_FRACTION = 0.20


# -------- 80/20 holdout ------------------------------------------------------

def make_holdout(
    labels: pd.Series,
    test_size: float = HOLDOUT_FRACTION,
    seed: int = RANDOM_SEED,
) -> dict:
    """Stratified train/test split. Returns {"train": [...], "test": [...]}."""
    sample_ids = labels.index.to_numpy()
    y = (labels == "MSI-H").astype(int).to_numpy()
    train_ids, test_ids = train_test_split(
        sample_ids, test_size=test_size, stratify=y, random_state=seed,
    )
    return {"train": train_ids.tolist(), "test": test_ids.tolist()}


def save_holdout(holdout: dict, path: Path = HOLDOUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(holdout, f, indent=2)
    print(f"Saved holdout split to {path}")


def load_holdout(path: Path = HOLDOUT_PATH) -> dict:
    with path.open() as f:
        return json.load(f)


# -------- CV folds (on the train portion only) ------------------------------

def make_cv_splits(
    labels: pd.Series,
    train_ids: list[str],
    n_folds: int = N_FOLDS,
    seed: int = RANDOM_SEED,
) -> list[dict]:
    """Stratified k-fold CV restricted to ``train_ids``."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    train_arr = np.array(train_ids)
    y = (labels.loc[train_ids] == "MSI-H").astype(int).to_numpy()

    folds = []
    for k, (tr_idx, val_idx) in enumerate(skf.split(train_arr, y)):
        folds.append({
            "fold": k,
            "train": train_arr[tr_idx].tolist(),
            "val":   train_arr[val_idx].tolist(),
        })
    return folds


def save_splits(folds: list[dict], path: Path = SPLITS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(folds, f, indent=2)
    print(f"Saved {len(folds)} CV folds to {path}")


def load_splits(path: Path = SPLITS_PATH) -> list[dict]:
    with path.open() as f:
        return json.load(f)


# -------- diagnostics --------------------------------------------------------

def summarize_holdout(holdout: dict, labels: pd.Series) -> None:
    train_y = labels.loc[holdout["train"]]
    test_y = labels.loc[holdout["test"]]
    print(f"  train portion: n={len(train_y):>3}  MSI-H frac={(train_y == 'MSI-H').mean():.3f}")
    print(f"  test  portion: n={len(test_y):>3}  MSI-H frac={(test_y == 'MSI-H').mean():.3f}")


def summarize_splits(folds: list[dict], labels: pd.Series) -> None:
    print(f"  {'Fold':>4}  {'Train n':>8}  {'Val n':>6}  {'Train MSI-H':>12}  {'Val MSI-H':>10}")
    print("  " + "-" * 50)
    for fold in folds:
        train = labels.loc[fold["train"]]
        val = labels.loc[fold["val"]]
        print(
            f"  {fold['fold']:>4}  "
            f"{len(train):>8}  {len(val):>6}  "
            f"{(train == 'MSI-H').mean():>11.3f}   {(val == 'MSI-H').mean():>9.3f}"
        )
