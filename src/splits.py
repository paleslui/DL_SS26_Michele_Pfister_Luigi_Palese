"""Stratified 5-fold cross-validation splits.

Splits are generated once with a fixed seed and saved to disk so every model
trains and evaluates on identical folds — making the comparison between models
fair and reproducible.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .data_loading import REPO_ROOT


SPLITS_PATH = REPO_ROOT / "data" / "cv_splits.json"
RANDOM_SEED = 42
N_FOLDS = 5


def make_splits(
    labels: pd.Series,
    n_folds: int = N_FOLDS,
    seed: int = RANDOM_SEED,
) -> list[dict]:
    """Generate stratified k-fold splits.

    Returns a list of dicts, one per fold:
        {"fold": int, "train": [...sample_ids...], "val": [...sample_ids...]}
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    sample_ids = labels.index.to_numpy()
    y = (labels == "MSI-H").astype(int).to_numpy()

    folds = []
    for k, (train_idx, val_idx) in enumerate(skf.split(sample_ids, y)):
        folds.append({
            "fold": k,
            "train": sample_ids[train_idx].tolist(),
            "val": sample_ids[val_idx].tolist(),
        })
    return folds


def save_splits(folds: list[dict], path: Path = SPLITS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(folds, f, indent=2)
    print(f"Saved {len(folds)} folds to {path}")


def load_splits(path: Path = SPLITS_PATH) -> list[dict]:
    with path.open() as f:
        return json.load(f)


def summarize_splits(folds: list[dict], labels: pd.Series) -> None:
    """Print per-fold class balance — sanity check that stratification worked."""
    print(f"{'Fold':>4}  {'Train n':>8}  {'Val n':>6}  {'Train MSI-H':>12}  {'Val MSI-H':>10}")
    print("-" * 50)
    for fold in folds:
        train = labels.loc[fold["train"]]
        val = labels.loc[fold["val"]]
        print(
            f"{fold['fold']:>4}  "
            f"{len(train):>8}  {len(val):>6}  "
            f"{(train == 'MSI-H').mean():>11.3f}   {(val == 'MSI-H').mean():>9.3f}"
        )
