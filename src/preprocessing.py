"""Expression preprocessing: log transform, gene filtering, z-scoring.

All transforms are fit on the *training* portion of a CV fold and then applied
to the held-out portion — to avoid information leakage that would otherwise
inflate validation/test metrics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# -------- constants -----------------------------------------------------------

LOG_PSEUDOCOUNT = 1.0  # log2(TPM + 1)


# -------- functions -----------------------------------------------------------

def log_transform(tpm: pd.DataFrame) -> pd.DataFrame:
    """Apply log2(TPM + 1). Operates on a (genes x samples) DataFrame."""
    return np.log2(tpm + LOG_PSEUDOCOUNT)


def filter_low_expression(
    log_tpm: pd.DataFrame,
    min_samples_fraction: float = 0.05,
    min_log_value: float = 1.0,
) -> pd.DataFrame:
    """Drop genes expressed below ``min_log_value`` in fewer than ``min_samples_fraction`` of samples.

    Operates row-wise (genes are rows). With log2(TPM+1), log_value=1 corresponds
    to TPM=1, a standard "expressed" threshold.
    """
    n_samples = log_tpm.shape[1]
    threshold_n = int(np.ceil(min_samples_fraction * n_samples))
    expressed_count = (log_tpm > min_log_value).sum(axis=1)
    keep = expressed_count >= threshold_n
    return log_tpm.loc[keep]


def fit_zscore_params(log_tpm_train: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Compute per-gene mean and std on training samples only."""
    means = log_tpm_train.mean(axis=1)
    stds = log_tpm_train.std(axis=1).replace(0, 1.0)  # guard against zero variance
    return means, stds


def apply_zscore(
    log_tpm: pd.DataFrame,
    means: pd.Series,
    stds: pd.Series,
) -> pd.DataFrame:
    """Apply per-gene z-score using pre-computed train-set parameters."""
    # Align gene order; reindex protects against future mismatches.
    means = means.reindex(log_tpm.index)
    stds = stds.reindex(log_tpm.index)
    return (log_tpm.sub(means, axis=0)).div(stds, axis=0)


def select_top_variable_genes(
    log_tpm_train: pd.DataFrame,
    n_top: int = 5000,
) -> list[str]:
    """Pick the N most-variable genes by variance on the training set.

    Returns a list of gene names in original order (use to subset later).
    """
    variances = log_tpm_train.var(axis=1)
    top = variances.sort_values(ascending=False).head(n_top).index
    # Preserve the original ordering of the matrix (do not re-sort by variance)
    return [g for g in log_tpm_train.index if g in set(top)]


# -------- end-to-end pipeline -------------------------------------------------

def preprocess_for_fold(
    tpm: pd.DataFrame,
    train_samples: list[str],
    eval_samples: list[str],
    n_top_variable: int | None = None,
    filter_low: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply log → filter → (optional) gene selection → z-score, fit on train only.

    Parameters
    ----------
    tpm : (genes x samples) DataFrame of raw TPM values
    train_samples : list of sample IDs used for fitting transforms
    eval_samples : list of sample IDs to apply transforms to (val or test)
    n_top_variable : if set, keep only the top-N variable genes (selected on train)
    filter_low : drop genes expressed in <5% of samples (computed on train)

    Returns
    -------
    train_z : (genes x train_samples) z-scored DataFrame
    eval_z  : (genes x eval_samples)  z-scored DataFrame, same gene set as train_z
    """
    log_all = log_transform(tpm)
    log_train = log_all[train_samples]

    if filter_low:
        log_train = filter_low_expression(log_train)

    if n_top_variable is not None:
        keep_genes = select_top_variable_genes(log_train, n_top=n_top_variable)
        log_train = log_train.loc[keep_genes]

    means, stds = fit_zscore_params(log_train)
    train_z = apply_zscore(log_train, means, stds)

    log_eval = log_all.loc[log_train.index, eval_samples]
    eval_z = apply_zscore(log_eval, means, stds)

    return train_z, eval_z
