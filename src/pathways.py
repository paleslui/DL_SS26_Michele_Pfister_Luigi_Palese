"""MSigDB Hallmark pathway scoring.

Each tumor's pathway score is the mean of its z-scored expression values for
the genes belonging to that pathway. This gives one summary number per pathway
per sample.

Score interpretation:
  > 0  → genes in this pathway are over-expressed relative to the cohort mean
  < 0  → genes in this pathway are under-expressed
  = 0  → the pathway is at the cohort baseline

CRITICAL: pathway scores must be computed from z-scores fitted on the TRAINING
fold only, never on the full dataset. Otherwise we leak validation data into
the scoring.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data_loading import REPO_ROOT


GMT_PATH = REPO_ROOT / "data" / "genesets" / "hallmark_v2024.1.Hs.symbols.gmt"


# -------- GMT parsing --------------------------------------------------------

def load_gmt(path: Path = GMT_PATH) -> dict[str, list[str]]:
    """Parse a .gmt file → {pathway_name: [gene1, gene2, ...]}.

    GMT format: tab-separated, one pathway per line:
        <name>\t<source/url>\t<gene1>\t<gene2>\t...
    """
    sets: dict[str, list[str]] = {}
    with path.open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            genes = [g for g in parts[2:] if g]  # skip empty trailing fields
            sets[name] = genes
    return sets


# -------- pathway scoring ----------------------------------------------------

def compute_pathway_scores(
    z_expression: pd.DataFrame,
    gene_sets: dict[str, list[str]],
    min_genes_present: int = 5,
) -> pd.DataFrame:
    """Compute pathway scores from a z-scored expression matrix.

    Parameters
    ----------
    z_expression : DataFrame
        Genes (rows) x samples (columns), already z-scored per gene.
    gene_sets : dict
        Pathway name → list of gene symbols.
    min_genes_present : int
        Drop pathways with fewer than this many genes present in z_expression.

    Returns
    -------
    DataFrame of shape (n_samples, n_pathways).
    """
    available_genes = set(z_expression.index)
    scores: dict[str, pd.Series] = {}
    skipped: list[str] = []

    for name, genes in gene_sets.items():
        present = [g for g in genes if g in available_genes]
        if len(present) < min_genes_present:
            skipped.append(f"{name} ({len(present)} genes)")
            continue
        # Mean across the genes that ARE present, per sample
        scores[name] = z_expression.loc[present].mean(axis=0)

    if skipped:
        print(f"  skipped {len(skipped)} pathways with <{min_genes_present} genes present")

    df = pd.DataFrame(scores)  # index = samples, columns = pathways
    df.index.name = "sample"
    df.columns.name = "pathway"
    return df


def gene_set_coverage(
    z_expression: pd.DataFrame,
    gene_sets: dict[str, list[str]],
) -> pd.DataFrame:
    """Diagnostic: report how many genes from each pathway are in the matrix."""
    available = set(z_expression.index)
    rows = []
    for name, genes in gene_sets.items():
        n_total = len(genes)
        n_present = sum(g in available for g in genes)
        rows.append({
            "pathway": name,
            "n_total": n_total,
            "n_present": n_present,
            "coverage": n_present / n_total if n_total else 0.0,
        })
    return pd.DataFrame(rows).sort_values("coverage")
