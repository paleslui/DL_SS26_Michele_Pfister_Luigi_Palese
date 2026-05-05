"""Order genes along the genome by (chromosome, start position).

Used by the chromosome-ordered CNN/RNN models. The expression matrix gets
reshaped from (genes x samples) into (samples x genes_in_genomic_order),
with the gene axis carrying a 1D "position along the genome" structure that
1D convolutions can exploit.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data_loading import REPO_ROOT


GENE_POS_PATH = REPO_ROOT / "data" / "annotation" / "gencode_v45_gene_positions.csv"

# Chromosomes we keep, in genomic order.
# 1..22, X, Y. We exclude the mitochondrial chromosome and unplaced/alt contigs.
CHROMOSOME_ORDER: list[str] = (
    [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
)
_CHR_RANK: dict[str, int] = {c: i for i, c in enumerate(CHROMOSOME_ORDER)}


def load_gene_positions(path: Path = GENE_POS_PATH) -> pd.DataFrame:
    """Return DataFrame with columns: gene_symbol, chromosome, start, end, strand."""
    return pd.read_csv(path)


def order_genes_by_genome(
    expression_genes: list[str],
    positions: pd.DataFrame | None = None,
) -> list[str]:
    """Return a subset of ``expression_genes`` ordered by (chromosome, start).

    Genes missing from the annotation, or on excluded chromosomes (chrM, alt
    contigs, unplaced), are dropped. The returned list is the canonical
    genomic ordering for downstream CNN/RNN input.
    """
    if positions is None:
        positions = load_gene_positions()

    # Restrict annotation to canonical chromosomes
    pos = positions[positions["chromosome"].isin(_CHR_RANK)].copy()

    # If a gene has multiple entries (rare; alternative loci), keep the first
    pos = pos.drop_duplicates(subset="gene_symbol", keep="first")

    # Subset to genes we have expression data for
    pos = pos[pos["gene_symbol"].isin(expression_genes)].copy()

    # Sort by (chromosome rank, start)
    pos["chr_rank"] = pos["chromosome"].map(_CHR_RANK)
    pos = pos.sort_values(["chr_rank", "start"])

    return pos["gene_symbol"].tolist()


def chromosome_block_sizes(
    ordered_genes: list[str],
    positions: pd.DataFrame | None = None,
) -> dict[str, int]:
    """Return {chromosome: number_of_genes_in_block}, in genomic order.

    Useful diagnostic: shows how many genes per chromosome made it into the
    ordered input.
    """
    if positions is None:
        positions = load_gene_positions()
    pos = positions.drop_duplicates(subset="gene_symbol", keep="first")
    pos = pos[pos["gene_symbol"].isin(ordered_genes)]
    counts = pos["chromosome"].value_counts()
    return {c: int(counts.get(c, 0)) for c in CHROMOSOME_ORDER}
