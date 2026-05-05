"""Curated gene panels for the transformer model.

The transformer treats each gene as a token, so the panel size directly
controls model capacity and training cost. We curate panels that are
biologically focused on MSI-relevant biology:

  * MSI-H tumors are characterized by mismatch-repair deficiency
    (MMR genes: MLH1, MSH2, MSH6, PMS2, ...)
  * They display elevated immune infiltration / activation
    (cytotoxic, IFN response, antigen presentation, checkpoint genes)
  * They have higher mutation burden affecting specific pathways

The panel is the union of:
  1. Genes from MSI-relevant Hallmark pathways (see HALLMARK_SUBSET below)
  2. A small set of canonical MMR / immune / checkpoint genes
"""
from __future__ import annotations

from .pathways import load_gmt


# Hallmark pathways biologically associated with MSI / immune state
HALLMARK_SUBSET: tuple[str, ...] = (
    # Four most MSI-relevant pathways. Kept small to keep transformer
    # tractable given n=399 training samples — a wider panel risks
    # over-parameterization and is much slower to train.
    "HALLMARK_INTERFERON_GAMMA_RESPONSE",
    "HALLMARK_ALLOGRAFT_REJECTION",
    "HALLMARK_DNA_REPAIR",
    "HALLMARK_INFLAMMATORY_RESPONSE",
)

# Canonical MSI / immune / checkpoint genes — included regardless of Hallmark membership
CANONICAL_GENES: tuple[str, ...] = (
    # Mismatch repair (the core MSI mechanism)
    "MLH1", "MSH2", "MSH6", "PMS2", "PMS1", "MLH3", "EPCAM",
    # Cytotoxic effectors
    "GZMA", "GZMB", "GZMK", "GZMH", "PRF1", "GNLY", "NKG7",
    # T-cell markers
    "CD8A", "CD8B", "CD4", "CD3D", "CD3E", "CD3G", "FOXP3",
    # Immune checkpoints (clinical relevance for pembrolizumab)
    "PDCD1", "CD274", "PDCD1LG2", "CTLA4", "LAG3", "HAVCR2", "TIGIT",
    # Antigen presentation
    "B2M", "HLA-A", "HLA-B", "HLA-C", "TAP1", "TAP2",
    # Interferon signaling
    "IFNG", "STAT1", "IRF1", "IRF8",
)


def build_panel(
    available_genes: set[str],
    hallmark_subset: tuple[str, ...] = HALLMARK_SUBSET,
    canonical_genes: tuple[str, ...] = CANONICAL_GENES,
) -> list[str]:
    """Build the transformer gene panel by union of pathway genes + canonical genes.

    Parameters
    ----------
    available_genes : set
        Genes present in the expression matrix (used to filter).
    hallmark_subset : tuple of str
        Pathway names whose genes are included.
    canonical_genes : tuple of str
        Individual genes always included if present.

    Returns
    -------
    list of gene symbols, sorted alphabetically (deterministic ordering).
    """
    sets = load_gmt()
    panel: set[str] = set()
    for pathway_name in hallmark_subset:
        if pathway_name in sets:
            panel.update(sets[pathway_name])
    panel.update(canonical_genes)
    return sorted(panel & available_genes)
