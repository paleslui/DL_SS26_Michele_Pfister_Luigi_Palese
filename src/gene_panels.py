"""Curated gene panels for the transformer model.

The panel size is itself a hyperparameter. For Optuna search we expose four
named panels ranging from focused (MMR + cytotoxic) to broad (multiple immune
Hallmarks):

  * "small"  : MSI core only (MMR genes + cytotoxic effectors + checkpoints)
  * "medium" : MSI core + IFN-gamma response Hallmark
  * "large"  : MSI core + IFN-gamma + DNA-repair + inflammation + allograft
              (this is the original 663-gene panel)
  * "xlarge" : "large" + complement + IL2/STAT5 + IL6/JAK/STAT3 + TNFA + apoptosis
              (broad immune; ~1500 genes, tests if more is better)
"""
from __future__ import annotations

from .pathways import load_gmt


# Pathway groupings for each panel size
_PANELS_PATHWAYS: dict[str, tuple[str, ...]] = {
    "small": (),  # canonical only
    "medium": (
        "HALLMARK_INTERFERON_GAMMA_RESPONSE",
    ),
    "large": (
        "HALLMARK_INTERFERON_GAMMA_RESPONSE",
        "HALLMARK_ALLOGRAFT_REJECTION",
        "HALLMARK_DNA_REPAIR",
        "HALLMARK_INFLAMMATORY_RESPONSE",
    ),
    "xlarge": (
        "HALLMARK_INTERFERON_GAMMA_RESPONSE",
        "HALLMARK_INTERFERON_ALPHA_RESPONSE",
        "HALLMARK_ALLOGRAFT_REJECTION",
        "HALLMARK_DNA_REPAIR",
        "HALLMARK_INFLAMMATORY_RESPONSE",
        "HALLMARK_COMPLEMENT",
        "HALLMARK_IL2_STAT5_SIGNALING",
        "HALLMARK_IL6_JAK_STAT3_SIGNALING",
        "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
        "HALLMARK_APOPTOSIS",
    ),
}

# Canonical MSI / immune / checkpoint genes - always included
CANONICAL_GENES: tuple[str, ...] = (
    # Mismatch repair (the core MSI mechanism)
    "MLH1", "MSH2", "MSH6", "PMS2", "PMS1", "MLH3", "EPCAM",
    # Cytotoxic effectors
    "GZMA", "GZMB", "GZMK", "GZMH", "PRF1", "GNLY", "NKG7",
    # T-cell markers
    "CD8A", "CD8B", "CD4", "CD3D", "CD3E", "CD3G", "FOXP3",
    # Immune checkpoints
    "PDCD1", "CD274", "PDCD1LG2", "CTLA4", "LAG3", "HAVCR2", "TIGIT",
    # Antigen presentation
    "B2M", "HLA-A", "HLA-B", "HLA-C", "TAP1", "TAP2",
    # Interferon signaling
    "IFNG", "STAT1", "IRF1", "IRF8",
)


# ---- panel size dictionary kept for backward-compat with old script ---------

# Old code imports HALLMARK_SUBSET; re-export the "large" panel's pathways
HALLMARK_SUBSET: tuple[str, ...] = _PANELS_PATHWAYS["large"]


def build_panel_sized(
    available_genes: set[str],
    size: str = "large",
) -> list[str]:
    """Build a panel of the given named size.

    Parameters
    ----------
    available_genes : set
        Genes present in the expression matrix (used to filter).
    size : {"small", "medium", "large", "xlarge"}
        Named panel size.

    Returns
    -------
    list of gene symbols, sorted alphabetically (deterministic ordering).
    """
    if size not in _PANELS_PATHWAYS:
        raise ValueError(
            f"Unknown panel size: {size!r}. "
            f"Choices: {list(_PANELS_PATHWAYS)}"
        )

    sets = load_gmt()
    panel: set[str] = set()
    for pathway_name in _PANELS_PATHWAYS[size]:
        if pathway_name in sets:
            panel.update(sets[pathway_name])
    panel.update(CANONICAL_GENES)
    return sorted(panel & available_genes)


def build_panel(
    available_genes: set[str],
    hallmark_subset: tuple[str, ...] = HALLMARK_SUBSET,
    canonical_genes: tuple[str, ...] = CANONICAL_GENES,
) -> list[str]:
    """Backward-compatible wrapper used by the existing scripts/05.

    Equivalent to build_panel_sized(available_genes, "large").
    """
    sets = load_gmt()
    panel: set[str] = set()
    for pathway_name in hallmark_subset:
        if pathway_name in sets:
            panel.update(sets[pathway_name])
    panel.update(canonical_genes)
    return sorted(panel & available_genes)
