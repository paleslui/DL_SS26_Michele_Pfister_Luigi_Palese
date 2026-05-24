"""Transformer interpretability: which genes does the [CLS] token attend to?

The gene-attention transformer (Model 4) treats each gene in the 38-gene panel
as a token and uses a learned [CLS] token for classification. This script backs
the interpretability claim in src/models/gene_transformer.py: we retrain the
transformer on the full 80% holdout-train portion using the tuned v2
hyperparameters, then read the [CLS]->gene attention weights of the final
encoder layer on the held-out test samples.

We report mean attention per gene, separately for true MSI-H and MSS test
samples, and rank genes by how much the model attends to them. If the model
learned biologically sensible structure, the mismatch-repair genes (MLH1, MSH2,
...), cytotoxic effectors (GZMA/B, PRF1) and IFN-gamma genes should rank highly.

Outputs:
    results/transformer_attention.json     per-gene mean attention + ranking
    results/figures/fig7_transformer_attention.png   ranked bar chart
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loading import load_aligned_dataset, REPO_ROOT
from src.preprocessing import log_transform, fit_zscore_params, apply_zscore
from src.gene_panels import build_panel_sized, CANONICAL_GENES
from src.models.gene_transformer import GeneTransformer
from src.training import TrainConfig, train_one_fold, best_device
from sklearn.metrics import roc_auc_score

RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
DEVICE = best_device()
SEED = 42

# Gene-function groups, used only for coloring the figure.
GENE_GROUPS = {
    "MMR / DNA repair":   {"MLH1", "MSH2", "MSH6", "PMS2", "PMS1", "MLH3", "EPCAM"},
    "Cytotoxic effector": {"GZMA", "GZMB", "GZMK", "GZMH", "PRF1", "GNLY", "NKG7"},
    "T-cell marker":      {"CD8A", "CD8B", "CD4", "CD3D", "CD3E", "CD3G", "FOXP3"},
    "Immune checkpoint":  {"PDCD1", "CD274", "PDCD1LG2", "CTLA4", "LAG3", "HAVCR2", "TIGIT"},
    "Antigen present.":   {"B2M", "HLA-A", "HLA-B", "HLA-C", "TAP1", "TAP2"},
    "IFN signaling":      {"IFNG", "STAT1", "IRF1", "IRF8"},
}
GROUP_COLORS = {
    "MMR / DNA repair":   "tab:red",
    "Cytotoxic effector": "tab:orange",
    "T-cell marker":      "tab:green",
    "Immune checkpoint":  "tab:purple",
    "Antigen present.":   "tab:brown",
    "IFN signaling":      "tab:blue",
    "Other":              "tab:gray",
}


def gene_group(gene: str) -> str:
    for grp, members in GENE_GROUPS.items():
        if gene in members:
            return grp
    return "Other"


def load_holdout_split():
    with (REPO_ROOT / "data" / "holdout_split.json").open() as f:
        d = json.load(f)
    return d["train"], d["test"]


def preprocess_transformer(tpm, panel, train_ids, test_ids):
    """Identical to scripts/07's transformer preprocessing."""
    log_panel = log_transform(tpm.loc[panel])
    log_train = log_panel[train_ids]
    means, stds = fit_zscore_params(log_train)
    train_z = apply_zscore(log_train, means, stds)
    test_z = apply_zscore(log_panel[test_ids], means, stds)
    return (train_z[train_ids].to_numpy().T.astype(np.float32),
            test_z[test_ids].to_numpy().T.astype(np.float32))


def main():
    print(f"Device: {DEVICE}", flush=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ---- tuned hyperparameters ----
    with (RESULTS_DIR / "transformer_tpe_v2_best.json").open() as f:
        params = json.load(f)["best_params"]
    print(f"Loaded tuned transformer params (panel={params['panel']}, "
          f"d_model={params['d_model']}, layers={params['n_layers']})")

    # ---- data ----
    print("Loading data...", flush=True)
    tpm, _, labels = load_aligned_dataset()
    train_ids, test_ids = load_holdout_split()
    panel = build_panel_sized(set(tpm.index), size=params["panel"])
    print(f"  panel: {len(panel)} genes")

    X_tr, X_te = preprocess_transformer(tpm, panel, train_ids, test_ids)
    y_tr = (labels.loc[train_ids] == "MSI-H").astype(int).to_numpy()
    y_te = (labels.loc[test_ids] == "MSI-H").astype(int).to_numpy()

    # ---- build + train (mirrors scripts/07 run_transformer) ----
    model = GeneTransformer(
        n_genes=len(panel),
        d_model=params["d_model"], n_heads=params["n_heads"],
        n_layers=params["n_layers"], dim_feedforward=params["dim_feedforward"],
        dropout=params["dropout"],
    )
    cfg = TrainConfig(
        epochs=200, batch_size=params["batch_size"],
        learning_rate=params["learning_rate"], weight_decay=params["weight_decay"],
        patience=30, min_epochs=30,
        scheduler_type=params["scheduler_type"], optimizer=params["optimizer"],
    )
    print("Training transformer on full 80% holdout-train...", flush=True)
    y_prob, history = train_one_fold(model, X_tr, y_tr, X_te, y_te, config=cfg, device=DEVICE)
    auc = roc_auc_score(y_te, y_prob)
    print(f"  holdout AUC = {auc:.3f}  (best epoch {history['best_epoch']}); "
          f"sanity-check against reported 0.923")

    # ---- extract attention on the test set ----
    model.eval()
    X_te_t = torch.tensor(X_te, dtype=torch.float32, device=DEVICE)
    attn = model.cls_attention(X_te_t).cpu().numpy()  # (n_test, n_genes)

    # Mean attention per gene, split by true class
    msih_mask = y_te == 1
    mss_mask = y_te == 0
    mean_all = attn.mean(axis=0)
    mean_msih = attn[msih_mask].mean(axis=0)
    mean_mss = attn[mss_mask].mean(axis=0)

    # Rank genes by attention on MSI-H samples
    order = np.argsort(-mean_msih)
    ranking = [
        {
            "gene": panel[i],
            "group": gene_group(panel[i]),
            "attn_msih": float(mean_msih[i]),
            "attn_mss": float(mean_mss[i]),
            "attn_all": float(mean_all[i]),
        }
        for i in order
    ]

    print("\nTop 12 genes by [CLS] attention on MSI-H test samples:")
    print(f"  {'gene':<10s} {'group':<20s} {'attn(MSI-H)':>12s} {'attn(MSS)':>11s}")
    uniform = 1.0 / len(panel)
    for r in ranking[:12]:
        print(f"  {r['gene']:<10s} {r['group']:<20s} "
              f"{r['attn_msih']:>12.4f} {r['attn_mss']:>11.4f}")
    print(f"  (uniform attention baseline = {uniform:.4f})")

    # ---- save JSON ----
    out = {
        "panel_size": params["panel"],
        "n_genes": len(panel),
        "holdout_auc_this_run": float(auc),
        "uniform_baseline": uniform,
        "n_test_msih": int(msih_mask.sum()),
        "n_test_mss": int(mss_mask.sum()),
        "ranking": ranking,
    }
    with (RESULTS_DIR / "transformer_attention.json").open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results/transformer_attention.json")

    # ---- figure ----
    make_figure(ranking, uniform, FIGURES_DIR / "fig7_transformer_attention.png")
    print(f"Saved results/figures/fig7_transformer_attention.png")


def make_figure(ranking, uniform, path, top_n=20):
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
    })

    top = ranking[:top_n][::-1]  # reverse so highest is at top of barh
    genes = [r["gene"] for r in top]
    vals = [r["attn_msih"] for r in top]
    colors = [GROUP_COLORS[r["group"]] for r in top]

    fig, ax = plt.subplots(figsize=(8, 8))
    y = np.arange(len(genes))
    ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.axvline(uniform, color="black", linestyle="--", lw=1.2, alpha=0.6)
    ax.text(uniform, len(genes) - 0.5, " uniform", fontsize=10,
            color="black", va="center")

    ax.set_yticks(y)
    ax.set_yticklabels(genes)
    ax.set_xlabel("Mean [CLS] attention on MSI-H test samples")
    ax.set_title(f"Transformer gene attention (top {top_n} of {len(ranking)})")
    ax.grid(axis="x", alpha=0.3, linestyle=":")

    # Legend for gene groups actually present in the top_n
    from matplotlib.patches import Patch
    groups_present = []
    for r in top:
        if r["group"] not in groups_present:
            groups_present.append(r["group"])
    handles = [Patch(facecolor=GROUP_COLORS[g], edgecolor="black", label=g)
               for g in groups_present]
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.9)

    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
