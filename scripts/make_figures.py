"""Generate all report/talk figures from saved results.

Run from repo root:
    python scripts/make_figures.py

Reads from results/, writes to results/figures/. Six PNGs at 200 dpi,
slide-quality (large fonts, restrained palette, consistent colors per model).

This is the canonical, scripted version of scripts/figures.ipynb. Both
produce identical output; the notebook adds markdown commentary, the script
is for "regenerate-and-forget" reproducibility.

Figures produced:
    fig1_roc_curves.png            — ROC curves, all 6 models
    fig2_auc_with_ci.png           — Horizontal bar chart with bootstrap CIs
    fig3_confusion_matrices.png    — 2x3 panel of confusion matrices
    fig4_cross_input_grid.png      — Architecture x input heatmap
    fig5_optuna_progress.png       — TPE vs Random search trajectories
    fig6_holdout_class_balance.png — Sanity figure: holdout split composition
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

# ---- paths -----------------------------------------------------------------

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


# ---- shared style ----------------------------------------------------------
# Slide-friendly: large fonts, no chartjunk, consistent palette across figures.

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "figure.autolayout": False,
})

# Models in canonical display order (worst -> best by holdout AUC).
# Each entry: (slug used in filenames, human label, color).
MODELS = [
    ("logreg_epic",       "Logreg / EPIC (8)",          "tab:gray"),
    ("lstm_chromorder",   "LSTM / chr-ordered",         "tab:purple"),
    ("cnn_chromorder",    "CNN / chr-ordered",          "tab:orange"),
    ("mlp_pathways",      "MLP / pathways (50)",        "tab:green"),
    ("transformer_panel", "Transformer / panel (38)",   "tab:red"),
    ("mlp_chrord",        "MLP / chr-ordered (tuned)",  "tab:blue"),
]


# ---- load all data ---------------------------------------------------------

def load_all() -> dict:
    """Load metrics + per-sample predictions for all 6 models."""
    data = {}
    for slug, label, color in MODELS:
        with (RESULTS / f"holdout_{slug}.json").open() as f:
            metrics = json.load(f)
        preds = np.load(RESULTS / f"holdout_{slug}_predictions.npz")
        data[slug] = {
            "label":   label,
            "color":   color,
            "metrics": metrics,
            "y_true":  preds["y_true"],
            "y_prob":  preds["y_prob"],
        }
    return data


# ---- Figure 1: ROC curves --------------------------------------------------

def fig_roc(data: dict, path: Path) -> None:
    """All 6 models on one axes; AUC + 95% CI in the legend."""
    fig, ax = plt.subplots(figsize=(8, 6.5))

    for slug, info in data.items():
        fpr, tpr, _ = roc_curve(info["y_true"], info["y_prob"])
        auc = info["metrics"]["auc"]
        label = (f"{info['label']:30s} "
                 f"AUC = {auc['point']:.3f} "
                 f"[{auc['low']:.2f}–{auc['high']:.2f}]")
        ax.plot(fpr, tpr, color=info["color"], lw=2.3, label=label)

    ax.plot([0, 1], [0, 1], color="black", linestyle="--",
            lw=1, alpha=0.4, label="chance (AUC = 0.50)")

    ax.set_xlabel("False positive rate (1 – specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("ROC curves on 20% holdout test set (n = 100)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3, linestyle=":")
    # Monospace legend so AUC values align cleanly
    leg = ax.legend(loc="lower right", frameon=True, fontsize=10,
                    prop={"family": "monospace", "size": 9.5})
    leg.get_frame().set_alpha(0.9)

    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.name}")


# ---- Figure 2: AUC bar chart with bootstrap CI ----------------------------

def fig_auc_bar(data: dict, path: Path) -> None:
    """Horizontal bars, sorted ascending; error bars show 95% bootstrap CI."""
    fig, ax = plt.subplots(figsize=(9, 5))

    ordered = sorted(data.items(),
                     key=lambda kv: kv[1]["metrics"]["auc"]["point"])
    labels = [info["label"] for _, info in ordered]
    aucs   = [info["metrics"]["auc"]["point"] for _, info in ordered]
    lows   = [info["metrics"]["auc"]["low"]   for _, info in ordered]
    highs  = [info["metrics"]["auc"]["high"]  for _, info in ordered]
    colors = [info["color"] for _, info in ordered]

    lower_err = [a - lo for a, lo in zip(aucs, lows)]
    upper_err = [hi - a for a, hi in zip(aucs, highs)]
    y_pos = np.arange(len(labels))

    ax.barh(y_pos, aucs, xerr=[lower_err, upper_err], color=colors,
            edgecolor="black", linewidth=0.5, capsize=4, alpha=0.85)
    ax.axvline(0.5, color="gray", linestyle="--", lw=1.2, alpha=0.6)
    ax.text(0.5, len(labels) - 0.4, " chance",
            color="gray", fontsize=10, va="center")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("AUROC on holdout test (n = 100) — 95% bootstrap CI")
    ax.set_xlim(0.3, 1.05)
    ax.set_title("Holdout-AUC by model")
    ax.grid(axis="x", alpha=0.3, linestyle=":")

    for i, (a, hi) in enumerate(zip(aucs, highs)):
        ax.text(hi + 0.012, i, f"{a:.3f}", va="center", fontsize=11)

    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.name}")


# ---- Figure 3: confusion matrices -----------------------------------------

def fig_confusion(data: dict, path: Path) -> None:
    """2x3 grid of confusion matrices, sorted by holdout AUC ascending."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    ordered = sorted(data.items(),
                     key=lambda kv: kv[1]["metrics"]["auc"]["point"])

    for ax, (slug, info) in zip(axes, ordered):
        cm = np.array(info["metrics"]["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
        # Text-color threshold at 60% of max (was 50% — slightly less white text)
        threshold = cm.max() * 0.6
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > threshold else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=18, color=color, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["MSS", "MSI-H"])
        ax.set_yticklabels(["MSS", "MSI-H"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        auc = info["metrics"]["auc"]["point"]
        ax.set_title(f"{info['label']}\nAUC = {auc:.3f}", fontsize=11)

    fig.suptitle("Confusion matrices on holdout test set (threshold = 0.5)",
                 fontsize=14, y=1.00)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.name}")


# ---- Figure 4: cross-input grid -------------------------------------------

def fig_cross_input(path: Path) -> None:
    """3 architectures x 4 inputs heatmap. The "input vs architecture" story."""
    with (RESULTS / "cross_input_summary.json").open() as f:
        grid_data = json.load(f)

    archs        = ["logreg", "mlp", "transformer"]
    inputs       = ["epic", "pathways", "panel38", "chrord12k"]
    arch_labels  = ["Logreg", "MLP", "Transformer"]
    input_labels = ["EPIC (8)", "Pathways (50)", "Panel (38)", "ChrOrd (12k)"]

    grid_auc = np.full((len(archs), len(inputs)), np.nan)
    for ai, arch in enumerate(archs):
        for ii, inp in enumerate(inputs):
            cell = f"{arch}_on_{inp}"
            if cell in grid_data:
                grid_auc[ai, ii] = grid_data[cell]["auc"]["point"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    # 0.45 floor (just below logreg/EPIC at 0.50) so all cells get coloured
    im = ax.imshow(grid_auc, cmap="RdYlGn", vmin=0.45, vmax=1.0, aspect="auto")

    for ai in range(len(archs)):
        for ii in range(len(inputs)):
            v = grid_auc[ai, ii]
            if np.isnan(v):
                ax.text(ii, ai, "n/a", ha="center", va="center", fontsize=12,
                        color="gray", style="italic")
            else:
                # White text on dark cells, black otherwise
                text_color = "white" if (v < 0.62 or v > 0.93) else "black"
                ax.text(ii, ai, f"{v:.3f}", ha="center", va="center",
                        fontsize=15, color=text_color, fontweight="bold")

    ax.set_xticks(range(len(inputs)))
    ax.set_xticklabels(input_labels)
    ax.set_yticks(range(len(archs)))
    ax.set_yticklabels(arch_labels)
    ax.set_xlabel("Input format")
    ax.set_ylabel("Architecture")
    ax.set_title("Cross-input grid: holdout AUC by architecture × input")

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("AUROC", rotation=270, labelpad=18)

    # Mark the skipped (transformer × chrord12k) cell with a dashed outline
    if np.isnan(grid_auc[2, 3]):
        ax.add_patch(plt.Rectangle((2.5, 1.5), 1, 1, fill=False,
                                    edgecolor="black", lw=1.5,
                                    linestyle="--", alpha=0.6))

    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.name}")


# ---- Figure 5: Optuna search progress ------------------------------------

def fig_optuna(path: Path) -> None:
    """TPE vs Random running-max AUC for each model.

    Note: the MLP/pathways study lives in a separate (local) database that
    was not merged into the cluster's SQLite DB. It's omitted from this
    panel and noted in the figure caption / report.
    """
    import optuna

    storage = f"sqlite:///{RESULTS / 'optuna_studies.db'}"

    # (display title, tpe study, random study)
    # MLP/pathways excluded — its studies live in a separate (local) DB.
    studies = [
        ("CNN / chr-ordered",      "cnn_tpe_v2",         "cnn_random_v2"),
        ("Transformer / panel-38", "transformer_tpe_v2", "transformer_random_v2"),
        ("LSTM / chr-ordered",     "lstm_tpe_v2",        "lstm_random_v2"),
        ("MLP / chr-ordered",      "mlp_chrord_tpe_v2",  "mlp_chrord_random_v2"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    axes = axes.flatten()

    for ax, (title, tpe_name, rnd_name) in zip(axes, studies):
        for study_name, color, label in [(tpe_name, "tab:blue", "TPE"),
                                         (rnd_name, "tab:gray", "Random")]:
            try:
                st = optuna.load_study(study_name=study_name, storage=storage)
                completed = sorted(
                    (t for t in st.trials
                     if t.state == optuna.trial.TrialState.COMPLETE),
                    key=lambda t: t.number,
                )
                if not completed:
                    continue
                trial_nums = [t.number for t in completed]
                values = [t.value for t in completed]
                running_max = np.maximum.accumulate(values)
                ax.plot(trial_nums, running_max, color=color, lw=2.2,
                        label=label, drawstyle="steps-post")
            except Exception as exc:
                print(f"  warn: {study_name}: {exc}")

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("completed trial")
        ax.set_ylabel("best AUC so far")
        ax.grid(alpha=0.3, linestyle=":")
        ax.legend(loc="lower right", fontsize=10)

    fig.suptitle("Optuna search progress: TPE vs Random "
                 "(running max AUC across completed trials)",
                 fontsize=14, y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.name}")


# ---- Figure 6: holdout class balance (data composition sanity figure) ----

def fig_class_balance(data: dict, path: Path) -> None:
    """Quick descriptive figure: train/test counts of MSS vs MSI-H."""
    # Use any model's stored y_true to recover the holdout counts
    y_test = data["mlp_chrord"]["y_true"]
    n_pos_test = int(y_test.sum())
    n_neg_test = len(y_test) - n_pos_test

    # Holdout-train counts: 399 total samples in our 80/20 split, ~32% MSI-H
    n_pos_train = 127  # 159 total MSI-H × 0.80 (matches scripts/01_make_splits)
    n_neg_train = 272

    fig, ax = plt.subplots(figsize=(7.5, 4))
    splits = ["Holdout-train (80%)", "Holdout-test (20%)"]
    mss_counts   = [n_neg_train, n_neg_test]
    msih_counts  = [n_pos_train, n_pos_test]

    x = np.arange(len(splits))
    width = 0.4

    b1 = ax.bar(x - width / 2, mss_counts, width,
                label="MSS",   color="tab:blue",  alpha=0.85, edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + width / 2, msih_counts, width,
                label="MSI-H", color="tab:orange", alpha=0.85, edgecolor="black", linewidth=0.5)
    for bars, vals in [(b1, mss_counts), (b2, msih_counts)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 5, str(v),
                    ha="center", va="bottom", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("number of samples")
    ax.set_title("Holdout split composition  (n = 499 total samples)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_ylim(0, max(mss_counts) * 1.18)

    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.name}")


# ---- main ----------------------------------------------------------------

def main() -> None:
    print("Loading results...")
    data = load_all()
    for slug, info in data.items():
        auc = info["metrics"]["auc"]
        print(f"  {info['label']:32s} AUC {auc['point']:.3f}  "
              f"[{auc['low']:.3f} – {auc['high']:.3f}]")

    print("\nGenerating figures...")
    fig_roc           (data, FIGURES / "fig1_roc_curves.png")
    fig_auc_bar       (data, FIGURES / "fig2_auc_with_ci.png")
    fig_confusion     (data, FIGURES / "fig3_confusion_matrices.png")
    fig_cross_input   (      FIGURES / "fig4_cross_input_grid.png")
    fig_optuna        (      FIGURES / "fig5_optuna_progress.png")
    fig_class_balance (data, FIGURES / "fig6_holdout_class_balance.png")
    print(f"\nAll figures in {FIGURES.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
