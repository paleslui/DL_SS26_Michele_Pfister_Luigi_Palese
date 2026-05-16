import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# DATA
# =========================

data = {
    "Representation": [
        "EPIC", "EPIC", "EPIC",
        "Pathways", "Pathways", "Pathways",
        "Panel38", "Panel38", "Panel38",
        "ChrOrd12k", "ChrOrd12k", "ChrOrd12k", "ChrOrd12k"
    ],
    "Model": [
        "LogReg", "MLP", "Transformer",
        "LogReg", "MLP", "Transformer",
        "LogReg", "MLP", "Transformer",
        "LogReg", "MLP", "CNN", "LSTM"
    ],
    "AUC": [
        0.50, 0.66, 0.60,
        0.88, 0.90, 0.81,
        0.88, 0.93, 0.92,
        0.94, 0.96, 0.83, 0.79
    ]
}

df = pd.DataFrame(data)

# =========================
# 1. GROUPED BARPLOT
# =========================

representations = df["Representation"].unique()
models = df["Model"].unique()

x = np.arange(len(representations))
width = 0.18

fig, ax = plt.subplots(figsize=(10, 6))

for i, model in enumerate(models):
    vals = []

    for rep in representations:
        subset = df[
            (df["Representation"] == rep) &
            (df["Model"] == model)
        ]

        if len(subset) > 0:
            vals.append(subset["AUC"].values[0])
        else:
            vals.append(np.nan)

    bars = ax.bar(
        x + i * width,
        vals,
        width,
        label=model
    )

    # Add value labels
    for bar in bars:
        height = bar.get_height()

        if not np.isnan(height):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.01,
                f"{height:.2f}",
                ha='center',
                va='bottom',
                fontsize=9
            )

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(representations)

ax.set_ylim(0, 1.05)

ax.set_ylabel("AUC")
ax.set_xlabel("Biological Input Representation")
ax.set_title("MSI Classification Performance Across Models")

ax.legend(title="Architecture")

plt.tight_layout()
plt.show()


# =========================
# 2. REPRESENTATION SUMMARY PLOT
# =========================

summary = df.groupby("Representation")["AUC"].mean().reset_index()

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    summary["Representation"],
    summary["AUC"]
)

# Add average labels
for bar in bars:
    height = bar.get_height()

    ax.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.01,
        f"{height:.2f}",
        ha='center',
        va='bottom',
        fontsize=10
    )

# Add individual model points
for rep in representations:
    vals = df[df["Representation"] == rep]["AUC"]

    x_pos = list(summary["Representation"]).index(rep)

    jitter = np.linspace(-0.08, 0.08, len(vals))

    ax.scatter(
        np.full(len(vals), x_pos) + jitter,
        vals,
        s=60,
        zorder=3
    )

ax.set_ylim(0, 1.05)

ax.set_ylabel("Mean AUC")
ax.set_xlabel("Biological Input Representation")

ax.set_title("Average MSI Classification Performance by Biological Representation")

plt.tight_layout()
plt.show()