# `scripts/` — entry-point pipeline

Numbered scripts that run the project end to end. Run them in order from the
repo root (e.g. `python scripts/01_make_splits.py`). They import the reusable
logic from [`../src/`](../src) and read/write [`../results/`](../results).

## Order of execution

| Script | What it does |
|---|---|
| `01_make_splits.py` | Build the 80/20 holdout split + 5-fold CV indices (`data/*_split.json`). Run once; everything downstream depends on it. |
| `02_train_logreg_epic.py` | Baseline: logistic regression on the 8 EPIC cell fractions. |
| `03_train_mlp_pathways.py` | MLP on the 50 Hallmark pathway scores. |
| `04_train_cnn_chromorder.py` | CNN on chromosome-ordered genes. |
| `05_train_transformer_panel.py` | Transformer on the 38-gene panel. |
| `06_tune_pathway_mlp.py` | Local Optuna search for the pathway-MLP (small enough to run on a laptop). |
| `07_final_holdout_evaluation.py` | **Final unbiased evaluation** of all 5 tuned models on the held-out test set. Produces the headline numbers. |
| `08_cross_input_eval.py` | The architecture × input experiment (every model on every representation). |
| `09_final_mlp_chrord_holdout.py` | Holdout evaluation of the post-hoc-tuned MLP on chr-ordered genes (the winner). |
| `10_transformer_attention.py` | Interpretability: extract the transformer's `[CLS]` attention over the 38 genes → `fig7`. |
| `make_figures.py` | Regenerate figures 1–6 from saved results. (Figure 7 is made by script 10.) |

## "Initial" vs "final" scripts — important

Scripts **02–05** train each model with **default, un-tuned** hyperparameters
and only report cross-validation scores. They are the *first-pass* versions,
kept to show the modelling progression.

The **authoritative results** come from **07** and **09**, which load the
Optuna-tuned hyperparameters (from `results/*_v2_best.json`, produced on the
cluster — see [`../cluster/`](../cluster)) and evaluate once on the untouched
holdout-test set. When a number in the report and a number from an 02–05 script
disagree, the 07/09 number is the correct one.

## Where the tuning happens

These scripts do **not** run the heavy hyperparameter searches — those ran on
the ZHAW SLURM cluster (see [`../cluster/`](../cluster)). The scripts here load
the resulting best-parameter JSON files and do the final training/evaluation.
