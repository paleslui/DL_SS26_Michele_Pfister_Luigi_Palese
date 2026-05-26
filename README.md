# Predicting Microsatellite Instability from Bulk RNA-seq

**Deep Learning project (Spring 2026)** — comparing five neural-network architectures for
classifying microsatellite-instability-high (MSI-H) versus microsatellite-stable (MSS)
endometrial tumors (TCGA-UCEC) from bulk RNA-seq expression data.

**Research question:** *Which feature representation and model architecture
perform best at classifying MSI-H vs MSS from bulk RNA-seq?*

**Authors:** Luigi Palese, Michèle Pfister
**Module:** Deep Learning, ZHAW Master in Life Sciences
**Submission deadline:** 26.05.2026

---

## Headline result

| # | Model | Input | Holdout AUC | 95% CI |
|---|---|---|---|---|
| 1 | Logistic regression | EPIC cell fractions (8) | 0.497 | [0.372, 0.614] |
| 2 | MLP | Hallmark pathway scores (50) | 0.900 | [0.832, 0.956] |
| 3 | 1D CNN | Chromosome-ordered genes (12 k) | 0.835 | [0.751, 0.904] |
| 4 | Gene-attention Transformer | Curated 38-gene panel | 0.923 | [0.851, 0.980] |
| 5 | BiLSTM | Chromosome-ordered genes (20 k) | 0.789 | [0.691, 0.871] |
| **6** | **MLP (tuned for 12 k input)** | **Chromosome-ordered genes (8 k)** | **0.958** | **[0.917, 0.988]** |

All numbers are unbiased holdout estimates on a 100-sample test set that was
never touched during hyperparameter search.

See [`results/figures/`](results/figures/) for visualizations.

---

## Background

This project builds on Pfister's prior work (Tracking Module 1, 2026), which used
the EPIC linear deconvolution method on the same TCGA-UCEC cohort. That study
found **no significant differences in immune-cell *abundance* between MSI-H and
MSS tumors**.

Our hypothesis was that the MSI signal lives in pathway-level *activation*
rather than cell counts — i.e. the same cells do different things in MSI-H
vs MSS tumors, even if they are present in similar proportions. We test this by
training non-linear models on increasingly rich representations of the
expression matrix, from the EPIC fractions Pfister already had (8 features) up
through chromosome-ordered raw expression (12 000+ features).

**Key finding:** The MSI signal is broadly distributed across the transcriptome.
A simple MLP on 8 000 most-variable chromosome-ordered genes (AUC 0.958)
outperforms a sophisticated transformer on a hand-curated 38-gene panel
(AUC 0.923). With proper tuning, **the input format matters more than the
architecture**.

---

## Repository structure

```
.
├── README.md                            ← you are here
├── requirements.txt                     ← Python dependencies
├── .gitignore                           ← excludes large data + build outputs
│
├── data/                                ← CSV inputs (TPM matrix not tracked, 145 MB)
│   ├── ucec_tpm.csv                       expression matrix (genes × samples)
│   ├── ucec_clinical.csv                  clinical annotations incl. MSI status
│   ├── ucec_epic_cellFractions.csv        EPIC immune-cell fractions
│   ├── ucec_merged_clinical_epic.csv      joined convenience file
│   ├── holdout_split.json                 80/20 sample IDs (created by script 01)
│   ├── cv_splits.json                     5-fold CV inside the 80% (created by script 01)
│   ├── annotation/
│   │   └── gencode_v45_gene_positions.csv  chromosome coordinates for ordering
│   └── genesets/
│       └── hallmark_v2024.1.Hs.symbols.gmt MSigDB Hallmark gene sets
│
├── src/                                 ← reusable library code
│   ├── data_loading.py                    load CSVs, intersect samples, return aligned dataset
│   ├── preprocessing.py                   log+filter+zscore, pathway-score computation
│   ├── splits.py                          load the JSON split files
│   ├── training.py                        shared PyTorch train loop, early stopping
│   ├── evaluation.py                      metrics + bootstrap CIs
│   ├── pathways.py                        load MSigDB Hallmark gene sets
│   ├── gene_panels.py                     curated MMR + immune gene panels
│   ├── genome_ordering.py                 sort genes by chromosomal position
│   └── models/                            one .py per architecture
│       ├── mlp.py                           tabular MLP
│       ├── cnn1d.py                         1D CNN with flexible depth + pooling
│       ├── gene_transformer.py              transformer with gene-identity embeddings
│       └── lstm_chrom.py                    chunked bidirectional LSTM
│
├── scripts/                             ← entry-point scripts (run in numerical order)
│   ├── 01_make_splits.py                  build holdout_split.json + cv_splits.json
│   ├── 02_train_logreg_epic.py            train + evaluate baseline logistic regression
│   ├── 03_train_mlp_pathways.py           train pathway-MLP (initial, un-tuned version)
│   ├── 04_train_cnn_chromorder.py         train CNN (initial, un-tuned version)
│   ├── 05_train_transformer_panel.py      train transformer (initial, un-tuned version)
│   ├── 06_tune_pathway_mlp.py             local Optuna search for the pathway-MLP
│   ├── 07_final_holdout_evaluation.py     final unbiased eval of all 5 tuned models
│   ├── 08_cross_input_eval.py             the architecture × input grid experiment
│   ├── 09_final_mlp_chrord_holdout.py     holdout eval for the post-hoc-tuned MLP/chrord
│   ├── 10_transformer_attention.py        extract [CLS] attention → interpretability figure
│   └── make_figures.py                    regenerate figures 1–6 (fig7 comes from script 10)
│
├── cluster/                             ← scripts to run on the ZHAW Earth-5 SLURM cluster
│   ├── optuna_search.py                   generic Optuna search worker (CNN/transformer/LSTM/MLP-chrord)
│   ├── analyze_results.py                 post-hoc summary of all Optuna studies
│   ├── search_cnn_tpe.sh                   1000-trial TPE search for the CNN
│   ├── search_cnn_random.sh                200-trial random-search baseline
│   ├── search_transformer_tpe.sh           300-trial TPE search for the transformer
│   ├── search_transformer_random.sh        100-trial random-search baseline
│   ├── search_lstm_tpe.sh                  300-trial TPE search for the LSTM
│   ├── search_lstm_random.sh               100-trial random-search baseline
│   ├── search_mlp_chrord.sh                200-trial TPE + 75 random for MLP on chr-ordered
│   ├── final_mlp_chrord_holdout.sh         holdout eval for the post-hoc MLP/chrord (SLURM wrapper)
│   ├── install_dl_msi_env.sh               one-off env setup on the cluster
│   └── dl_msi_env.yml                      conda environment definition
│
└── results/                             ← model outputs (mostly gitignored)
    ├── <model>_tpe_v2_best.json          best hyperparameters per study (tracked)
    ├── <model>_random_v2_best.json       random-search baseline best (tracked)
    ├── holdout_<model>.json              final holdout metrics + CIs (tracked)
    ├── holdout_<model>_predictions.npz   per-sample y_true + y_prob (gitignored)
    ├── cross_input_summary.json          full 3×4 grid output (tracked)
    ├── transformer_attention.json        per-gene [CLS] attention ranking (tracked)
    ├── optuna_studies.db                 SQLite of every Optuna trial (gitignored, ~10 MB)
    └── figures/                          generated PNGs (tracked)
        ├── fig1_roc_curves.png
        ├── fig2_auc_with_ci.png
        ├── fig3_confusion_matrices.png
        ├── fig4_cross_input_grid.png
        ├── fig5_optuna_progress.png
        ├── fig6_holdout_class_balance.png
        └── fig7_transformer_attention.png
```

---

## Pipeline overview

```
ucec_tpm.csv            ┐
ucec_clinical.csv       ┤── load_aligned_dataset()  →  499 samples × MSI label
ucec_epic_cellFractions ┘     (340 MSS, 159 MSI-H — Pfister-confirmed counts)

                                    ↓

                          scripts/01_make_splits.py
                                    ↓
              holdout_split.json    +    cv_splits.json
              (399 train / 100 test)     (5 stratified folds in the 80%)

                                    ↓

           Per-fold preprocessing (NEVER global, prevents leakage):
              ┌─ log₂(TPM + 1)
              ├─ filter low-expression (train-only threshold)
              ├─ (CNN/LSTM only) chromosome-order, then top-N variance
              └─ per-gene z-score (train-fit μ/σ)

                                    ↓ ↓ ↓ ↓ ↓

           ┌────────────┬────────────┬────────────┬────────────┐
           │  Input A   │  Input B   │  Input C   │  Input D   │
           │ EPIC (8)   │ Pathway 50 │ Panel 38   │ Chrord 12k │
           └────────────┴────────────┴────────────┴────────────┘
                  │            │            │            │
                  ▼            ▼            ▼            ▼
              Logreg        MLP        Transformer    CNN + LSTM

                                    ↓

           Per-model Optuna search (cluster) →  best hyperparameters

                                    ↓

           scripts/07 + 09  → final clean training on FULL 80%
                            → single evaluation on UNSEEN 20%
                            → bootstrap 95% CI on the 100 predictions

                                    ↓

                       results/holdout_*.json
                       results/figures/*.png
```

The 20% holdout-test set is never seen by any model or hyperparameter search.
It is only touched by scripts 07 and 09 for the final unbiased evaluation.

---

## Feature representations

Every representation below is derived from the **same** expression matrix
(`ucec_tpm.csv`); they differ in how much of it the model sees and how it is
organized. They range from the most compressed and interpretable to the most
raw and high-dimensional:

| Representation | Size | What it encodes | Why we included it |
|---|---|---|---|
| EPIC cell fractions | 8 | Estimated immune/stromal cell proportions | Pfister's prior input — the baseline our hypothesis says should fail. |
| Hallmark pathway scores | 50 | Mean expression per biological program | Tests the "activation, not abundance" hypothesis at the pathway level. |
| Curated gene panel | 38 | Hand-picked MSI/immune genes | Tests whether a small, biology-driven feature set is enough. |
| Chromosome-ordered genes | ~8–22 k | Most-variable genes in genomic order | The raw, minimally-engineered signal; lets CNN/LSTM exploit genomic locality. |

**Why 50 pathways.** The 50 is not a number we chose — it is the *complete*
MSigDB Hallmark v2024.1 collection (50 curated gene sets covering the major
biological programs). Each sample is summarized as the mean z-scored expression
of the genes in each set, giving 50 "how active is this program" features.

**Why 38 genes (and why that exact set).** The panel is built from canonical MSI
biology (`src/gene_panels.py`): mismatch-repair genes (MLH1, MSH2, MSH6, PMS2,
PMS1, MLH3, EPCAM), cytotoxic effectors (granzymes, PRF1, GNLY, NKG7), T-cell
markers, immune checkpoints (PD-1, PD-L1, CTLA4, LAG3, TIGIT), antigen
presentation (B2M, HLA-A/B/C, TAP1/2), and interferon signalling (IFNG, STAT1,
IRF1/8) — 38 genes total. The panel *size* was itself a tuned hyperparameter
(four tiers: `small`/`medium`/`large`/`xlarge`, adding progressively more immune
Hallmark sets). The focused `small` tier — these 38 canonical genes — won, so
adding more genes did not help the transformer.

**How the chromosome-ordered subset is selected.** Two steps, both fit on the
training fold only (no leakage): (1) **variance filter** — keep the top-N
most-variable genes, where N is a tuned hyperparameter; (2) **genomic ordering** —
sort those genes by (chromosome, start position) using GENCODE v45, so adjacent
positions are physical neighbors the CNN/LSTM can exploit. The tuned N differs by
model: **CNN → 12 000, LSTM → 20 000, winning MLP → 8 000**. Because the variance
filter is recomputed per fold, the *exact* gene set differs slightly across folds —
there is no single fixed gene list, by design.

**Which model uses which** is shown in the table below (and compared head-to-head
in the cross-input experiment further down).

---

## Models

All five architectures share the same preprocessing pipeline (above) but differ in
what they receive as input, and in their inductive bias.

| Model | Input | Architecture (tuned) | Parameters | Holdout AUC |
|---|---|---|---|---|
| **1. Logistic regression** | 8 EPIC cell fractions | sklearn LogisticRegression, balanced class weight | 9 | 0.497 |
| **2. Pathway MLP** | 50 Hallmark pathway scores | 50 → 32 → 64 → 1, dropout 0.33, AdamW | 3.8 K | 0.900 |
| **3. CNN** | 12 000 chr-ordered genes | 4 conv blocks, kernel 71, base 16, avg-pool/4, Adam | 782 K | 0.835 |
| **4. Transformer** | 38-gene curated panel | d_model 192, 4 heads, 3 layers, plateau-LR, AdamW | 1.15 M | 0.923 |
| **5. LSTM** | 20 000 chr-ordered genes, chunked into 200 windows of 100 | bidirectional, hidden 128, 1 layer, Adam | 183 K | 0.789 |
| **6. MLP (chr-ordered)** | 8 000 chr-ordered genes | 8000 → 128 → 1, heavy L2 (wd 0.066), Adam | 1.02 M | **0.958** |

Model 6 is the same MLP architecture family as Model 2, but tuned for the
high-dimensional 8 000-gene input rather than the 50-dim pathway input. It
emerged from the cross-input experiment described below.

### The cross-input experiment

To disentangle the contributions of architecture and input format,
`scripts/08_cross_input_eval.py` evaluates three architectures
(logreg, MLP, transformer) on each of four input formats, with the tuned
hyperparameters of each architecture held fixed and only the input changing.

The result (`results/figures/fig4_cross_input_grid.png`) shows that AUC
increases monotonically as one moves from 8 → 50 → 38 → 12000 input
features, for *every* architecture tested. The architecture's choice
matters far less than the input format.

(One cell — transformer on 12 000 chr-ordered genes — was not evaluated
because attention's O(n²) memory at our tuned hyperparameters exceeded
the A100's 40 GB. Adapting the architecture to fit would have changed
what was being tested.)

### Interpretability: what the transformer attends to

`scripts/10_transformer_attention.py` retrains the transformer with its tuned
hyperparameters and extracts the [CLS] token's attention over the 38 panel
genes on the holdout-test set (via `GeneTransformer.cls_attention`). The result
(`results/figures/fig7_transformer_attention.png`) shows the model concentrates
attention on the biologically expected genes: the top-attended are **IFNG, PMS2,
GZMB, CD274, GNLY** — i.e. interferon-γ signalling, a mismatch-repair gene, a
cytotoxic effector, the PD-L1 checkpoint, and granulysin. Attention on these is
4–5× the uniform baseline (1/38 = 0.026).

Splitting attention by true class shows the model weights the mismatch-repair
and antigen-presentation machinery (EPCAM, MSH2, MLH1, HLA-A/B/C, B2M) more
heavily on MSI-H samples, and cytotoxic effectors more on MSS — evidence the
model keys on the MSI mechanism itself, not just a generic inflammation signal.

---

## Methodology

### Train / test split
- **80 / 20 stratified holdout** at the sample level, preserving MSI-H / MSS
  proportions (399 train / 100 test, ~32% MSI-H in both).
- **5-fold stratified cross-validation inside the 80%** for hyperparameter search.
  Each Optuna trial trains 5 models (~319 train + ~80 val per fold) and reports
  the mean cross-validation AUROC.

This nested design means the 100-sample holdout-test estimates are unbiased: they
reflect generalization to genuinely unseen samples, not the artifact of having
selected hyperparameters that overfit the cross-validation folds.

### Preprocessing
All preprocessing is performed **per fold using only training-fold statistics**:

1. `log₂(TPM + 1)` transformation
2. Low-expression gene filter (genes expressed in ≥ 5% of training samples)
3. (For CNN / LSTM:) chromosome ordering using GENCODE v45 annotations
4. Top-N variance selection (when applicable), N treated as a hyperparameter
5. Per-gene z-score using train-fold mean and standard deviation

### Hyperparameter search

`cluster/optuna_search.py` implements a generic Optuna worker that supports all
four tunable architectures. For each model we ran:

- **TPE** (Tree-structured Parzen Estimator): `n_startup_trials=50`,
  `multivariate=True`, 48 expected-improvement candidates per acquisition step.
- **Random search** as a methodological baseline.

Trials are pruned at fold-level granularity using a median-stopping rule: after
each completed CV fold, the running mean AUC is compared to the median of
previously-completed trials at the same fold step, and underperforming trials
are stopped. Across the wider search spaces this achieved ~70% pruning,
substantially reducing GPU time.

| Search | Trials | Pruned | Best CV AUC |
|---|---|---|---|
| CNN / TPE | 1000 | 703 | 0.861 |
| CNN / Random | 200 | 156 | 0.817 |
| Transformer / TPE | 300 | 117 | 0.970 |
| Transformer / Random | 100 | 59 | 0.971 |
| LSTM / TPE | 300 | 79 | 0.813 |
| LSTM / Random | 100 | 32 | 0.804 |
| MLP-chrord / TPE | 200 | 143 | 0.974 |
| MLP-chrord / Random | 75 | 52 | 0.967 |
| MLP-pathways / TPE (local) | 50 | 4 | 0.819 |
| MLP-pathways / Random (local) | 25 | 7 | 0.808 |

Storage is SQLite (`results/optuna_studies.db`), so any search can be killed and
resumed without losing trials.

### Final evaluation

For each architecture, `scripts/07` (and `scripts/09` for the MLP-chrord) does:

1. Load the chosen hyperparameters from `<model>_tpe_v2_best.json`.
2. Train one final model on the full 399-sample holdout-train portion.
3. Predict on the 100-sample holdout-test (never previously seen).
4. Compute AUROC, F1, confusion matrix, and 95% bootstrap CIs (2000 resamples).
5. Save `holdout_<model>.json` and `holdout_<model>_predictions.npz`.

---

## Reproducing the results

### Local setup (Mac / Linux)

```bash
git clone https://github.com/paleslui/DL_SS26_Michele_Pfister_Luigi_Palese.git
cd DL_SS26_Michele_Pfister_Luigi_Palese

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The TPM expression matrix `data/ucec_tpm.csv` (~145 MB) is not tracked in git
(too large for GitHub). Download it and place it at `data/ucec_tpm.csv` before
running any script that touches expression data:

- **Download:** https://drive.google.com/file/d/1lNdT3ozx3aD_QkTAamCTdjixGAgNTAi0/view

All other inputs (clinical labels, EPIC fractions, gene positions, gene sets)
are already in the repo. See [`data/README.md`](data/README.md) for details on
each file and data provenance ([`acg-team/shared_files`](https://github.com/acg-team/shared_files)).

### Run the full local pipeline

```bash
# Build the deterministic train/test splits (once)
python scripts/01_make_splits.py

# Train and evaluate each model with default hyperparameters
python scripts/02_train_logreg_epic.py
python scripts/03_train_mlp_pathways.py
python scripts/04_train_cnn_chromorder.py
python scripts/05_train_transformer_panel.py

# Hyperparameter search for the pathway-MLP (small enough to run locally)
python scripts/06_tune_pathway_mlp.py

# Final unbiased evaluation on the holdout-test set
python scripts/07_final_holdout_evaluation.py

# Cross-input experiment
python scripts/08_cross_input_eval.py

# Final eval for the post-hoc-tuned MLP/chrord
python scripts/09_final_mlp_chrord_holdout.py

# Interpretability: extract the transformer's [CLS] attention over the 38 genes
python scripts/10_transformer_attention.py

# Regenerate figures 1–6 (fig7 is produced by script 10 above)
python scripts/make_figures.py
```

### Run the cluster hyperparameter searches

The CNN, transformer, LSTM, and MLP-chrord searches were run on the ZHAW Earth-5
SLURM cluster (NVIDIA A100 PCIe 40 GB). To reproduce:

```bash
# On the cluster:
cd /cfs/earth/scratch/<user>/<workspace>
git clone <this repo>
cd DL_SS26_Michele_Pfister_Luigi_Palese
bash cluster/install_dl_msi_env.sh   # one-off, ~15-30 min

# Submit each search (each writes to results/optuna_studies.db)
sbatch cluster/search_cnn_tpe.sh
sbatch cluster/search_cnn_random.sh
sbatch cluster/search_transformer_tpe.sh
sbatch cluster/search_transformer_random.sh
sbatch cluster/search_lstm_tpe.sh
sbatch cluster/search_lstm_random.sh
sbatch cluster/search_mlp_chrord.sh

# (optional) final holdout eval for MLP-chrord on the cluster
sbatch cluster/final_mlp_chrord_holdout.sh
```

Total cluster time: roughly 20-25 GPU-hours wall-clock for all 8 searches.

---

## Key files (what produces what)

| Output | Produced by |
|---|---|
| `data/holdout_split.json`, `data/cv_splits.json` | `scripts/01_make_splits.py` |
| `results/<model>_tpe_v2_best.json` | `cluster/optuna_search.py` (one study per model) |
| `results/<model>_random_v2_best.json` | same — `--sampler random` |
| `results/holdout_<model>.json` | `scripts/07_final_holdout_evaluation.py` |
| `results/holdout_mlp_chrord.json` | `scripts/09_final_mlp_chrord_holdout.py` |
| `results/cross_input_summary.json` | `scripts/08_cross_input_eval.py` |
| `results/transformer_attention.json` + `fig7` | `scripts/10_transformer_attention.py` |
| `results/figures/fig1`–`fig6` | `scripts/make_figures.py` |
| `results/optuna_studies.db` | every cluster search writes here |

---

## Tested versions

- Python 3.11+
- PyTorch 2.5.1 + CUDA 12.4 on the cluster; PyTorch 2.5+ MPS on Mac
- scikit-learn ≥ 1.5, numpy ≥ 2.1, pandas ≥ 2.2
- Optuna 4.0
- See `requirements.txt` for the full list

---

## References

- Pfister, M. (2026). *Comparative immune-cell deconvolution of TCGA-UCEC tumors
  across MSI status using EPIC.* Tracking Module 1, ZHAW Master in Life Sciences.
  — Source of the 499-sample cohort, MSI labels, and EPIC cell-fraction features
  (Model 1), and the basis for our hypothesis that signal lives in pathway
  activation rather than immune-cell abundance.
- Kondrateva, O. (2025). *UCEC clinical data* [Dataset].
  https://github.com/acg-team/shared_files
- *ucec_tpm* (n.d.). [CSV file]. Google Drive. Retrieved 26 May 2026, from
  https://drive.google.com/file/d/1lNdT3ozx3aD_QkTAamCTdjixGAgNTAi0/view
- Palese, L. (2026). *DL_SS26_Michele_Pfister_Luigi_Palese* [Code repository].
  https://github.com/paleslui/DL_SS26_Michele_Pfister_Luigi_Palese
- Denu, R. (2026). *DNA mismatch repair deficiency* [BioRender template].
  https://app.biorender.com (used for presentation figures only).

### Methods & tools

- Liberzon, A. et al. (2015). *The Molecular Signatures Database (MSigDB)
  Hallmark gene set collection.* Cell Systems 1(6), 417–425. — pathway
  representation (Hallmark v2024.1).
- Racle, J. et al. (2017). *EPIC: Estimating the proportion of immune and cancer
  cells from bulk tumor gene expression data.* eLife 6, e26476. — cell-fraction
  deconvolution.
- Frankish, A. et al. (2021). *GENCODE reference annotation.* Nucleic Acids
  Research 49(D1), D916–D923. — gene genomic positions (v45).
- Akiba, T. et al. (2019). *Optuna: A next-generation hyperparameter
  optimization framework.* KDD 2019. — hyperparameter search.
