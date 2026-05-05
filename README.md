# Predicting Microsatellite Instability from Bulk RNA-seq

Deep Learning project (HS26) — comparing neural network architectures for
classifying MSI-H vs MSS endometrial tumors (TCGA-UCEC) from bulk RNA-seq.

**Authors:** Luigi Palese, Michèle Pfister
**Module:** Deep Learning, ZHAW Master in Life Sciences

## Background

Builds on Pfister (Tracking Module 1, 2026), which used EPIC linear deconvolution
on the same data and found no significant differences in immune cell *abundance*
between MSI-H and MSS tumors. Hypothesis: the MSI signal lives in pathway-level
*activation* rather than cell counts. We test this with non-linear models.

## Models compared

1. **Logistic regression on EPIC cell fractions** — linear baseline.
2. **MLP on pathway scores** — non-linear, biology-engineered features.
3. **1D CNN on chromosome-ordered gene expression** — local genomic structure.
4. **Transformer on a curated gene panel** — gene-to-gene relationships, interpretable attention.
5. *(Optional)* **LSTM on chromosome-ordered gene expression** — recurrent comparison to CNN.

## Data setup

The expression matrix `ucec_tpm.csv` (~145 MB) is not tracked in git.
Place the following in `data/`:
- `ucec_tpm.csv` — TPM-normalized expression (genes × samples)
- `ucec_tpm.txt` — same data, alternate format
- `ucec_clinical.csv` — clinical annotations including MSI status (tracked)
- `ucec_epic_cellFractions.csv` — EPIC outputs from Pfister (tracked)
- `ucec_merged_clinical_epic.csv` — merged clinical + EPIC (tracked)

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project structure

```
.
├── data/                     # CSV inputs (tpm gitignored)
├── src/
│   ├── data_loading.py       # Load CSVs, align samples
│   ├── preprocessing.py      # log+zscore, pathway scoring
│   ├── splits.py             # Stratified 5-fold CV
│   ├── training.py           # Shared train loop
│   ├── evaluation.py         # Metrics + bootstrap CIs
│   └── models/               # One file per architecture
├── scripts/                  # Entry points (run-everything scripts)
├── notebooks/                # EDA + interpretability
└── results/                  # Outputs (gitignored)
```
