# `data/` — inputs

All model inputs. The large expression matrix is **not** in the repo and must be
downloaded (see below); everything else is small enough to track in git.

## Getting the data

`ucec_tpm.csv` (~145 MB, the raw RNA-seq expression matrix) is too large for
GitHub and is git-ignored. Download it and place it in this folder:

- **Download:** https://drive.google.com/file/d/1lNdT3ozx3aD_QkTAamCTdjixGAgNTAi0/view
- **Save as:** `data/ucec_tpm.csv`

Without this file the pipeline cannot run — every representation is built from it.

## Files

| File | Tracked | Description |
|---|---|---|
| `ucec_tpm.csv` | ✗ (download) | Raw RNA-seq expression: ~20 000 genes × 499 samples, TPM units. |
| `ucec_clinical.csv` | ✓ | Clinical annotations incl. the **MSI status label** (the prediction target). |
| `ucec_epic_cellFractions.csv` | ✓ | EPIC-deconvolution estimates: 8 immune/stromal cell fractions per sample. Input for the logistic-regression baseline. |
| `annotation/gencode_v45_gene_positions.csv` | ✓ | Gene → (chromosome, start) map. Used to order genes along the genome for the CNN/LSTM. |
| `genesets/hallmark_v2024.1.Hs.symbols.gmt` | ✓ | MSigDB Hallmark gene sets (50 pathways). Used to build the pathway-score representation. |
| `holdout_split.json` | ✓ | The fixed 80/20 sample IDs for train/test (built by `scripts/01`). |
| `cv_splits.json` | ✓ | The 5-fold cross-validation indices within the train portion. |

## Data provenance

The expression and clinical data are TCGA-UCEC (endometrial carcinoma),
distributed for the course via `acg-team/shared_files` (Kondrateva, 2025):
https://github.com/acg-team/shared_files. The EPIC cell fractions were produced
by EPIC deconvolution of the expression matrix (Pfister, 2026, Tracking Module 1).
Gene positions are from GENCODE v45; pathway definitions from MSigDB Hallmark
v2024.1.

## A note on the "representations"

The four input representations the models use — EPIC fractions, pathway scores,
the 38-gene panel, and chromosome-ordered genes — are **not stored as files**.
Only the EPIC fractions are pre-computed (because they come from an external R
tool). The other three are rebuilt from `ucec_tpm.csv` at runtime, **per
cross-validation fold**, by the code in [`../src/`](../src):

- pathway scores ← TPM + the Hallmark `.gmt`
- 38-gene panel ← TPM, genes listed in `src/gene_panels.py`
- chromosome-ordered ← TPM + the GENCODE positions

This is deliberate: per-fold preprocessing (variance filtering, z-scoring) must
be fit on the training fold only, so caching a single global version would leak
test information. Once `ucec_tpm.csv` is in place, every representation
regenerates automatically.
