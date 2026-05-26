"""Load and align TCGA-UCEC expression and clinical data.

Data files expected in ``data/``:
    - ucec_tpm.csv               : TPM matrix (genes x samples)
    - ucec_clinical.csv          : Clinical table with MSI column
    - ucec_epic_cellFractions.csv: Pfister's EPIC outputs (samples x cell types)

The function ``load_aligned_dataset`` returns three DataFrames whose rows are
the samples that have BOTH expression data AND a defined MSI status (MSS or
MSI-H - empty/NA values dropped).
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd


# -------- paths ---------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

TPM_PATH = DATA_DIR / "ucec_tpm.csv"
CLINICAL_PATH = DATA_DIR / "ucec_clinical.csv"
EPIC_PATH = DATA_DIR / "ucec_epic_cellFractions.csv"


# -------- loaders -------------------------------------------------------------

def load_tpm() -> pd.DataFrame:
    """Load the TPM matrix. Returns a DataFrame with genes as rows and samples as columns."""
    if not TPM_PATH.exists():
        raise FileNotFoundError(
            f"{TPM_PATH} not found. The file is gitignored - copy it from your "
            "local backup or the course Drive."
        )
    df = pd.read_csv(TPM_PATH, index_col=0)
    df.index.name = "gene"
    df.columns.name = "sample"
    return df


def load_clinical() -> pd.DataFrame:
    """Load clinical annotations. Indexed by sample_submitter_id."""
    df = pd.read_csv(CLINICAL_PATH)
    df = df.set_index("sample_submitter_id")
    return df


def load_epic() -> pd.DataFrame:
    """Load EPIC cell fraction outputs. Indexed by sample_submitter_id."""
    df = pd.read_csv(EPIC_PATH)
    df = df.set_index("sample_submitter_id")
    # Drop the patient_id helper column if present
    df = df.drop(columns=[c for c in ("patient_id",) if c in df.columns])
    return df


# -------- alignment -----------------------------------------------------------

def load_aligned_dataset(
    drop_unknown_msi: bool = True,
    msi_classes: tuple[str, ...] = ("MSS", "MSI-H"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load TPM, EPIC, and clinical data and align them on common samples.

    Parameters
    ----------
    drop_unknown_msi : bool
        If True (default), drop samples whose MSI value is not in ``msi_classes``.
    msi_classes : tuple of str
        MSI labels to keep. Default ("MSS", "MSI-H").

    Returns
    -------
    tpm : pd.DataFrame
        Genes (rows) x samples (columns), aligned to the kept samples.
    epic : pd.DataFrame
        Samples (rows) x cell types (columns).
    labels : pd.Series
        MSI labels indexed by sample.
    """
    tpm = load_tpm()
    clinical = load_clinical()
    epic = load_epic()

    # Find the intersection of samples present in all three sources
    tpm_samples = set(tpm.columns)
    epic_samples = set(epic.index)
    clinical_samples = set(clinical.index)
    common = sorted(tpm_samples & epic_samples & clinical_samples)

    if not common:
        raise RuntimeError(
            "No samples are common to TPM, EPIC, and clinical data. "
            "Check that sample IDs use the same TCGA barcode format."
        )

    # Pull MSI labels and (optionally) drop unknown values
    labels = clinical.loc[common, "MSI"].astype(str).str.strip()
    labels = labels.replace({"": pd.NA, "nan": pd.NA, "NA": pd.NA})

    if drop_unknown_msi:
        keep = labels.isin(msi_classes)
        labels = labels[keep]
        common = labels.index.tolist()

    # Subset everything to the kept samples
    tpm = tpm[common]
    epic = epic.loc[common]
    labels = labels.loc[common]
    labels.name = "MSI"

    return tpm, epic, labels


# -------- diagnostics ---------------------------------------------------------

def summarize() -> None:
    """Print a quick summary of the aligned dataset (sanity check)."""
    tpm, epic, labels = load_aligned_dataset()
    print(f"Aligned samples : {len(labels)}")
    print(f"Genes (TPM rows): {tpm.shape[0]}")
    print(f"EPIC features   : {epic.shape[1]}  ({list(epic.columns)})")
    print(f"\nMSI label counts:")
    print(labels.value_counts().to_string())
    print(f"\nClass balance (MSI-H fraction): {(labels == 'MSI-H').mean():.3f}")


if __name__ == "__main__":
    summarize()
