"""Generate the holdout split and the 5-fold CV splits.

Run once after data is in place. Produces:
  data/holdout_split.json — 80/20 train/test split (test never used until final reporting)
  data/cv_splits.json     — 5-fold CV inside the 80% train portion

All training scripts read these files so every model trains on identical splits.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loading import load_aligned_dataset
from src.splits import (
    make_holdout, save_holdout, summarize_holdout,
    make_cv_splits, save_splits, summarize_splits,
)


def main() -> None:
    print("Loading aligned dataset...")
    _, _, labels = load_aligned_dataset()
    print(f"  {len(labels)} samples with defined MSI status")

    print("\n80/20 stratified holdout (seed=42):")
    holdout = make_holdout(labels)
    summarize_holdout(holdout, labels)
    save_holdout(holdout)

    print("\n5-fold stratified CV inside the train portion:")
    folds = make_cv_splits(labels, holdout["train"])
    summarize_splits(folds, labels)
    save_splits(folds)

    print("\nDone.")


if __name__ == "__main__":
    main()
