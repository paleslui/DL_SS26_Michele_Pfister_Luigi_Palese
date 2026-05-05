"""Generate the 5-fold CV splits and save them to data/cv_splits.json.

Run once after data is in place. All training scripts then read the same splits.
"""
import sys
from pathlib import Path

# Make src/ importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loading import load_aligned_dataset
from src.splits import make_splits, save_splits, summarize_splits


def main() -> None:
    print("Loading aligned dataset...")
    _, _, labels = load_aligned_dataset()
    print(f"  {len(labels)} samples with defined MSI status")

    print("\nGenerating stratified 5-fold splits (seed=42)...")
    folds = make_splits(labels)

    print("\nPer-fold summary:")
    summarize_splits(folds, labels)

    save_splits(folds)
    print("\nDone.")


if __name__ == "__main__":
    main()
