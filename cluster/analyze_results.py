"""Read the Optuna SQLite database and print a summary of all studies.

Saves the best hyperparameters of each study as JSON in results/, ready to be
loaded by a final clean training script. Run after the search jobs finish:

    python cluster/analyze_results.py

It does not need a GPU — runs fine on the login node or any laptop with the
project repo + SQLite database file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import optuna

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
RESULTS_DIR = REPO_ROOT / "results"
DB_PATH = RESULTS_DIR / "optuna_studies.db"


def main() -> None:
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        sys.exit(1)

    storage = f"sqlite:///{DB_PATH}"
    summaries = optuna.study.get_all_study_summaries(storage)
    if not summaries:
        print("No studies in database.")
        return

    print(f"Studies in {DB_PATH.name}:\n")
    print(f"  {'Study':30s}  {'#trials':>8s}  {'#pruned':>8s}  {'best AUC':>9s}")
    print("  " + "-" * 60)
    for s in summaries:
        n_total = s.n_trials
        try:
            best = s.best_trial.value if s.best_trial else None
        except Exception:
            best = None
        # Reload to count pruned (summary doesn't include it directly)
        st = optuna.load_study(study_name=s.study_name, storage=storage)
        n_pruned = sum(1 for t in st.trials if t.state == optuna.trial.TrialState.PRUNED)
        best_str = f"{best:.4f}" if best is not None else "-"
        print(f"  {s.study_name:30s}  {n_total:>8d}  {n_pruned:>8d}  {best_str:>9s}")

    # Per-study details + best-params JSON
    print("\n" + "=" * 70)
    for s in summaries:
        st = optuna.load_study(study_name=s.study_name, storage=storage)
        completed = [t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            print(f"\n[{s.study_name}] no completed trials, skipping")
            continue

        best = st.best_trial
        print(f"\n[{s.study_name}]")
        print(f"  best AUC : {best.value:.4f}  (trial {best.number})")
        print(f"  params:")
        for k, v in best.params.items():
            print(f"    {k}: {v}")

        # Top 5 trials
        ranked = sorted(
            (t for t in completed),
            key=lambda t: t.value, reverse=True,
        )[:5]
        print(f"  top 5 trials:")
        for t in ranked:
            print(f"    #{t.number:>4d}  AUC={t.value:.4f}")

        # Try parameter importance (TPE only — cheap)
        try:
            importances = optuna.importance.get_param_importances(st)
            print(f"  param importances:")
            for k, v in sorted(importances.items(), key=lambda kv: -kv[1])[:6]:
                print(f"    {k:25s} {v:.3f}")
        except Exception as e:
            pass  # FANOVA needs >1 trial with varied params; ignore for tiny studies

        # Save best params JSON
        out_path = RESULTS_DIR / f"{s.study_name}_best.json"
        out = {
            "study_name": s.study_name,
            "best_trial_number": best.number,
            "best_value_auc": best.value,
            "best_params": best.params,
            "n_completed": len(completed),
            "n_pruned": sum(1 for t in st.trials if t.state == optuna.trial.TrialState.PRUNED),
        }
        with out_path.open("w") as f:
            json.dump(out, f, indent=2)
        print(f"  saved → {out_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
