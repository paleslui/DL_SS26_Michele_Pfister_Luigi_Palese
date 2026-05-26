# `cluster/` — hyperparameter search on the ZHAW HPC cluster

The heavy Optuna searches (CNN, transformer, LSTM, MLP-chrord) ran on the ZHAW
**Earth** SLURM cluster on an NVIDIA A100 (40 GB). This folder holds the search
worker, the SLURM job scripts, and the environment setup. Everything writes to
a shared SQLite database so jobs can be killed and resumed without losing
progress.

## Files

| File | Purpose |
|---|---|
| `optuna_search.py` | The search worker. Defines the search space + objective for each model and runs the Optuna study. One generic script, selected via `--model`. |
| `search_<model>_<sampler>.sh` | SLURM job wrappers — one per (model × sampler). E.g. `search_cnn_tpe.sh` submits a 1000-trial TPE search for the CNN. `_random` variants are the random-search baselines. |
| `final_mlp_chrord_holdout.sh` | SLURM wrapper to run the final MLP-chrord holdout evaluation on the cluster. |
| `analyze_results.py` | Reads the SQLite DB after searches finish and writes each study's best hyperparameters to `results/<study>_best.json`. No GPU needed — runs on the login node. |
| `install_dl_msi_env.sh` | One-off environment setup (creates the conda env from `dl_msi_env.yml` via micromamba). |
| `dl_msi_env.yml` | Conda environment definition for the cluster. |

## How a search was run

```bash
# 1. one-off: create the environment
sbatch cluster/install_dl_msi_env.sh

# 2. submit searches (each writes to results/optuna_studies.db)
sbatch cluster/search_cnn_tpe.sh
sbatch cluster/search_cnn_random.sh
# ... etc for transformer / lstm / mlp_chrord

# 3. after they finish: extract best hyperparameters to results/*_best.json
python cluster/analyze_results.py
```

The samplers: **TPE** (the real optimiser) and **Random** (a control baseline,
to show the tuned result reflects informed search, not luck).

## Note on paths

The `.sh` scripts and `install_dl_msi_env.sh` contain **hardcoded cluster paths**
(`/cfs/earth/scratch/paleslui/...`) and a ZHAW account. To reproduce on another
account, adjust those paths and the `--mail-user` line. The Python
(`optuna_search.py`, `analyze_results.py`) is path-agnostic and runs anywhere
the repo + environment exist.
