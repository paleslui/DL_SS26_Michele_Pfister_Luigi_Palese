#!/usr/bin/env bash
# Convenience launcher — submits all 8 SLURM jobs:
#   2 models (cnn, transformer)
#   x 2 samplers (tpe, random)
#   x 2 parallel workers per study
#
# All workers writing to the same study coordinate via the SQLite database.
# Run once; if some jobs die, just rerun and they'll resume.
#
# Usage:  bash launch_searches.sh [N_TPE_TRIALS_PER_WORKER] [N_RANDOM_TRIALS_PER_WORKER]
# Defaults: 100 TPE, 25 random per worker (so 200 TPE / 50 random total per model)

set -euo pipefail

N_TPE_PER_WORKER=${1:-100}
N_RANDOM_PER_WORKER=${2:-25}

SLURM_SCRIPT="$(dirname "$0")/run_optuna.slurm"

echo "Submitting Optuna search jobs..."
echo "  TPE   : $N_TPE_PER_WORKER trials/worker × 2 workers = $((N_TPE_PER_WORKER * 2)) trials/study"
echo "  Random: $N_RANDOM_PER_WORKER trials/worker × 2 workers = $((N_RANDOM_PER_WORKER * 2)) trials/study"
echo ""

submit() {
    local model=$1
    local sampler=$2
    local n_trials=$3
    local worker_id=$4
    sbatch \
        --job-name="opt_${model}_${sampler}_w${worker_id}" \
        --export="MODEL=${model},SAMPLER=${sampler},N_TRIALS=${n_trials},WORKER_ID=${worker_id}" \
        "$SLURM_SCRIPT"
}

for model in cnn transformer; do
    for sampler in tpe random; do
        if [[ "$sampler" == "tpe" ]]; then
            n=$N_TPE_PER_WORKER
        else
            n=$N_RANDOM_PER_WORKER
        fi
        for worker in 0 1; do
            submit "$model" "$sampler" "$n" "$worker"
        done
    done
done

echo ""
echo "All jobs submitted. Watch progress:"
echo "  squeue -u \$USER"
echo "  tail -f /cfs/earth/scratch/paleslui/DL_SS26/logs/opt_*.out"
