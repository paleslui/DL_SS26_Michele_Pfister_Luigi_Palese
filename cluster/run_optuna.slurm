#!/usr/bin/env bash
#SBATCH --job-name=opt_${MODEL}_${SAMPLER}
#SBATCH --output=/cfs/earth/scratch/paleslui/DL_SS26/logs/opt_%x_%j.out
#SBATCH --error=/cfs/earth/scratch/paleslui/DL_SS26/logs/opt_%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=earth-5
#SBATCH --constraint=rhel8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --mail-user=paleslui@students.zhaw.ch
#SBATCH --mail-type=END,FAIL

# ---- Required environment variables (set by sbatch --export) ---------------
#  MODEL    : cnn | transformer
#  SAMPLER  : tpe | random
#  N_TRIALS : integer
#  WORKER_ID: integer (different seeds for parallel workers on same study)
# ---------------------------------------------------------------------------

set -euo pipefail

REPO=/cfs/earth/scratch/paleslui/DL_SS26/DL_SS26_Michele_Pfister_Luigi_Palese
mkdir -p /cfs/earth/scratch/paleslui/DL_SS26/logs

# Activate dl_msi via micromamba
export MAMBA_ROOT_PREFIX=/cfs/earth/scratch/paleslui/miniforge3
eval "$(/cfs/earth/scratch/paleslui/miniforge3/micromamba shell hook -s bash)"
micromamba activate dl_msi

# Sanity print
echo "=== Job info ==="
echo "Hostname    : $(hostname)"
echo "Started at  : $(date)"
echo "MODEL       : ${MODEL}"
echo "SAMPLER     : ${SAMPLER}"
echo "N_TRIALS    : ${N_TRIALS}"
echo "WORKER_ID   : ${WORKER_ID:-0}"
echo "GPU         : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo "==============="

cd "$REPO"

python cluster/optuna_search.py \
    --model "${MODEL}" \
    --sampler "${SAMPLER}" \
    --n-trials "${N_TRIALS}"

echo "=== Job ended at $(date) ==="
