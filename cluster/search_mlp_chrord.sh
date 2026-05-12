#!/bin/bash
#SBATCH --job-name=opt_mlp_chrord
#SBATCH --mail-user=paleslui@students.zhaw.ch
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=earth-5
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/cfs/earth/scratch/paleslui/DL_SS26/logs/opt_mlp_chrord_%j.out
#SBATCH --error=/cfs/earth/scratch/paleslui/DL_SS26/logs/opt_mlp_chrord_%j.err

module purge
module load DefaultModules
module load gcc/9.4.0-pe5.34

PY=/cfs/earth/scratch/paleslui/.conda/envs/dl_msi/bin/python

echo "=== Job info ==="
echo "Hostname    : $(hostname)"
echo "Started at  : $(date)"
$PY --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "================"

$PY -c "import torch; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

cd /cfs/earth/scratch/paleslui/DL_SS26/DL_SS26_Michele_Pfister_Luigi_Palese

# TPE first
$PY cluster/optuna_search.py \
    --model mlp_chrord \
    --sampler tpe \
    --n-trials 200 \
    --study-name mlp_chrord_tpe_v2

# Then random baseline
$PY cluster/optuna_search.py \
    --model mlp_chrord \
    --sampler random \
    --n-trials 75 \
    --study-name mlp_chrord_random_v2

echo "=== Job ended at $(date) ==="
