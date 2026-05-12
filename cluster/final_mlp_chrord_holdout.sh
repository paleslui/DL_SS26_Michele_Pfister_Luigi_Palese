#!/bin/bash
#SBATCH --job-name=final_mlp_chrord
#SBATCH --mail-user=paleslui@students.zhaw.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=earth-5
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/cfs/earth/scratch/paleslui/DL_SS26/logs/final_mlp_chrord_%j.out
#SBATCH --error=/cfs/earth/scratch/paleslui/DL_SS26/logs/final_mlp_chrord_%j.err

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

cd /cfs/earth/scratch/paleslui/DL_SS26/DL_SS26_Michele_Pfister_Luigi_Palese
$PY scripts/09_final_mlp_chrord_holdout.py

echo "=== Job ended at $(date) ==="
