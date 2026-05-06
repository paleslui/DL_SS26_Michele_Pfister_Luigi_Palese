#!/usr/bin/env bash
#SBATCH --job-name=dl_msi_install
#SBATCH --output=/cfs/earth/scratch/paleslui/DL_SS26/install_dl_msi_%j.log
#SBATCH --error=/cfs/earth/scratch/paleslui/DL_SS26/install_dl_msi_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --partition=earth-5
#SBATCH --constraint=rhel8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --mail-user=paleslui@students.zhaw.ch
#SBATCH --mail-type=ALL

# Use the standalone micromamba binary directly — bypasses the slow spack conda
MICROMAMBA=/cfs/earth/scratch/paleslui/miniforge3/micromamba
export MAMBA_ROOT_PREFIX=/cfs/earth/scratch/paleslui/miniforge3

# Remove old env if it exists (uncomment when rebuilding)
# $MICROMAMBA env remove -n dl_msi -y

# Create the env from the YAML — micromamba solves in seconds
$MICROMAMBA env create -y -f /cfs/earth/scratch/paleslui/DL_SS26/DL_SS26_Michele_Pfister_Luigi_Palese/cluster/dl_msi_env.yml

echo "dl_msi environment setup completed!"