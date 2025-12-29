#!/bin/bash
#SBATCH --job-name=gaus_nlsb
#SBATCH --output=output_%A_%a.out
#SBATCH --error=error_%A_%a.out
#SBATCH --array=1-30
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=fos-gpu-l40s
#SBATCH --qos=fos
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00

dims=(2 5 10 25 50 100)
seeds=(1 2 3 4 5)
n_seeds=${#seeds[@]}

idx_dim=$(((SLURM_ARRAY_TASK_ID-1) / n_seeds + 1))
idx_seed=$(((SLURM_ARRAY_TASK_ID-1) % n_seeds + 1))

dim=${dims[$idx_dim-1]}
seed=${seeds[$idx_seed-1]}

source ~/.bashrc
conda activate nlsb
N=128

time python gaussian_nlsb.py --seed $seed --epochs 2500 --suffix "seed_"$seed"_dim_"$dim"_N_"$N
