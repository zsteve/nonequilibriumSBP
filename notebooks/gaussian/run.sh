#!/bin/bash
#SBATCH --job-name=gaussian
#SBATCH --output=output_%A_%a.out
#SBATCH --error=error_%A_%a.out
#SBATCH --array=1-30
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=fos
#SBATCH --qos=fos
#SBATCH --time=01:00:00

dims=(2 5 10 25 50 100)
seeds=(1 2 3 4 5)
n_seeds=${#seeds[@]}

idx_dim=$(((SLURM_ARRAY_TASK_ID-1) / n_seeds + 1))
idx_seed=$(((SLURM_ARRAY_TASK_ID-1) % n_seeds + 1))

dim=${dims[$idx_dim-1]}
seed=${seeds[$idx_seed-1]}

source ~/.bashrc
conda activate lfm
N=128

python gaussian.py --seed $seed --N $N --d $dim --train_otfm --sigma 1.0 --suffix "seed_"$seed"_dim_"$dim"_N_"$N
