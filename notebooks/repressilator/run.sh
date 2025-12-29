#!/bin/bash
#SBATCH --job-name=repress
#SBATCH --output=output_%A_%a.out
#SBATCH --error=error_%A_%a.out
#SBATCH --array=1-45
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=fos-gpu-l40s
#SBATCH --qos=fos
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

leaveouts=(-1 1 2 3 4 5 6 7 8)
seeds=(1 2 3 4 5)
n_seeds=${#seeds[@]}

idx_leaveout=$(((SLURM_ARRAY_TASK_ID-1) / n_seeds + 1))
idx_seed=$(((SLURM_ARRAY_TASK_ID-1) % n_seeds + 1))

leaveout=${leaveouts[$idx_leaveout-1]}
seed=${seeds[$idx_seed-1]}
sbirr=OU

source ~/.bashrc
conda activate lfm

suffix="sbirr_"$sbirr"_leaveout_"$leaveout"_seed_"$seed
echo $suffix
python repressilator.py --seed $seed --suffix $suffix --holdout $leaveout --train_otfm --make_plots --make_evals --sbirr $sbirr

if [ "$leaveout" == "-1" ]; then
    suffix="sbirr_"$sbirr"_leaveout_"$leaveout"_seed_"$seed"_gtref"
    echo $suffix
    python repressilator.py --seed $seed --suffix $suffix --holdout $leaveout --train_otfm --make_evals --gtref --sbirr $sbirr
fi
