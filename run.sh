#! /bin/bash

#SBATCH --job-name=test
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=8
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --account=tolgalab
#SBATCH --partition=spartacus,beasts,gods
#SBATCH --output=logs/log_%J.txt

source ~/.micromamba/etc/profile.d/mamba.sh
micromamba activate edo

echo $SLURM_JOB_NAME

python main.py --no-tqdm -E 25
