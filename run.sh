#! /bin/bash

#SBATCH --job-name=test
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:02:00
#SBATCH --mem=16G
#SBATCH --account=tolgalab
#SBATCH --partition=spartacus,beasts,gods
#SBATCH --output=logs/log_%J.txt
#SBATCH --signal=TERM@30
#SBATCH --requeue

source ~/.micromamba/etc/profile.d/mamba.sh
micromamba activate edo

echo $SLURM_JOB_NAME

srun python main.py --name requeuetest --no-tqdm -E 25 # || scontrol requeue $SLURM_JOB_ID
