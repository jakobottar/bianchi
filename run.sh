#! /bin/bash

#SBATCH --job-name=test
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --mem=16G
#SBATCH --account=tolgalab
#SBATCH --partition=spartacus,beasts,gods
#SBATCH --output=logs/log_%J.txt
#SBATCH --signal=TERM@60

source ~/.micromamba/etc/profile.d/mamba.sh
micromamba activate edo

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

srun python main.py -S $SLURM_JOB_ID --name requeuetest --no-tqdm -E 50
EXIT_CODE=$?

echo "=========================================="
echo "Python program exited with code: $EXIT_CODE"
echo "Job finished at: $(date)"

# TODO: signal ntfy on success/failure/requeue

if [ $EXIT_CODE -eq 0 ]; then
    echo "Work completed successfully!"
    echo "=========================================="
    exit 0
elif [ $EXIT_CODE -eq 99 ]; then
    echo "Program indicates it needs to continue (exit code 99)"
    scontrol requeue $SLURM_JOB_ID
    echo "Requeue command issued."
else
    echo "Not requeuing due to unexpected failure."
fi

echo "=========================================="
