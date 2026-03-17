#!/bin/bash
#SBATCH --job-name=compression-sweep
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=compression_results/slurm_%A_%a.out

# Experiment 3: Genuine Compression Regime
# Submits one job per |S| value via Slurm array tasks.
#
# Usage:
#   sbatch slurm_compression_sweep.sh
#
# After all jobs complete, aggregate results:
#   python run_compression_experiment.py --aggregate

N_STATES_VALUES=(10 20 50 100 200)
N_STATES=${N_STATES_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "Running compression experiment: |S| = ${N_STATES}"
echo "Device: cuda"
echo "Job ID: ${SLURM_JOB_ID}, Array Task: ${SLURM_ARRAY_TASK_ID}"

# Activate environment (adjust path as needed)
source .venv/bin/activate

python run_compression_experiment.py \
    --n-states ${N_STATES} \
    --device cuda \
    --epochs 200 \
    --kl-epochs 1000

echo "Done: |S| = ${N_STATES}"
