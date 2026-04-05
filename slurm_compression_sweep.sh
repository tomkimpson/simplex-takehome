#!/bin/bash
#SBATCH --job-name=compression-sweep
#SBATCH --array=0-26
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=compression_results/slurm_%A_%a.out

# Experiment 3: Genuine Compression Regime
# 9 n_states values x 3 seeds = 27 jobs
#
# Usage:
#   sbatch slurm_compression_sweep.sh
#
# After all jobs complete, aggregate results:
#   python aggregate_compression_results.py

N_STATES_VALUES=(5 10 20 35 50 64 100 150 200)
SEEDS=(42 1042 2042)

# Decode 2D index from array task ID
N_IDX=$((SLURM_ARRAY_TASK_ID / 3))
S_IDX=$((SLURM_ARRAY_TASK_ID % 3))
N_STATES=${N_STATES_VALUES[$N_IDX]}
SEED=${SEEDS[$S_IDX]}

echo "Running compression experiment: |S| = ${N_STATES}, seed = ${SEED}"
echo "Device: cuda"
echo "Job ID: ${SLURM_JOB_ID}, Array Task: ${SLURM_ARRAY_TASK_ID}"

# Activate environment (adjust path as needed)
source .venv/bin/activate

python run_compression_experiment.py \
    --n-states ${N_STATES} \
    --seed ${SEED} \
    --device cuda \
    --epochs 200 \
    --kl-epochs 1000

echo "Done: |S| = ${N_STATES}, seed = ${SEED}"
