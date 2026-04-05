# HPC Compression Sweep — Handoff Instructions

## What this is

A compression sweep testing whether transformers preferentially preserve information-geometric (KL) structure over Euclidean structure when forced to compress high-dimensional belief states. We train transformers on random sparse HMMs with varying numbers of hidden states (5 to 200) while keeping d_model=64 fixed.

## Branch

`information-geometry` — make sure you're on this branch.

## How to run the sweep

```bash
# Pull latest code
git pull origin information-geometry

# Activate environment
source .venv/bin/activate

# Submit the Slurm array job (27 jobs: 9 n_states x 3 seeds)
sbatch slurm_compression_sweep.sh

# Monitor progress
squeue -u $USER
```

Each job runs `run_compression_experiment.py` for one (n_states, seed) combination. Results go to `compression_results/n_states_{N}/seed_{S}/`.

## Expected runtime

- Small n_states (5-35): ~30-60 min per job
- Medium (50-64): ~1-2 hours
- Large (100-200): ~2-4 hours

Total wall-clock: ~4-6 hours with all jobs running in parallel.

## After the sweep completes

```bash
python aggregate_compression_results.py
```

This generates:
- `compression_results/sweep_summary.json` — aggregated metrics
- `compression_results/figures/money_plot.png` — **the key figure**
- `compression_results/figures/probe_comparison.png`
- `compression_results/figures/convergence.png`
- `compression_results/figures/layer_structure.png`
- `compression_results/figures/effective_rank.png`

## What to check

1. **Convergence report** — printed by the aggregation script. Any model with convergence gap > 10% may be undertrained. Fix with:
   ```bash
   python run_compression_experiment.py --n-states N --seed S --device cuda --epochs 400 --resume
   ```

2. **Money plot** — the central figure. Look for:
   - KL alignment (orange) > Euclidean alignment (blue) at all compression ratios
   - The gap widening as compression ratio increases past 1.0
   - If the gap is flat or reverses, the central hypothesis is wrong (still a valid finding)

3. **Effective rank** — should increase with n_states as the model uses more of its 64D capacity

## Key metrics

- `euclidean_alignment`: Spearman(activation distances, belief Euclidean distances) — probe-free
- `kl_alignment`: Spearman(activation distances, belief KL divergences) — probe-free
- `alignment_gap`: kl_alignment - euclidean_alignment (positive = info-geometric preference)

## File structure

```
run_compression_experiment.py   # Main experiment: train + probe + geometry
aggregate_compression_results.py # Collect results, compute stats, plot
slurm_compression_sweep.sh      # Slurm array job (27 jobs)
src/config.py                   # CompressionExperimentConfig
src/analysis.py                 # compute_raw_geometry_metrics(), compute_effective_rank()
src/plots.py                    # plot_money_plot() and 4 other sweep plots
```

## If something goes wrong

- **Job OOM**: Reduce `--n-analysis` (default 5000) or `--n-train` (default 300k)
- **KL probe doesn't converge for large n_states**: Try `--kl-lr 5e-4 --kl-epochs 2000`
- **Model doesn't learn (loss near log(3)=1.099)**: May need more epochs or training data. Use `--resume --epochs 400` or `--n-train 500000`
