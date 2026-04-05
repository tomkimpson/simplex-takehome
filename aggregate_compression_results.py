#!/usr/bin/env python3
"""Aggregate compression sweep results across seeds and generate figures."""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

from src.plots import (plot_money_plot, plot_probe_comparison_sweep,
                       plot_convergence_diagnostics, plot_layer_structure,
                       plot_effective_dimensionality)


def collect_results(results_dir: str) -> dict:
    """Walk compression_results/n_states_*/seed_*/ and collect all JSONs."""
    results_dir = Path(results_dir)
    runs = defaultdict(list)

    for n_dir in sorted(results_dir.glob('n_states_*')):
        n_states = int(n_dir.name.split('_')[-1])
        for seed_dir in sorted(n_dir.glob('seed_*')):
            seed = int(seed_dir.name.split('_')[-1])

            probe_path = seed_dir / 'probe_metrics.json'
            hist_path = seed_dir / 'training_history.json'

            if not probe_path.exists() or not hist_path.exists():
                print(f"  Skipping {n_dir.name}/{seed_dir.name}: missing files")
                continue

            with open(probe_path) as f:
                probe_metrics = json.load(f)
            with open(hist_path) as f:
                history = json.load(f)

            runs[n_states].append({
                'seed': seed,
                'n_states': n_states,
                'compression_ratio': n_states / 64,
                'probe_metrics': probe_metrics,
                'entropy_rate_nats': history.get('entropy_rate_nats', float('nan')),
                'convergence_gap': history.get('convergence_gap', float('nan')),
                'epochs_trained': history.get('epochs_trained', len(history.get('train_loss', []))),
                'train_loss': history.get('train_loss', []),
                'eval_loss': history.get('eval_loss', []),
            })

    return dict(runs)


def aggregate_across_seeds(runs: dict) -> dict:
    """Compute mean and stderr for each metric across seeds per n_states."""
    summary = {}

    for n_states in sorted(runs.keys()):
        seed_runs = runs[n_states]
        n_seeds = len(seed_runs)

        # Collect per-layer metrics across seeds
        layer_names = list(seed_runs[0]['probe_metrics'].keys())

        per_layer_agg = {}
        for layer in layer_names:
            # Raw geometry metrics (probe-free)
            euc_aligns = []
            kl_aligns = []
            gaps = []
            eff_ranks = []
            # Probe-based metrics
            mse_pairwise_euc = []
            mse_pairwise_kl = []
            kl_pairwise_euc = []
            kl_pairwise_kl = []

            for run in seed_runs:
                lm = run['probe_metrics'].get(layer, {})

                # Raw geometry (may not exist in old runs)
                rg = lm.get('raw_geometry', {})
                if rg:
                    euc_aligns.append(rg['euclidean_alignment'])
                    kl_aligns.append(rg['kl_alignment'])
                    gaps.append(rg['alignment_gap'])

                er = lm.get('effective_rank')
                if er is not None:
                    eff_ranks.append(er)

                # Probe metrics
                mm = lm.get('mse_metrics', {})
                km = lm.get('kl_metrics', {})
                if mm:
                    mse_pairwise_euc.append(mm.get('pairwise_r2_euclidean', float('nan')))
                    mse_pairwise_kl.append(mm.get('pairwise_rho_kl', float('nan')))
                if km:
                    kl_pairwise_euc.append(km.get('pairwise_r2_euclidean', float('nan')))
                    kl_pairwise_kl.append(km.get('pairwise_rho_kl', float('nan')))

            def _stats(vals):
                if not vals:
                    return {'mean': float('nan'), 'stderr': float('nan'), 'n': 0}
                arr = np.array(vals)
                return {
                    'mean': float(np.mean(arr)),
                    'stderr': float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
                    'n': len(arr),
                }

            per_layer_agg[layer] = {
                'raw_geometry': {
                    'euclidean_alignment': _stats(euc_aligns),
                    'kl_alignment': _stats(kl_aligns),
                    'alignment_gap': _stats(gaps),
                },
                'effective_rank': _stats(eff_ranks),
                'probe_metrics': {
                    'mse_pairwise_r2_euclidean': _stats(mse_pairwise_euc),
                    'mse_pairwise_rho_kl': _stats(mse_pairwise_kl),
                    'kl_pairwise_r2_euclidean': _stats(kl_pairwise_euc),
                    'kl_pairwise_rho_kl': _stats(kl_pairwise_kl),
                },
            }

        # Convergence info
        conv_gaps = [r['convergence_gap'] for r in seed_runs]

        summary[n_states] = {
            'n_states': n_states,
            'compression_ratio': n_states / 64,
            'n_seeds': n_seeds,
            'convergence_gap_mean': float(np.nanmean(conv_gaps)),
            'per_layer': per_layer_agg,
        }

    return summary


def print_convergence_report(runs: dict):
    """Print convergence diagnostics for all runs."""
    print("\n" + "=" * 70)
    print("CONVERGENCE REPORT")
    print("=" * 70)
    print(f"{'|S|':>5} {'seed':>6} {'epochs':>7} {'gap%':>7} {'status':>12}")
    print("-" * 70)

    any_warning = False
    for n_states in sorted(runs.keys()):
        for run in runs[n_states]:
            gap = run['convergence_gap']
            epochs = run['epochs_trained']
            status = "OK" if gap <= 0.10 else "WARNING"
            if status == "WARNING":
                any_warning = True
            print(f"  {n_states:>5} {run['seed']:>6} {epochs:>7} "
                  f"{gap:>6.1%} {status:>12}")

    if any_warning:
        print("\nSome models may be undertrained. Consider re-running with "
              "--resume --epochs 400")
    else:
        print("\nAll models within convergence threshold.")


def print_money_plot_table(summary: dict):
    """Print the key results table: compression ratio vs geometry alignment."""
    print("\n" + "=" * 80)
    print("GEOMETRY ALIGNMENT vs COMPRESSION (probe-free, best layer per n_states)")
    print("=" * 80)
    print(f"{'|S|':>5} {'ratio':>6} {'euc_align':>12} {'kl_align':>12} "
          f"{'gap':>10} {'eff_rank':>10}")
    print("-" * 80)

    for n_states in sorted(summary.keys()):
        s = summary[n_states]
        # Find layer with highest KL alignment (mean across seeds)
        best_layer = max(
            s['per_layer'],
            key=lambda l: s['per_layer'][l]['raw_geometry']['kl_alignment']['mean'])
        lg = s['per_layer'][best_layer]
        rg = lg['raw_geometry']
        er = lg['effective_rank']

        euc_m = rg['euclidean_alignment']['mean']
        euc_e = rg['euclidean_alignment']['stderr']
        kl_m = rg['kl_alignment']['mean']
        kl_e = rg['kl_alignment']['stderr']
        gap_m = rg['alignment_gap']['mean']

        print(f"  {n_states:>5} {s['compression_ratio']:>6.2f} "
              f"{euc_m:>7.3f}±{euc_e:.3f} "
              f"{kl_m:>7.3f}±{kl_e:.3f} "
              f"{gap_m:>+9.3f} "
              f"{er['mean']:>7.1f}")


def generate_figures(summary: dict, runs: dict, fig_dir: str):
    """Generate all figures from aggregated results."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating figures...")

    plot_money_plot(summary, save_path=str(fig_dir / 'money_plot.png'))
    print("  money_plot.png")

    plot_probe_comparison_sweep(summary, save_path=str(fig_dir / 'probe_comparison.png'))
    print("  probe_comparison.png")

    plot_convergence_diagnostics(runs, save_path=str(fig_dir / 'convergence.png'))
    print("  convergence.png")

    plot_layer_structure(summary, save_path=str(fig_dir / 'layer_structure.png'))
    print("  layer_structure.png")

    plot_effective_dimensionality(summary,
                                  save_path=str(fig_dir / 'effective_rank.png'))
    print("  effective_rank.png")


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate compression sweep results')
    parser.add_argument('--results-dir', type=str,
                        default='compression_results',
                        help='Root directory of compression results')
    parser.add_argument('--figures-dir', type=str,
                        default='compression_results/figures',
                        help='Output directory for figures')
    args = parser.parse_args()

    print("Collecting results...")
    runs = collect_results(args.results_dir)

    if not runs:
        print("No results found!")
        return

    print(f"Found results for {len(runs)} n_states values: "
          f"{sorted(runs.keys())}")

    print_convergence_report(runs)

    print("\nAggregating across seeds...")
    summary = aggregate_across_seeds(runs)

    print_money_plot_table(summary)

    # Save summary
    out_path = Path(args.results_dir) / 'sweep_summary.json'
    # Convert for JSON (numpy types)
    with open(str(out_path), 'w') as f:
        json.dump(summary, f, indent=2, default=lambda x: float(x)
                  if isinstance(x, (np.floating, np.integer)) else x)
    print(f"\nSaved aggregated summary to {out_path}")

    generate_figures(summary, runs, args.figures_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
