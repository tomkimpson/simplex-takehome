#!/usr/bin/env python3
"""Main orchestration script: train model and run analysis."""

import argparse
import sys
sys.path.insert(0, '.')

from src.config import ExperimentConfig
from src.train import train


def run_analysis(checkpoint_path: str, device: str = 'cpu', n_analysis: int = 5000):
    """Run the full analysis pipeline and generate all figures."""
    from src.analysis import run_full_analysis
    from src.plots import (plot_training_curves, plot_pca_by_layer,
                           plot_pca_by_position, plot_r2_progression,
                           plot_component_identification,
                           plot_recovered_simplices,
                           plot_subspace_orthogonality,
                           plot_attention_patterns,
                           plot_explained_variance)
    from src.belief import compute_mixture_beliefs
    import numpy as np

    fig_dir = 'writeup/figures'

    # Training curves
    print("\n=== Training Curves ===")
    plot_training_curves('results/logs/training_history.json',
                         save_path=f'{fig_dir}/training_curves.png')

    # Full analysis
    print("\n=== Running Analysis ===")
    results = run_full_analysis(checkpoint_path, device=device,
                                n_analysis=n_analysis)

    # PCA by layer
    print("\n=== PCA by Layer ===")
    plot_pca_by_layer(results['pca'],
                      save_path=f'{fig_dir}/pca_by_layer.png')

    # PCA by position
    print("\n=== PCA by Position ===")
    plot_pca_by_position(results['pca'], layer_key='final',
                         save_path=f'{fig_dir}/pca_by_position.png')

    # Explained variance
    print("\n=== Explained Variance ===")
    plot_explained_variance(results['pca'],
                            save_path=f'{fig_dir}/explained_variance.png')

    # R² progression
    print("\n=== R² Progression ===")
    plot_r2_progression(results['regression'],
                        save_path=f'{fig_dir}/r2_progression.png')

    # Component identification
    print("\n=== Component Identification ===")
    # Bayesian optimal baseline
    data = results['data']
    processes = results['processes']
    _, comp_posteriors = compute_mixture_beliefs(data['tokens'][:1000], processes)
    # Bayesian accuracy = fraction correctly identified per position
    true_labels = data['component_labels'][:1000]
    bayesian_acc = []
    for pos in range(comp_posteriors.shape[1] - 1):  # -1 because model sees L-1 positions
        predicted = comp_posteriors[:, pos + 1, :].argmax(axis=1)
        bayesian_acc.append((predicted == true_labels).mean())
    bayesian_acc = np.array(bayesian_acc)

    plot_component_identification(results['probes'],
                                  bayesian_accuracy=bayesian_acc,
                                  save_path=f'{fig_dir}/component_identification.png')

    # Recovered simplices — use layer_2 (pre-LayerNorm) rather than final,
    # because the final LayerNorm compresses to ~2D for the output head,
    # destroying component-specific fractal structure.
    print("\n=== Recovered Simplices ===")
    plot_recovered_simplices(results['simplices'], layer_key='layer_2',
                             save_path=f'{fig_dir}/recovered_simplices.png')

    # Subspace orthogonality
    print("\n=== Subspace Orthogonality ===")
    for key in ['final', 'layer_2', 'layer_1', 'layer_0']:
        if key in results['orthogonality']:
            plot_subspace_orthogonality(results['orthogonality'], layer_key=key,
                                        save_path=f'{fig_dir}/subspace_orthogonality_{key}.png')

    # Attention patterns
    print("\n=== Attention Patterns ===")
    plot_attention_patterns(results['attention']['weights'],
                            results['attention']['labels'],
                            save_path=f'{fig_dir}/attention_patterns.png')

    # Print summary statistics
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    for layer_name, layer_res in results['regression']['per_layer'].items():
        r2 = layer_res['r2_overall']
        r2_comp = layer_res['r2_per_component']
        comp_str = ', '.join(f'{n}={r:.3f}' for n, r in zip(['A', 'B', 'C'], r2_comp))
        print(f"  {layer_name:12s}  R²={r2:.3f}  [{comp_str}]")

    print("\nComponent ID accuracy (final layer, last position):")
    for layer_name, probe_res in results['probes'].items():
        acc = probe_res['per_position_accuracy'][-1]
        print(f"  {layer_name:12s}  accuracy={acc:.3f}")

    print("\nSubspace overlap (final layer):")
    if 'final' in results['orthogonality']:
        overlap = results['orthogonality']['final']['subspace_overlap']
        K = overlap.shape[0]
        for i in range(K):
            for j in range(i + 1, K):
                print(f"  {['A', 'B', 'C'][i]}-{['A', 'B', 'C'][j]}: {overlap[i, j]:.3f}")

    print("\nAll figures saved to writeup/figures/")
    return results


def run_kl_analysis(checkpoint_path: str, device: str = 'cpu',
                    n_analysis: int = 5000, experiment: int = None,
                    kl_lr: float = 1e-3, kl_epochs: int = 1000):
    """Run the information-geometric probing analysis.

    Args:
        experiment: which experiment to run (1=head-to-head, 2=PCA sweep,
                   4=layer-by-layer). None runs 1 and 4.
    """
    from src.analysis import (run_full_analysis, head_to_head_comparison,
                              kl_belief_regression, pca_compressed_probing,
                              recovered_simplices, recovered_simplices_kl)
    from src.plots import (plot_probe_comparison, plot_pca_compression_sweep,
                           plot_recovered_simplices_comparison,
                           plot_kl_layer_profile)

    fig_dir = 'writeup/figures'

    # Reuse the full MSE analysis
    print("\n=== Loading baseline analysis ===")
    results = run_full_analysis(checkpoint_path, device=device,
                                n_analysis=n_analysis)

    data = results['data']
    processes = results['processes']
    experiments = [experiment] if experiment is not None else [1, 4]

    if 1 in experiments:
        print("\n" + "=" * 60)
        print("EXPERIMENT 1: Head-to-Head Comparison (MSE vs KL)")
        print("=" * 60)

        comparison = head_to_head_comparison(
            data['activations'], data['beliefs'],
            data['component_labels'], processes,
            mse_results=results['regression'],
            kl_lr=kl_lr, kl_epochs=kl_epochs, device=device)

        plot_probe_comparison(comparison,
                              save_path=f'{fig_dir}/probe_comparison.png')

        # Recovered simplices comparison
        print("\n=== KL Recovered Simplices ===")
        kl_simplices = recovered_simplices_kl(
            data['activations'], data['beliefs'],
            data['component_labels'],
            kl_lr=kl_lr, kl_epochs=kl_epochs, device=device)

        plot_recovered_simplices_comparison(
            results['simplices'], kl_simplices, layer_key='layer_2',
            save_path=f'{fig_dir}/simplices_comparison.png')

        # Print comparison summary
        print("\n" + "-" * 60)
        print("EXPERIMENT 1 SUMMARY")
        print("-" * 60)
        print(f"{'Layer':<12} {'Metric':<22} {'MSE Probe':>10} {'KL Probe':>10}")
        print("-" * 60)
        for layer_name in comparison['mse']:
            if layer_name not in comparison['kl']:
                continue
            mse_m = comparison['mse'][layer_name]['metrics']
            kl_m = comparison['kl'][layer_name]
            for metric in ['mse', 'kl_divergence', 'boundary_kl',
                           'simplex_violation_rate', 'pairwise_r2_euclidean']:
                print(f"  {layer_name:<12} {metric:<22} {mse_m[metric]:>10.6f} {kl_m[metric]:>10.6f}")
            print()

    if 2 in experiments:
        print("\n" + "=" * 60)
        print("EXPERIMENT 2: PCA Compression Sweep")
        print("=" * 60)

        sweep_results = pca_compressed_probing(
            data['activations'], data['beliefs'],
            data['component_labels'], processes,
            layer_key='layer_2',
            kl_lr=kl_lr, kl_epochs=kl_epochs, device=device)

        plot_pca_compression_sweep(sweep_results,
                                   save_path=f'{fig_dir}/pca_sweep.png')

        # Print sweep summary
        print("\n" + "-" * 60)
        print("EXPERIMENT 2 SUMMARY")
        print("-" * 60)
        print(f"{'k':>4} {'MSE(MSE)':>10} {'MSE(KL)':>10} {'KL(MSE)':>10} {'KL(KL)':>10} {'BndKL(MSE)':>12} {'BndKL(KL)':>12}")
        for k in sorted(sweep_results.keys()):
            mm = sweep_results[k]['mse_metrics']
            km = sweep_results[k]['kl_metrics']
            print(f"  {k:>4} {mm['mse']:>10.6f} {km['mse']:>10.6f} "
                  f"{mm['kl_divergence']:>10.6f} {km['kl_divergence']:>10.6f} "
                  f"{mm['boundary_kl']:>12.6f} {km['boundary_kl']:>12.6f}")

    if 4 in experiments:
        print("\n" + "=" * 60)
        print("EXPERIMENT 4: Layer-by-Layer Profile")
        print("=" * 60)

        kl_results = kl_belief_regression(
            data['activations'], data['beliefs'],
            data['component_labels'], processes,
            lr=kl_lr, n_epochs=kl_epochs, device=device)

        plot_kl_layer_profile(results['regression'], kl_results,
                              save_path=f'{fig_dir}/kl_layer_profile.png')

        # Print layer profile summary
        print("\n" + "-" * 60)
        print("EXPERIMENT 4 SUMMARY")
        print("-" * 60)
        print(f"{'Layer':<12} {'MSE R²':>8} {'KL div':>10} {'Bnd KL':>10} {'Pairwise R²':>12}")
        for layer_name in kl_results['per_layer']:
            mse_r2 = results['regression']['per_layer'].get(layer_name, {}).get('r2_overall', float('nan'))
            kl_m = kl_results['per_layer'][layer_name]['metrics']
            print(f"  {layer_name:<12} {mse_r2:>8.4f} {kl_m['kl_divergence']:>10.6f} "
                  f"{kl_m['boundary_kl']:>10.6f} {kl_m['pairwise_r2_euclidean']:>12.4f}")

    print("\nKL probe figures saved to writeup/figures/")


def main():
    parser = argparse.ArgumentParser(description='Non-Ergodic Mess3 Experiment')
    parser.add_argument('--device', type=str, default=None,
                        help='Device: cpu, mps, cuda')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--train-only', action='store_true',
                        help='Only run training, skip analysis')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only run analysis (requires trained model)')
    parser.add_argument('--checkpoint', type=str,
                        default='results/checkpoints/model_final.pt',
                        help='Path to model checkpoint for analysis')
    parser.add_argument('--n-analysis', type=int, default=5000,
                        help='Number of sequences for analysis')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint path')
    # KL probe arguments
    parser.add_argument('--kl-probe', action='store_true',
                        help='Run KL (softmax-affine) probe analysis')
    parser.add_argument('--experiment', type=int, choices=[1, 2, 4], default=None,
                        help='Run specific experiment (1=head-to-head, 2=PCA sweep, '
                             '4=layer-by-layer)')
    parser.add_argument('--kl-lr', type=float, default=1e-3,
                        help='Learning rate for KL probe')
    parser.add_argument('--kl-epochs', type=int, default=1000,
                        help='Number of training epochs for KL probe')
    args = parser.parse_args()

    config = ExperimentConfig()
    if args.epochs is not None:
        config.n_epochs = args.epochs

    device_str = args.device or 'cpu'

    if not args.analyze_only:
        model, history = train(config, device_str=args.device,
                               resume_from=args.resume)

    if not args.train_only:
        run_analysis(args.checkpoint, device=device_str,
                     n_analysis=args.n_analysis)
        if args.kl_probe or args.experiment is not None:
            run_kl_analysis(args.checkpoint, device=device_str,
                            n_analysis=args.n_analysis,
                            experiment=args.experiment,
                            kl_lr=args.kl_lr, kl_epochs=args.kl_epochs)


if __name__ == '__main__':
    main()
