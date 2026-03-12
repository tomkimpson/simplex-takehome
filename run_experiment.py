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


if __name__ == '__main__':
    main()
