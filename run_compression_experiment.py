#!/usr/bin/env python3
"""Experiment 3: Genuine Compression Regime.

Trains transformers on random sparse HMMs with |S| >> d_resid and compares
MSE vs KL probes for belief state recovery under genuine compression.
"""

import argparse
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.linear_model import LinearRegression

sys.path.insert(0, '.')

from src.config import CompressionExperimentConfig
from src.hmm import random_sparse_hmm
from src.dataset import GeneralHMMDataset
from src.transformer import Mess3Transformer
from src.belief import compute_beliefs_general
from src.kl_probe import train_kl_probe, predict_beliefs_kl
from src.analysis import (compute_probe_metrics, compute_raw_geometry_metrics,
                          compute_effective_rank)


def train_on_hmm(model, train_loader, eval_loader, config, device='cpu',
                 start_epoch=0):
    """Train transformer on general HMM data."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history = {'train_loss': [], 'eval_loss': []}
    total_epochs = start_epoch + config.n_epochs

    for epoch in range(start_epoch, total_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            inputs = batch['input_tokens'].to(device)
            targets = batch['target_tokens'].to(device)

            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, config.vocab_size), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / n_batches
        history['train_loss'].append(avg_train)

        if (epoch + 1) % config.eval_every == 0 or epoch == total_epochs - 1:
            model.eval()
            eval_loss = 0.0
            eval_batches = 0
            with torch.no_grad():
                for batch in eval_loader:
                    inputs = batch['input_tokens'].to(device)
                    targets = batch['target_tokens'].to(device)
                    logits = model(inputs)
                    loss = F.cross_entropy(
                        logits.reshape(-1, config.vocab_size), targets.reshape(-1))
                    eval_loss += loss.item()
                    eval_batches += 1

            avg_eval = eval_loss / eval_batches
            history['eval_loss'].append((epoch, avg_eval))
            print(f"  Epoch {epoch+1:3d}/{total_epochs}: "
                  f"train={avg_train:.4f}  eval={avg_eval:.4f}")

    return history


def extract_and_probe(model, dataset, hmm, config, device='cpu'):
    """Extract activations, compute beliefs, run both probes."""
    n_analysis = min(config.n_analysis, len(dataset))
    input_tokens = dataset.tokens[:n_analysis, :-1]
    full_tokens = dataset.tokens[:n_analysis].numpy()

    # Compute ground-truth beliefs
    print("  Computing ground-truth beliefs...")
    beliefs = compute_beliefs_general(full_tokens, hmm)

    # Extract activations
    print("  Extracting activations...")
    model.eval()
    all_activations = {}
    batch_size = 256

    for start in range(0, n_analysis, batch_size):
        end = min(start + batch_size, n_analysis)
        batch_input = input_tokens[start:end].to(device)

        with torch.no_grad():
            _, acts = model(batch_input, return_activations=True)

        for key, val in acts.items():
            if 'attn_weights' in key:
                continue
            val_np = val.cpu().numpy()
            if key not in all_activations:
                all_activations[key] = []
            all_activations[key].append(val_np)

    activations = {k: np.concatenate(v, axis=0) for k, v in all_activations.items()}

    # Run probes and geometry analysis on each layer
    print("  Running probes and geometry analysis...")
    results_per_layer = {}
    n_states = hmm.n_states

    for layer_name, acts in activations.items():
        N, L, D = acts.shape
        target_beliefs = beliefs[:, 1:L+1, :]

        X = acts.reshape(-1, D)
        Y = target_beliefs.reshape(-1, n_states)

        # MSE probe
        reg = LinearRegression()
        reg.fit(X, Y)
        Y_pred_mse = reg.predict(X)
        Y_pred_mse_norm = np.clip(Y_pred_mse, 0, None)
        row_sums = Y_pred_mse_norm.sum(axis=1, keepdims=True)
        Y_pred_mse_norm = Y_pred_mse_norm / np.where(row_sums > 0, row_sums, 1)

        mse_metrics = compute_probe_metrics(Y, Y_pred_mse_norm)

        # KL probe
        probe, kl_history = train_kl_probe(
            X, Y, n_states=n_states,
            lr=config.kl_probe_lr, n_epochs=config.kl_probe_epochs,
            device=device)
        Y_pred_kl = predict_beliefs_kl(probe, X, device=device)

        kl_metrics = compute_probe_metrics(Y, Y_pred_kl)

        # Probe-free geometry metrics (the primary evidence)
        raw_geometry = compute_raw_geometry_metrics(acts, target_beliefs)

        # Effective dimensionality
        eff_rank = compute_effective_rank(acts)

        results_per_layer[layer_name] = {
            'mse_metrics': mse_metrics,
            'kl_metrics': kl_metrics,
            'raw_geometry': raw_geometry,
            'effective_rank': eff_rank,
        }

        print(f"    {layer_name}: "
              f"euc_align={raw_geometry['euclidean_alignment']:.3f}  "
              f"kl_align={raw_geometry['kl_alignment']:.3f}  "
              f"gap={raw_geometry['alignment_gap']:+.3f}  "
              f"eff_rank={eff_rank:.1f}")

    return results_per_layer


def run_single_point(n_states: int, config: CompressionExperimentConfig,
                     device: str = 'cpu', resume: bool = False):
    """Train transformer and evaluate probes for a single |S| value."""
    seed = config.seed
    print(f"\n{'='*60}")
    print(f"|S| = {n_states}  (d_resid = {config.d_model}, "
          f"compression ratio = {n_states/config.d_model:.2f}, seed={seed})")
    print(f"{'='*60}")

    out_dir = Path(f'compression_results/n_states_{n_states}/seed_{seed}')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate HMM (seed determines the HMM structure)
    rng = np.random.default_rng(seed)
    print(f"Generating random HMM with {n_states} states...")
    hmm = random_sparse_hmm(n_states, n_tokens=config.n_tokens, rng=rng)
    entropy = hmm.entropy_rate()
    print(f"  Entropy rate: {entropy:.4f} nats")

    # Save HMM
    np.savez(str(out_dir / 'hmm.npz'),
             T=hmm.T, stationary=hmm.stationary)

    # Create datasets
    print("Creating datasets...")
    rng_train = np.random.default_rng(seed + 1)
    rng_eval = np.random.default_rng(seed + 2)
    rng_analysis = np.random.default_rng(seed + 3)

    train_dataset = GeneralHMMDataset(
        hmm, config.n_train_sequences, config.seq_length, rng_train)
    eval_dataset = GeneralHMMDataset(
        hmm, config.n_eval_sequences, config.seq_length, rng_eval)
    analysis_dataset = GeneralHMMDataset(
        hmm, config.n_analysis, config.seq_length, rng_analysis)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(
        eval_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    # Build transformer
    print("Training transformer...")
    model = Mess3Transformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_mlp=config.d_mlp,
        max_seq_len=config.seq_length,
        dropout=config.dropout,
    )

    # Resume from checkpoint if requested
    start_epoch = 0
    if resume:
        ckpt_path = out_dir / 'model_final.pt'
        if ckpt_path.exists():
            ckpt = torch.load(str(ckpt_path), map_location='cpu',
                              weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            start_epoch = ckpt.get('epochs_trained', 0)
            print(f"  Resumed from checkpoint at epoch {start_epoch}")
        else:
            print("  No checkpoint found, training from scratch")

    history = train_on_hmm(model, train_loader, eval_loader, config,
                           device=device, start_epoch=start_epoch)

    # Convergence diagnostic
    final_eval = history['eval_loss'][-1][1] if history['eval_loss'] else float('nan')
    convergence_gap = (final_eval - entropy) / entropy if entropy > 0 else float('nan')
    total_epochs = start_epoch + len(history['train_loss'])

    if convergence_gap > config.convergence_threshold:
        print(f"  WARNING: convergence gap {convergence_gap:.1%} exceeds "
              f"threshold {config.convergence_threshold:.0%}. "
              f"Model may be undertrained.")

    # Save checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_states': n_states,
        'entropy_rate': entropy,
        'epochs_trained': total_epochs,
    }, str(out_dir / 'model_final.pt'))

    # Save training history
    with open(str(out_dir / 'training_history.json'), 'w') as f:
        json.dump({
            'train_loss': history['train_loss'],
            'eval_loss': history['eval_loss'],
            'entropy_rate_nats': entropy,
            'epochs_trained': total_epochs,
            'convergence_gap': convergence_gap,
        }, f, indent=2)

    # Extract activations and run probes
    print("Running probe analysis...")
    probe_results = extract_and_probe(
        model, analysis_dataset, hmm, config, device=device)

    # Save probe results
    with open(str(out_dir / 'probe_metrics.json'), 'w') as f:
        json.dump(probe_results, f, indent=2)

    # Determine best layer (lowest KL for KL probe)
    best_layer = min(probe_results,
                     key=lambda k: probe_results[k]['kl_metrics']['kl_divergence'])

    # Find best layer for raw geometry (highest KL alignment)
    best_geo_layer = max(
        probe_results,
        key=lambda k: probe_results[k]['raw_geometry']['kl_alignment'])

    summary = {
        'n_states': n_states,
        'd_model': config.d_model,
        'seed': seed,
        'compression_ratio': n_states / config.d_model,
        'entropy_rate_nats': entropy,
        'final_eval_loss': final_eval,
        'convergence_gap': convergence_gap,
        'epochs_trained': total_epochs,
        'best_layer': best_layer,
        'best_geo_layer': best_geo_layer,
        'best_layer_mse_metrics': probe_results[best_layer]['mse_metrics'],
        'best_layer_kl_metrics': probe_results[best_layer]['kl_metrics'],
        'best_layer_raw_geometry': probe_results[best_geo_layer]['raw_geometry'],
        'best_layer_effective_rank': probe_results[best_geo_layer]['effective_rank'],
        'all_layers': probe_results,
    }

    print(f"\nBest probe layer: {best_layer}")
    print(f"  Convergence gap: {convergence_gap:.1%}")
    print(f"  MSE probe KL: {summary['best_layer_mse_metrics']['kl_divergence']:.6f}")
    print(f"  KL  probe KL: {summary['best_layer_kl_metrics']['kl_divergence']:.6f}")
    geo = probe_results[best_geo_layer]['raw_geometry']
    print(f"  Raw geometry ({best_geo_layer}): "
          f"euc={geo['euclidean_alignment']:.3f}  "
          f"kl={geo['kl_alignment']:.3f}  "
          f"gap={geo['alignment_gap']:+.3f}")

    return summary


def run_sweep(config: CompressionExperimentConfig, device: str = 'cpu'):
    """Run the full |S| sweep across all seeds."""
    from copy import deepcopy

    seeds = [config.seed + i * 1000 for i in range(config.n_seeds)]
    results = {}

    for n_states in config.n_states_sweep:
        for seed in seeds:
            config_copy = deepcopy(config)
            config_copy.n_states = n_states
            config_copy.seed = seed
            summary = run_single_point(n_states, config_copy, device=device)
            results[(n_states, seed)] = summary

    # Print summary table
    print("\n" + "=" * 80)
    print("COMPRESSION SWEEP SUMMARY")
    print("=" * 80)
    print(f"{'|S|':>5} {'seed':>6} {'ratio':>6} {'gap%':>6} "
          f"{'MSE_kl':>10} {'KL_kl':>10}")
    print("-" * 80)
    for (n_states, seed) in sorted(results.keys()):
        s = results[(n_states, seed)]
        mm = s['best_layer_mse_metrics']
        km = s['best_layer_kl_metrics']
        print(f"  {n_states:>5} {seed:>6} {s['compression_ratio']:>6.2f} "
              f"{s['convergence_gap']:>5.1%} "
              f"{mm['kl_divergence']:>10.4f} {km['kl_divergence']:>10.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Experiment 3: Genuine Compression Regime')
    parser.add_argument('--n-states', type=int, default=None,
                        help='Run single point at this |S| value')
    parser.add_argument('--sweep', action='store_true',
                        help='Run full |S| sweep')
    parser.add_argument('--device', type=str, default=None,
                        help='Device: cpu, mps, cuda')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--n-analysis', type=int, default=None,
                        help='Number of sequences for probe analysis')
    parser.add_argument('--kl-epochs', type=int, default=None,
                        help='KL probe training epochs')
    parser.add_argument('--kl-lr', type=float, default=None,
                        help='KL probe learning rate')
    parser.add_argument('--seq-length', type=int, default=None,
                        help='Override sequence length')
    parser.add_argument('--n-train', type=int, default=None,
                        help='Override number of training sequences')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: from config)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from saved checkpoint')
    args = parser.parse_args()

    config = CompressionExperimentConfig()
    if args.epochs is not None:
        config.n_epochs = args.epochs
    if args.n_analysis is not None:
        config.n_analysis = args.n_analysis
    if args.kl_epochs is not None:
        config.kl_probe_epochs = args.kl_epochs
    if args.kl_lr is not None:
        config.kl_probe_lr = args.kl_lr
    if args.seq_length is not None:
        config.seq_length = args.seq_length
    if args.n_train is not None:
        config.n_train_sequences = args.n_train
    if args.seed is not None:
        config.seed = args.seed

    device_str = args.device or 'cpu'

    if args.sweep:
        run_sweep(config, device=device_str)
    elif args.n_states is not None:
        config.n_states = args.n_states
        run_single_point(args.n_states, config, device=device_str,
                         resume=args.resume)
    else:
        parser.print_help()
        print("\nProvide --n-states N or --sweep")


if __name__ == '__main__':
    main()
