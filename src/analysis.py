"""Activation extraction, PCA, linear probes, and geometry analysis."""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from pathlib import Path

from .transformer import Mess3Transformer
from .dataset import create_dataloaders, NonErgodicMess3Dataset
from .belief import compute_oracle_beliefs
from .mess3 import Mess3Process
from .config import ExperimentConfig


def load_trained_model(checkpoint_path: str, config: ExperimentConfig = None,
                       device: str = 'cpu'):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if config is None:
        config = checkpoint['config']
    model = Mess3Transformer.from_config(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, config


def extract_activations(model, dataset: NonErgodicMess3Dataset,
                        processes: list[Mess3Process],
                        n_sequences: int = None, device: str = 'cpu',
                        batch_size: int = 256):
    """Extract residual stream activations and compute ground-truth beliefs.

    Returns:
        result: dict with keys:
            'activations': dict mapping layer name -> (N, L-1, d_model)
            'component_labels': (N,) int array
            'tokens': (N, L) int array
            'beliefs': (N, L, 3) oracle beliefs
            'input_tokens': (N, L-1) int array (what model sees)
    """
    if n_sequences is None:
        n_sequences = len(dataset)
    n_sequences = min(n_sequences, len(dataset))

    # Collect data
    all_tokens = dataset.tokens[:n_sequences].numpy()
    all_labels = dataset.component_labels[:n_sequences].numpy()
    input_tokens = dataset.tokens[:n_sequences, :-1]  # (N, L-1)

    # Compute oracle beliefs on full token sequences
    beliefs = compute_oracle_beliefs(all_tokens, all_labels, processes)

    # Extract activations in batches
    model.eval()
    all_activations = {}

    for start in range(0, n_sequences, batch_size):
        end = min(start + batch_size, n_sequences)
        batch_input = input_tokens[start:end].to(device)

        with torch.no_grad():
            _, acts = model(batch_input, return_activations=True)

        for key, val in acts.items():
            if 'attn_weights' in key:
                continue  # skip attention weights for memory
            val_np = val.cpu().numpy()
            if key not in all_activations:
                all_activations[key] = []
            all_activations[key].append(val_np)

    # Concatenate batches
    activations = {k: np.concatenate(v, axis=0) for k, v in all_activations.items()}

    return {
        'activations': activations,
        'component_labels': all_labels,
        'tokens': all_tokens,
        'beliefs': beliefs,
        'input_tokens': input_tokens.numpy(),
    }


def extract_attention_weights(model, dataset: NonErgodicMess3Dataset,
                              n_sequences: int = 1000, device: str = 'cpu',
                              batch_size: int = 256):
    """Extract attention weights for attention pattern analysis.

    Returns:
        attn_weights: dict mapping 'attn_weights_{layer}' -> (N, H, L-1, L-1)
        component_labels: (N,) int array
    """
    n_sequences = min(n_sequences, len(dataset))
    input_tokens = dataset.tokens[:n_sequences, :-1]
    labels = dataset.component_labels[:n_sequences].numpy()

    model.eval()
    all_attn = {}

    for start in range(0, n_sequences, batch_size):
        end = min(start + batch_size, n_sequences)
        batch_input = input_tokens[start:end].to(device)

        with torch.no_grad():
            _, acts = model(batch_input, return_activations=True)

        for key, val in acts.items():
            if 'attn_weights' not in key:
                continue
            val_np = val.cpu().numpy()
            if key not in all_attn:
                all_attn[key] = []
            all_attn[key].append(val_np)

    attn_weights = {k: np.concatenate(v, axis=0) for k, v in all_attn.items()}
    return attn_weights, labels


def pca_analysis(activations: dict, component_labels: np.ndarray,
                 n_components: int = 6):
    """Run PCA on residual stream at each layer.

    Args:
        activations: dict mapping layer name -> (N, L, d_model)
        component_labels: (N,) int array
        n_components: number of PCA components

    Returns:
        pca_results: dict mapping layer name -> {
            'pca': fitted PCA object,
            'projected': (N*L, n_components) projected data,
            'labels': (N*L,) repeated component labels,
            'explained_variance_ratio': (n_components,)
        }
    """
    results = {}
    layer_keys = [k for k in activations if k not in ('embedding',)]

    for key in ['embedding'] + layer_keys:
        if key not in activations:
            continue
        acts = activations[key]  # (N, L, d_model)
        N, L, D = acts.shape

        # Flatten to (N*L, d_model)
        flat = acts.reshape(-1, D)
        labels_repeated = np.repeat(component_labels, L)

        pca = PCA(n_components=min(n_components, D))
        projected = pca.fit_transform(flat)

        results[key] = {
            'pca': pca,
            'projected': projected,
            'labels': labels_repeated,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'N': N, 'L': L,
        }

    return results


def linear_belief_regression(activations: dict, beliefs: np.ndarray,
                             component_labels: np.ndarray,
                             processes: list[Mess3Process]):
    """Linear regression from activations to oracle belief states.

    Fits per-layer and per-position regressions.

    Returns:
        results: dict with:
            'per_layer': dict mapping layer -> {'r2_overall', 'r2_per_component', 'model'}
            'per_position': dict mapping (layer, position) -> {'r2_overall'}
    """
    results = {'per_layer': {}, 'per_position': {}}
    K = len(processes)

    # Beliefs at positions 0..L-1 correspond to model input positions 0..L-1
    # beliefs shape is (N, L_full, 3), input activations are (N, L-1, d_model)
    # beliefs[:, t, :] = belief BEFORE seeing token t
    # activation at position t = after processing tokens 0..t
    # So activation at position t should predict beliefs[:, t+1, :] (belief after seeing token t)

    for layer_name, acts in activations.items():
        N, L, D = acts.shape

        # Use beliefs at positions 1..L (post-update beliefs)
        # These align with activations at positions 0..L-1
        target_beliefs = beliefs[:, 1:L+1, :]  # (N, L, 3)

        # Flatten
        X = acts.reshape(-1, D)
        Y = target_beliefs.reshape(-1, 3)
        labels_flat = np.repeat(component_labels, L)

        # Overall regression
        reg = LinearRegression()
        reg.fit(X, Y)
        Y_pred = reg.predict(X)
        r2 = r2_score(Y, Y_pred, multioutput='uniform_average')

        # Per-component R²
        r2_per_comp = []
        for k in range(K):
            mask = labels_flat == k
            if mask.sum() > 0:
                r2_k = r2_score(Y[mask], Y_pred[mask], multioutput='uniform_average')
                r2_per_comp.append(r2_k)
            else:
                r2_per_comp.append(float('nan'))

        results['per_layer'][layer_name] = {
            'r2_overall': r2,
            'r2_per_component': r2_per_comp,
            'model': reg,
        }

        # Per-position R² (for this layer)
        for pos in range(L):
            X_pos = acts[:, pos, :]
            Y_pos = target_beliefs[:, pos, :]
            reg_pos = LinearRegression()
            reg_pos.fit(X_pos, Y_pos)
            Y_pos_pred = reg_pos.predict(X_pos)
            r2_pos = r2_score(Y_pos, Y_pos_pred, multioutput='uniform_average')
            results['per_position'][(layer_name, pos)] = {
                'r2_overall': r2_pos,
            }

    return results


def component_identification_probes(activations: dict,
                                    component_labels: np.ndarray):
    """Train linear probes for component identification at each layer and position.

    Returns:
        results: dict mapping layer -> {
            'overall_accuracy': float,
            'per_position_accuracy': list of floats (one per position),
            'per_position_models': list of fitted LogisticRegression
        }
    """
    results = {}
    K = len(np.unique(component_labels))

    for layer_name, acts in activations.items():
        N, L, D = acts.shape

        # Overall probe (all positions)
        X_all = acts.reshape(-1, D)
        Y_all = np.repeat(component_labels, L)
        probe_all = LogisticRegression(max_iter=1000, solver='lbfgs')
        probe_all.fit(X_all, Y_all)
        overall_acc = accuracy_score(Y_all, probe_all.predict(X_all))

        # Per-position probes
        pos_accs = []
        pos_models = []
        for pos in range(L):
            X_pos = acts[:, pos, :]
            Y_pos = component_labels
            probe = LogisticRegression(max_iter=1000, solver='lbfgs')
            probe.fit(X_pos, Y_pos)
            acc = accuracy_score(Y_pos, probe.predict(X_pos))
            pos_accs.append(acc)
            pos_models.append(probe)

        results[layer_name] = {
            'overall_accuracy': overall_acc,
            'per_position_accuracy': pos_accs,
            'per_position_models': pos_models,
        }

    return results


def subspace_orthogonality(activations: dict, component_labels: np.ndarray,
                           n_subspace_dims: int = 3):
    """Analyze orthogonality of per-component principal subspaces.

    For each layer, fits PCA separately to each component's activations,
    then measures cosine similarity between the principal subspaces.

    Returns:
        results: dict mapping layer -> {
            'cosine_similarities': (K, K, n_subspace_dims) pairwise cos sim
                between principal axes
            'subspace_overlap': (K, K) average overlap between subspaces
        }
    """
    K = len(np.unique(component_labels))
    results = {}

    for layer_name, acts in activations.items():
        N, L, D = acts.shape
        flat = acts.reshape(-1, D)
        labels_flat = np.repeat(component_labels, L)

        # Fit PCA per component
        component_axes = []
        for k in range(K):
            mask = labels_flat == k
            pca_k = PCA(n_components=n_subspace_dims)
            pca_k.fit(flat[mask])
            component_axes.append(pca_k.components_)  # (n_dims, D)

        # Pairwise cosine similarities between principal axes
        cos_sim = np.zeros((K, K, n_subspace_dims))
        subspace_overlap = np.zeros((K, K))

        for i in range(K):
            for j in range(K):
                # Cosine similarity between corresponding axes
                for d in range(n_subspace_dims):
                    v1 = component_axes[i][d]
                    v2 = component_axes[j][d]
                    cos_sim[i, j, d] = np.abs(np.dot(v1, v2)) / (
                        np.linalg.norm(v1) * np.linalg.norm(v2))

                # Subspace overlap: average of absolute cosine similarities
                # between all pairs of axes
                overlap = 0.0
                count = 0
                for d1 in range(n_subspace_dims):
                    for d2 in range(n_subspace_dims):
                        v1 = component_axes[i][d1]
                        v2 = component_axes[j][d2]
                        overlap += np.abs(np.dot(v1, v2)) / (
                            np.linalg.norm(v1) * np.linalg.norm(v2))
                        count += 1
                subspace_overlap[i, j] = overlap / count

        results[layer_name] = {
            'cosine_similarities': cos_sim,
            'subspace_overlap': subspace_overlap,
            'component_axes': component_axes,
        }

    return results


def recovered_simplices(activations: dict, beliefs: np.ndarray,
                        component_labels: np.ndarray,
                        regression_results: dict = None):
    """Recover belief simplices from activations using per-component linear regression.

    Fits a separate linear regression for each component, mapping that
    component's activations to its oracle beliefs. This captures the
    component-specific linear encoding of belief geometry, avoiding the
    smearing that occurs when a single global regression compromises
    across components with different fractal structures.

    Args:
        activations: dict mapping layer name -> (N, L, d_model)
        beliefs: (N, L_full, 3) oracle beliefs
        component_labels: (N,) int array
        regression_results: unused (kept for API compatibility)

    Returns:
        results: dict mapping layer -> {
            'per_component': list of {
                'predicted_beliefs': (n_k, L, 3) regression-predicted beliefs,
                'beliefs': (n_k, L, 3) ground-truth beliefs,
                'r2': float, per-component R²
            }
        }
    """
    K = len(np.unique(component_labels))
    results = {}

    for layer_name, acts in activations.items():
        N, L, D = acts.shape
        target_beliefs = beliefs[:, 1:L+1, :]

        per_comp = []
        for k in range(K):
            mask = component_labels == k
            acts_k = acts[mask]  # (n_k, L, D)
            beliefs_k = target_beliefs[mask]  # (n_k, L, 3)

            # Fit per-component linear regression
            X_k = acts_k.reshape(-1, D)
            Y_k = beliefs_k.reshape(-1, 3)
            reg_k = LinearRegression()
            reg_k.fit(X_k, Y_k)
            r2_k = reg_k.score(X_k, Y_k)

            pred_k = reg_k.predict(X_k)
            # Clip and normalize to valid probability simplex
            pred_k = np.clip(pred_k, 0, None)
            row_sums = pred_k.sum(axis=1, keepdims=True)
            pred_k = pred_k / np.where(row_sums > 0, row_sums, 1)

            per_comp.append({
                'predicted_beliefs': pred_k.reshape(mask.sum(), L, 3),
                'beliefs': beliefs_k,
                'r2': r2_k,
            })

        results[layer_name] = {'per_component': per_comp}

    return results


def run_full_analysis(checkpoint_path: str, config: ExperimentConfig = None,
                      device: str = 'cpu', n_analysis: int = 5000):
    """Run the complete analysis pipeline.

    Returns a dict with all analysis results.
    """
    print("Loading model...")
    model, config = load_trained_model(checkpoint_path, config, device)

    print("Creating analysis dataset...")
    processes = [Mess3Process(c['alpha'], c['x']) for c in config.components]
    rng = np.random.default_rng(config.seed + 100)
    analysis_dataset = NonErgodicMess3Dataset(
        processes, n_analysis, config.seq_length, rng)

    print("Extracting activations...")
    data = extract_activations(model, analysis_dataset, processes,
                               n_sequences=n_analysis, device=device)

    print("Running PCA analysis...")
    pca_results = pca_analysis(data['activations'], data['component_labels'],
                               n_components=config.pca_components)

    print("Running belief regression...")
    regression_results = linear_belief_regression(
        data['activations'], data['beliefs'],
        data['component_labels'], processes)

    print("Running component identification probes...")
    probe_results = component_identification_probes(
        data['activations'], data['component_labels'])

    print("Running subspace orthogonality analysis...")
    ortho_results = subspace_orthogonality(
        data['activations'], data['component_labels'])

    print("Computing recovered simplices...")
    simplex_results = recovered_simplices(
        data['activations'], data['beliefs'],
        data['component_labels'], regression_results)

    print("Extracting attention weights...")
    attn_weights, attn_labels = extract_attention_weights(
        model, analysis_dataset, n_sequences=min(1000, n_analysis), device=device)

    print("Analysis complete!")
    return {
        'data': data,
        'pca': pca_results,
        'regression': regression_results,
        'probes': probe_results,
        'orthogonality': ortho_results,
        'simplices': simplex_results,
        'attention': {'weights': attn_weights, 'labels': attn_labels},
        'config': config,
        'processes': processes,
    }
