import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import json


# Barycentric coordinate conversion for 2-simplex visualization
def to_cartesian(beliefs: np.ndarray) -> tuple:
    """Convert belief vectors [b0, b1, b2] to 2D Cartesian coordinates.

    Maps the 2-simplex to an equilateral triangle with vertices:
      state 0 at (0, 0)
      state 1 at (1, 0)
      state 2 at (0.5, sqrt(3)/2)
    """
    b0, b1, b2 = beliefs[:, 0], beliefs[:, 1], beliefs[:, 2]
    x = b1 + b2 / 2
    y = b2 * np.sqrt(3) / 2
    return x, y


def draw_simplex_outline(ax, color='gray', linewidth=0.5):
    """Draw the outline of the 2-simplex (equilateral triangle)."""
    vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2], [0, 0]])
    ax.plot(vertices[:, 0], vertices[:, 1], color=color, linewidth=linewidth)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    # Label vertices
    offset = 0.04
    ax.text(0 - offset, 0 - offset, '$s_0$', ha='center', fontsize=8)
    ax.text(1 + offset, 0 - offset, '$s_1$', ha='center', fontsize=8)
    ax.text(0.5, np.sqrt(3)/2 + offset, '$s_2$', ha='center', fontsize=8)


def plot_ground_truth_fractals(beliefs_per_component: list,
                               component_names: list,
                               save_path: str = None):
    """Plot the ground-truth belief fractals for each Mess3 component.

    Args:
        beliefs_per_component: list of (N, L, 3) belief arrays, one per component
        component_names: list of component name strings
        save_path: if provided, save figure to this path
    """
    K = len(beliefs_per_component)
    fig, axes = plt.subplots(1, K, figsize=(4 * K, 4))
    if K == 1:
        axes = [axes]

    colors = ['#e41a1c', '#377eb8', '#4daf4a']

    for i, (beliefs, name) in enumerate(zip(beliefs_per_component, component_names)):
        ax = axes[i]
        # Flatten across sequences and positions (skip position 0 which is always uniform)
        b_flat = beliefs[:, 1:, :].reshape(-1, 3)
        x, y = to_cartesian(b_flat)

        ax.scatter(x, y, s=0.1, alpha=0.3, c=colors[i], rasterized=True)
        draw_simplex_outline(ax)
        ax.set_title(f'Component {name}', fontsize=12)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.close(fig)
    return fig


# --- Component colors used throughout ---
COMP_COLORS = ['#e41a1c', '#377eb8', '#4daf4a']
COMP_NAMES_DEFAULT = ['A', 'B', 'C']


def _save_fig(fig, save_path):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.close(fig)


def plot_training_curves(history_path: str, entropy_rates_nats: list = None,
                         component_names: list = None, save_path: str = None):
    """Plot training loss and per-component loss vs theoretical entropy rates."""
    with open(history_path) as f:
        history = json.load(f)

    if component_names is None:
        component_names = COMP_NAMES_DEFAULT
    if entropy_rates_nats is None:
        entropy_rates_nats = history.get('entropy_rates_nats', [])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: overall train/eval loss
    ax = axes[0]
    ax.plot(history['train_loss'], label='Train', alpha=0.7)
    eval_epochs = [e[0] for e in history['eval_loss']]
    eval_vals = [e[1] for e in history['eval_loss']]
    ax.plot(eval_epochs, eval_vals, 'o-', label='Eval', markersize=3)

    # Weighted theoretical entropy (average across components)
    if entropy_rates_nats:
        avg_h = np.mean(entropy_rates_nats)
        ax.axhline(avg_h, color='k', linestyle='--', alpha=0.5,
                   label=f'Avg entropy rate ({avg_h:.3f} nats)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (nats)')
    ax.set_title('Training Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: per-component loss vs theoretical
    ax = axes[1]
    comp_epochs = [e[0] for e in history['component_loss']]
    comp_losses = np.array([e[1] for e in history['component_loss']])
    for k in range(comp_losses.shape[1]):
        ax.plot(comp_epochs, comp_losses[:, k], 'o-', color=COMP_COLORS[k],
                label=f'{component_names[k]}', markersize=3)
        if k < len(entropy_rates_nats):
            ax.axhline(entropy_rates_nats[k], color=COMP_COLORS[k],
                       linestyle='--', alpha=0.5,
                       label=f'{component_names[k]} entropy ({entropy_rates_nats[k]:.3f})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (nats)')
    ax.set_title('Per-Component Loss vs Entropy Rate')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_pca_by_layer(pca_results: dict, component_names: list = None,
                      save_path: str = None):
    """PCA scatter plots at each layer, colored by component."""
    if component_names is None:
        component_names = COMP_NAMES_DEFAULT

    layer_keys = sorted([k for k in pca_results if k != 'embedding'],
                        key=lambda x: int(x.split('_')[-1]) if x.startswith('layer') else 999)
    all_keys = ['embedding'] + layer_keys
    all_keys = [k for k in all_keys if k in pca_results]

    n = len(all_keys)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for idx, key in enumerate(all_keys):
        ax = axes[idx]
        res = pca_results[key]
        proj = res['projected']
        labels = res['labels']
        evr = res['explained_variance_ratio']

        for k in range(len(component_names)):
            mask = labels == k
            ax.scatter(proj[mask, 0], proj[mask, 1], s=0.5, alpha=0.3,
                       c=COMP_COLORS[k], label=component_names[k], rasterized=True)

        title = key.replace('_', ' ').title()
        ax.set_title(f'{title}\nPC1: {evr[0]:.1%}, PC2: {evr[1]:.1%}', fontsize=10)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        if idx == 0:
            ax.legend(fontsize=7, markerscale=5)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_pca_by_position(pca_results: dict, layer_key: str = 'final',
                         component_names: list = None, save_path: str = None):
    """PCA at a single layer, split by context position, showing how
    component clusters sharpen with more context."""
    if component_names is None:
        component_names = COMP_NAMES_DEFAULT
    if layer_key not in pca_results:
        print(f"Layer {layer_key} not in PCA results")
        return

    res = pca_results[layer_key]
    N, L = res['N'], res['L']
    proj = res['projected'].reshape(N, L, -1)
    labels = res['labels'].reshape(N, L)[:, 0]  # same across positions

    positions_to_show = [0, 2, 5, 9, L - 1]
    positions_to_show = [p for p in positions_to_show if p < L]

    n = len(positions_to_show)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5))
    if n == 1:
        axes = [axes]

    for idx, pos in enumerate(positions_to_show):
        ax = axes[idx]
        for k in range(len(component_names)):
            mask = labels == k
            ax.scatter(proj[mask, pos, 0], proj[mask, pos, 1], s=1, alpha=0.3,
                       c=COMP_COLORS[k], label=component_names[k], rasterized=True)
        ax.set_title(f'Position {pos}', fontsize=10)
        ax.set_xlabel('PC1')
        if idx == 0:
            ax.set_ylabel('PC2')
            ax.legend(fontsize=7, markerscale=5)
        ax.grid(True, alpha=0.2)

    fig.suptitle(f'PCA at {layer_key} — Component Separation vs Position', fontsize=12, y=1.02)
    plt.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_r2_progression(regression_results: dict, component_names: list = None,
                        save_path: str = None):
    """Plot R² vs layer and vs position."""
    if component_names is None:
        component_names = COMP_NAMES_DEFAULT

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: R² vs layer
    ax = axes[0]
    layer_data = regression_results['per_layer']
    layer_keys = sorted([k for k in layer_data if k != 'embedding'],
                        key=lambda x: int(x.split('_')[-1]) if x.startswith('layer') else 999)
    all_keys = ['embedding'] + layer_keys
    all_keys = [k for k in all_keys if k in layer_data]

    r2_overall = [layer_data[k]['r2_overall'] for k in all_keys]
    ax.plot(range(len(all_keys)), r2_overall, 'ko-', label='Overall', linewidth=2)

    for comp_idx in range(len(component_names)):
        r2_comp = [layer_data[k]['r2_per_component'][comp_idx] for k in all_keys]
        ax.plot(range(len(all_keys)), r2_comp, 'o-', color=COMP_COLORS[comp_idx],
                label=component_names[comp_idx])

    ax.set_xticks(range(len(all_keys)))
    ax.set_xticklabels([k.replace('_', '\n') for k in all_keys], fontsize=8)
    ax.set_ylabel('R²')
    ax.set_title('Belief Regression R² by Layer')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Right: R² vs position (at final layer)
    ax = axes[1]
    final_layer = all_keys[-1]
    pos_keys = sorted([k for k in regression_results['per_position']
                       if k[0] == final_layer], key=lambda x: x[1])
    positions = [k[1] for k in pos_keys]
    r2_by_pos = [regression_results['per_position'][k]['r2_overall'] for k in pos_keys]

    ax.plot(positions, r2_by_pos, 'ko-', linewidth=2)
    ax.set_xlabel('Context Position')
    ax.set_ylabel('R²')
    ax.set_title(f'Belief Regression R² by Position ({final_layer})')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_component_identification(probe_results: dict,
                                  bayesian_accuracy: np.ndarray = None,
                                  component_names: list = None,
                                  save_path: str = None):
    """Plot component identification accuracy vs position across layers.

    Args:
        probe_results: from component_identification_probes
        bayesian_accuracy: (L,) array of Bayesian optimal accuracy per position
    """
    if component_names is None:
        component_names = COMP_NAMES_DEFAULT

    layer_keys = sorted([k for k in probe_results if k != 'embedding'],
                        key=lambda x: int(x.split('_')[-1]) if x.startswith('layer') else 999)
    all_keys = ['embedding'] + layer_keys
    all_keys = [k for k in all_keys if k in probe_results]

    fig, ax = plt.subplots(figsize=(8, 5))

    layer_colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(all_keys)))
    for idx, key in enumerate(all_keys):
        accs = probe_results[key]['per_position_accuracy']
        label = key.replace('_', ' ').title()
        ax.plot(range(len(accs)), accs, 'o-', color=layer_colors[idx],
                label=label, markersize=4)

    if bayesian_accuracy is not None:
        ax.plot(range(len(bayesian_accuracy)), bayesian_accuracy, 'k--',
                linewidth=2, label='Bayesian Optimal')

    ax.axhline(1.0 / 3.0, color='gray', linestyle=':', alpha=0.5, label='Chance')
    ax.set_xlabel('Context Position')
    ax.set_ylabel('Component ID Accuracy')
    ax.set_title('Component Identification vs Context Position')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_recovered_simplices(simplex_results: dict, layer_key: str = 'final',
                             component_names: list = None, save_path: str = None):
    """Plot recovered simplex geometry vs ground-truth beliefs.

    Top row: beliefs predicted by linear regression from activations,
    plotted in barycentric coordinates (same space as ground truth).
    Bottom row: ground-truth Bayesian beliefs in barycentric coordinates.
    """
    if component_names is None:
        component_names = COMP_NAMES_DEFAULT
    if layer_key not in simplex_results:
        print(f"Layer {layer_key} not in simplex results")
        return

    K = len(simplex_results[layer_key]['per_component'])
    fig, axes = plt.subplots(2, K, figsize=(4 * K, 8))

    for k in range(K):
        comp_data = simplex_results[layer_key]['per_component'][k]
        predicted = comp_data['predicted_beliefs']  # (n_k, L, 3)
        beliefs = comp_data['beliefs']  # (n_k, L, 3)

        # Top row: recovered (regression-predicted beliefs in barycentric coords)
        ax = axes[0, k]
        flat_pred = predicted[:, 1:, :].reshape(-1, 3)  # skip pos 0
        px, py = to_cartesian(flat_pred)
        ax.scatter(px, py, s=0.3, alpha=0.2,
                   c=COMP_COLORS[k], rasterized=True)
        draw_simplex_outline(ax)
        ax.set_title(f'{component_names[k]} — Recovered', fontsize=10)

        # Bottom row: ground-truth beliefs
        ax = axes[1, k]
        flat_beliefs = beliefs[:, 1:, :].reshape(-1, 3)
        bx, by = to_cartesian(flat_beliefs)
        ax.scatter(bx, by, s=0.3, alpha=0.2, c=COMP_COLORS[k], rasterized=True)
        draw_simplex_outline(ax)
        ax.set_title(f'{component_names[k]} — Ground Truth', fontsize=10)

    axes[0, 0].set_ylabel('Recovered\n(linear readout)', fontsize=10)
    axes[1, 0].set_ylabel('Ground Truth\n(Bayesian beliefs)', fontsize=10)
    fig.suptitle(f'Recovered vs Ground-Truth Simplices ({layer_key})', fontsize=13, y=1.01)
    plt.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_subspace_orthogonality(ortho_results: dict, layer_key: str = 'final',
                                component_names: list = None, save_path: str = None):
    """Heatmap of subspace overlap between components."""
    if component_names is None:
        component_names = COMP_NAMES_DEFAULT
    if layer_key not in ortho_results:
        print(f"Layer {layer_key} not in orthogonality results")
        return

    overlap = ortho_results[layer_key]['subspace_overlap']
    K = overlap.shape[0]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(overlap, cmap='RdYlBu_r', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Subspace Overlap')

    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(component_names[:K])
    ax.set_yticklabels(component_names[:K])

    for i in range(K):
        for j in range(K):
            ax.text(j, i, f'{overlap[i, j]:.2f}', ha='center', va='center',
                    fontsize=12, color='white' if overlap[i, j] > 0.5 else 'black')

    ax.set_title(f'Subspace Overlap ({layer_key})')
    plt.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_subspace_cosine_matrix(ortho_results: dict, layer_key: str = 'final',
                                component_names: list = None, save_path: str = None):
    """9x9 heatmap of absolute cosine similarities between all principal axes."""
    if component_names is None:
        component_names = COMP_NAMES_DEFAULT
    if layer_key not in ortho_results:
        print(f"Layer {layer_key} not in orthogonality results")
        return

    axes_list = ortho_results[layer_key]['component_axes']
    K = len(axes_list)
    n_dims = axes_list[0].shape[0]
    total = K * n_dims

    # Build full cosine similarity matrix
    cos_matrix = np.zeros((total, total))
    for i in range(total):
        for j in range(total):
            ci, di = divmod(i, n_dims)
            cj, dj = divmod(j, n_dims)
            v1 = axes_list[ci][di]
            v2 = axes_list[cj][dj]
            cos_matrix[i, j] = np.abs(np.dot(v1, v2)) / (
                np.linalg.norm(v1) * np.linalg.norm(v2))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cos_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='|Cosine Similarity|')

    # Labels: A-PC1, A-PC2, etc.
    labels = [f'{component_names[c]}-PC{d+1}' for c in range(K) for d in range(n_dims)]
    ax.set_xticks(range(total))
    ax.set_yticks(range(total))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    # Draw block separators
    for sep in range(1, K):
        pos = sep * n_dims - 0.5
        ax.axhline(pos, color='white', linewidth=2)
        ax.axvline(pos, color='white', linewidth=2)

    ax.set_title(f'Principal Axis Cosine Similarity ({layer_key})')
    plt.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_attention_patterns(attn_weights: dict, component_labels: np.ndarray,
                            component_names: list = None, save_path: str = None):
    """Plot average attention patterns per component and layer."""
    if component_names is None:
        component_names = COMP_NAMES_DEFAULT

    layer_keys = sorted(attn_weights.keys(),
                        key=lambda x: int(x.split('_')[-1]))
    n_layers = len(layer_keys)
    K = len(np.unique(component_labels))

    fig, axes = plt.subplots(K, n_layers, figsize=(4 * n_layers, 3.5 * K))
    if n_layers == 1:
        axes = axes[:, np.newaxis]
    if K == 1:
        axes = axes[np.newaxis, :]

    for li, layer_key in enumerate(layer_keys):
        weights = attn_weights[layer_key]  # (N, H, L, L)
        # Average over heads
        avg_attn = weights.mean(axis=1)  # (N, L, L)

        for k in range(K):
            mask = component_labels == k
            avg_k = avg_attn[mask].mean(axis=0)  # (L, L)

            ax = axes[k, li]
            im = ax.imshow(avg_k, cmap='hot', vmin=0)
            if k == 0:
                ax.set_title(f'Layer {li}', fontsize=10)
            if li == 0:
                ax.set_ylabel(f'Comp {component_names[k]}\nQuery pos')
            ax.set_xlabel('Key pos')

    fig.suptitle('Average Attention Patterns (across heads)', fontsize=12, y=1.01)
    plt.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_explained_variance(pca_results: dict, save_path: str = None):
    """Plot cumulative explained variance across layers."""
    layer_keys = sorted([k for k in pca_results if k != 'embedding'],
                        key=lambda x: int(x.split('_')[-1]) if x.startswith('layer') else 999)
    all_keys = ['embedding'] + layer_keys
    all_keys = [k for k in all_keys if k in pca_results]

    fig, ax = plt.subplots(figsize=(14, 3))
    layer_colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(all_keys)))

    for idx, key in enumerate(all_keys):
        evr = pca_results[key]['explained_variance_ratio']
        cumulative = np.cumsum(evr)
        label = key.replace('_', ' ').title()
        ax.plot(range(1, len(cumulative) + 1), cumulative, 'o-',
                color=layer_colors[idx], label=label, markersize=5)

    ax.set_xlabel('Number of PCA Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('Effective Dimensionality by Layer')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    _save_fig(fig, save_path)
    return fig


# --- Information-Geometric Probe Comparison Plots ---

def _get_ordered_layer_keys(data_dict: dict) -> list:
    """Get layer keys in order: embedding, layer_0, layer_1, ..., final."""
    layer_keys = sorted([k for k in data_dict if k not in ('embedding', 'final')],
                        key=lambda x: int(x.split('_')[-1]) if x.startswith('layer') else 999)
    all_keys = []
    if 'embedding' in data_dict:
        all_keys.append('embedding')
    all_keys.extend(layer_keys)
    if 'final' in data_dict:
        all_keys.append('final')
    return all_keys


def plot_probe_comparison(comparison_results: dict, save_path: str = None):
    """Bar chart comparing MSE and KL probe metrics side by side across layers."""
    mse_data = comparison_results['mse']
    kl_data = comparison_results['kl']

    layer_keys = _get_ordered_layer_keys(mse_data)
    layer_keys = [k for k in layer_keys if k in kl_data]

    metric_names = ['mse', 'kl_divergence', 'pairwise_r2_euclidean',
                    'pairwise_rho_kl', 'simplex_violation_rate', 'boundary_kl']
    metric_labels = ['MSE', 'KL Divergence', 'Pairwise R² (Euc)',
                     'Pairwise ρ (KL)', 'Simplex Violation %', 'Boundary KL']

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    x = np.arange(len(layer_keys))
    width = 0.35

    for idx, (metric, label) in enumerate(zip(metric_names, metric_labels)):
        ax = axes[idx]
        mse_vals = []
        kl_vals = []
        for lk in layer_keys:
            mse_m = mse_data[lk].get('metrics', mse_data[lk])
            kl_m = kl_data[lk].get('metrics', kl_data[lk]) if isinstance(kl_data[lk], dict) else kl_data[lk]
            mse_vals.append(mse_m.get(metric, float('nan')) if isinstance(mse_m, dict) else float('nan'))
            kl_vals.append(kl_m.get(metric, float('nan')) if isinstance(kl_m, dict) else float('nan'))

        ax.bar(x - width/2, mse_vals, width, label='MSE Probe', color='#4292c6')
        ax.bar(x + width/2, kl_vals, width, label='KL Probe', color='#e6550d')
        ax.set_xticks(x)
        ax.set_xticklabels([k.replace('_', '\n') for k in layer_keys], fontsize=7)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2, axis='y')

    fig.suptitle('MSE vs KL Probe: Head-to-Head Comparison', fontsize=13, y=1.01)
    plt.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_pca_compression_sweep(sweep_results: dict, save_path: str = None):
    """Plot probe quality vs PCA dimension for both probes."""
    k_values = sorted(sweep_results.keys())

    mse_mse_vals = [sweep_results[k]['mse_metrics']['mse'] for k in k_values]
    kl_mse_vals = [sweep_results[k]['kl_metrics']['mse'] for k in k_values]
    mse_kl_vals = [sweep_results[k]['mse_metrics']['kl_divergence'] for k in k_values]
    kl_kl_vals = [sweep_results[k]['kl_metrics']['kl_divergence'] for k in k_values]
    mse_bkl_vals = [sweep_results[k]['mse_metrics']['boundary_kl'] for k in k_values]
    kl_bkl_vals = [sweep_results[k]['kl_metrics']['boundary_kl'] for k in k_values]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # MSE in probability space
    ax = axes[0]
    ax.plot(k_values, mse_mse_vals, 'o-', color='#4292c6', label='MSE Probe')
    ax.plot(k_values, kl_mse_vals, 's-', color='#e6550d', label='KL Probe')
    ax.set_xlabel('PCA Dimensions (k)')
    ax.set_ylabel('MSE')
    ax.set_title('MSE in Probability Space')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)

    # KL divergence
    ax = axes[1]
    ax.plot(k_values, mse_kl_vals, 'o-', color='#4292c6', label='MSE Probe')
    ax.plot(k_values, kl_kl_vals, 's-', color='#e6550d', label='KL Probe')
    ax.set_xlabel('PCA Dimensions (k)')
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL Divergence')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)

    # Boundary KL
    ax = axes[2]
    ax.plot(k_values, mse_bkl_vals, 'o-', color='#4292c6', label='MSE Probe')
    ax.plot(k_values, kl_bkl_vals, 's-', color='#e6550d', label='KL Probe')
    ax.set_xlabel('PCA Dimensions (k)')
    ax.set_ylabel('Boundary KL')
    ax.set_title('Boundary Fidelity (KL)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)

    fig.suptitle('PCA Compression Sweep: MSE vs KL Probe', fontsize=13, y=1.02)
    plt.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_recovered_simplices_comparison(mse_simplex_results: dict,
                                        kl_simplex_results: dict,
                                        layer_key: str = 'layer_2',
                                        component_names: list = None,
                                        save_path: str = None):
    """Three-row simplex: MSE recovered (top), KL recovered (middle), ground truth (bottom)."""
    if component_names is None:
        component_names = COMP_NAMES_DEFAULT
    if layer_key not in mse_simplex_results or layer_key not in kl_simplex_results:
        print(f"Layer {layer_key} not in simplex results")
        return

    K = len(mse_simplex_results[layer_key]['per_component'])
    fig, axes = plt.subplots(3, K, figsize=(4 * K, 12))

    row_labels = ['MSE Probe\n(linear readout)', 'KL Probe\n(softmax-affine)',
                  'Ground Truth\n(Bayesian beliefs)']

    for k in range(K):
        # Top row: MSE recovered
        ax = axes[0, k]
        mse_pred = mse_simplex_results[layer_key]['per_component'][k]['predicted_beliefs']
        flat = mse_pred[:, 1:, :].reshape(-1, 3)
        px, py = to_cartesian(flat)
        ax.scatter(px, py, s=0.3, alpha=0.2, c=COMP_COLORS[k], rasterized=True)
        draw_simplex_outline(ax)
        ax.set_title(f'{component_names[k]}', fontsize=10)

        # Middle row: KL recovered
        ax = axes[1, k]
        kl_pred = kl_simplex_results[layer_key]['per_component'][k]['predicted_beliefs']
        flat = kl_pred[:, 1:, :].reshape(-1, 3)
        px, py = to_cartesian(flat)
        ax.scatter(px, py, s=0.3, alpha=0.2, c=COMP_COLORS[k], rasterized=True)
        draw_simplex_outline(ax)

        # Bottom row: ground truth
        ax = axes[2, k]
        gt = mse_simplex_results[layer_key]['per_component'][k]['beliefs']
        flat = gt[:, 1:, :].reshape(-1, 3)
        bx, by = to_cartesian(flat)
        ax.scatter(bx, by, s=0.3, alpha=0.2, c=COMP_COLORS[k], rasterized=True)
        draw_simplex_outline(ax)

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=10)

    fig.suptitle(f'Recovered vs Ground-Truth Simplices ({layer_key})', fontsize=13, y=1.01)
    plt.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_kl_layer_profile(mse_regression: dict, kl_regression: dict,
                          component_names: list = None,
                          save_path: str = None):
    """Plot MSE probe R² and KL probe metrics vs layer side by side."""
    if component_names is None:
        component_names = COMP_NAMES_DEFAULT

    mse_layer_data = mse_regression['per_layer']
    kl_layer_data = kl_regression['per_layer']

    layer_keys = _get_ordered_layer_keys(mse_layer_data)
    layer_keys = [k for k in layer_keys if k in kl_layer_data]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    x = range(len(layer_keys))

    # Panel 1: R² (MSE probe) vs layer
    ax = axes[0]
    r2_overall = [mse_layer_data[k]['r2_overall'] for k in layer_keys]
    ax.plot(x, r2_overall, 'ko-', label='Overall', linewidth=2)
    for ci in range(len(component_names)):
        r2_comp = [mse_layer_data[k]['r2_per_component'][ci] for k in layer_keys]
        ax.plot(x, r2_comp, 'o-', color=COMP_COLORS[ci], label=component_names[ci])
    ax.set_xticks(list(x))
    ax.set_xticklabels([k.replace('_', '\n') for k in layer_keys], fontsize=8)
    ax.set_ylabel('R²')
    ax.set_title('MSE Probe: R² by Layer')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Panel 2: KL divergence (KL probe) vs layer
    ax = axes[1]
    kl_overall = [kl_layer_data[k]['metrics']['kl_divergence'] for k in layer_keys]
    ax.plot(x, kl_overall, 'ko-', label='Overall', linewidth=2)
    for ci in range(len(component_names)):
        kl_comp = [kl_layer_data[k]['metrics_per_component'][ci]['kl_divergence']
                    for k in layer_keys]
        ax.plot(x, kl_comp, 'o-', color=COMP_COLORS[ci], label=component_names[ci])
    ax.set_xticks(list(x))
    ax.set_xticklabels([k.replace('_', '\n') for k in layer_keys], fontsize=8)
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL Probe: KL Divergence by Layer')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 3: Boundary KL comparison
    ax = axes[2]
    bkl_kl = [kl_layer_data[k]['metrics']['boundary_kl'] for k in layer_keys]
    ax.plot(x, bkl_kl, 's-', color='#e6550d', label='KL Probe', linewidth=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels([k.replace('_', '\n') for k in layer_keys], fontsize=8)
    ax.set_ylabel('Boundary KL')
    ax.set_title('Boundary Fidelity by Layer')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle('MSE vs KL Probe: Layer-by-Layer Profile', fontsize=13, y=1.02)
    plt.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_compression_sweep(sweep_data: dict, save_path: str = None):
    """Plot probe quality vs |S| for the compression experiment.

    Args:
        sweep_data: dict mapping str(n_states) -> {
            'n_states', 'compression_ratio', 'mse_metrics', 'kl_metrics', ...
        }
    """
    n_states_vals = sorted([int(k) for k in sweep_data.keys()])
    ratios = [sweep_data[str(n)]['compression_ratio'] for n in n_states_vals]

    mse_kl = [sweep_data[str(n)]['mse_metrics']['kl_divergence'] for n in n_states_vals]
    kl_kl = [sweep_data[str(n)]['kl_metrics']['kl_divergence'] for n in n_states_vals]
    mse_mse = [sweep_data[str(n)]['mse_metrics']['mse'] for n in n_states_vals]
    kl_mse = [sweep_data[str(n)]['kl_metrics']['mse'] for n in n_states_vals]
    mse_bkl = [sweep_data[str(n)]['mse_metrics']['boundary_kl'] for n in n_states_vals]
    kl_bkl = [sweep_data[str(n)]['kl_metrics']['boundary_kl'] for n in n_states_vals]
    mse_pr2 = [sweep_data[str(n)]['mse_metrics']['pairwise_r2_euclidean'] for n in n_states_vals]
    kl_pr2 = [sweep_data[str(n)]['kl_metrics']['pairwise_r2_euclidean'] for n in n_states_vals]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # KL divergence
    ax = axes[0, 0]
    ax.plot(n_states_vals, mse_kl, 'o-', color='#4292c6', label='MSE Probe', linewidth=2)
    ax.plot(n_states_vals, kl_kl, 's-', color='#e6550d', label='KL Probe', linewidth=2)
    ax.axvline(64, color='gray', linestyle='--', alpha=0.5, label='d_resid=64')
    ax.set_xlabel('|S| (number of hidden states)')
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL Divergence (lower is better)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # MSE
    ax = axes[0, 1]
    ax.plot(n_states_vals, mse_mse, 'o-', color='#4292c6', label='MSE Probe', linewidth=2)
    ax.plot(n_states_vals, kl_mse, 's-', color='#e6550d', label='KL Probe', linewidth=2)
    ax.axvline(64, color='gray', linestyle='--', alpha=0.5, label='d_resid=64')
    ax.set_xlabel('|S|')
    ax.set_ylabel('MSE')
    ax.set_title('MSE in Probability Space (lower is better)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Boundary KL
    ax = axes[1, 0]
    ax.plot(n_states_vals, mse_bkl, 'o-', color='#4292c6', label='MSE Probe', linewidth=2)
    ax.plot(n_states_vals, kl_bkl, 's-', color='#e6550d', label='KL Probe', linewidth=2)
    ax.axvline(64, color='gray', linestyle='--', alpha=0.5, label='d_resid=64')
    ax.set_xlabel('|S|')
    ax.set_ylabel('Boundary KL')
    ax.set_title('Boundary Fidelity (lower is better)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Pairwise R²
    ax = axes[1, 1]
    ax.plot(n_states_vals, mse_pr2, 'o-', color='#4292c6', label='MSE Probe', linewidth=2)
    ax.plot(n_states_vals, kl_pr2, 's-', color='#e6550d', label='KL Probe', linewidth=2)
    ax.axvline(64, color='gray', linestyle='--', alpha=0.5, label='d_resid=64')
    ax.set_xlabel('|S|')
    ax.set_ylabel('Pairwise R² (Euclidean)')
    ax.set_title('Pairwise Distance Correlation (higher is better)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Genuine Compression: MSE vs KL Probe as |S| Increases\n'
                 '(d_resid = 64 fixed; compression occurs when |S| > 64)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    _save_fig(fig, save_path)
    return fig


# --- Compression Sweep Aggregated Plots (from aggregate_compression_results.py) ---

def _get_best_layer_metrics(summary_entry: dict, metric_path: str):
    """Extract metrics from the best layer (highest KL alignment) for a sweep entry."""
    per_layer = summary_entry['per_layer']
    # Find layer with highest mean KL alignment
    best_layer = max(
        per_layer,
        key=lambda l: per_layer[l].get('raw_geometry', {}).get(
            'kl_alignment', {}).get('mean', -1))
    # Navigate the metric path (e.g. 'raw_geometry.kl_alignment')
    parts = metric_path.split('.')
    val = per_layer[best_layer]
    for p in parts:
        val = val[p]
    return val, best_layer


def plot_money_plot(summary: dict, save_path: str = None):
    """The central figure: KL alignment vs Euclidean alignment across compression ratios.

    Args:
        summary: aggregated sweep summary from aggregate_compression_results.py
    """
    n_states_vals = sorted(summary.keys())
    ratios = [summary[n]['compression_ratio'] for n in n_states_vals]

    euc_means, euc_errs = [], []
    kl_means, kl_errs = [], []

    for n in n_states_vals:
        euc_stats, _ = _get_best_layer_metrics(summary[n], 'raw_geometry.euclidean_alignment')
        kl_stats, _ = _get_best_layer_metrics(summary[n], 'raw_geometry.kl_alignment')
        euc_means.append(euc_stats['mean'])
        euc_errs.append(euc_stats['stderr'])
        kl_means.append(kl_stats['mean'])
        kl_errs.append(kl_stats['stderr'])

    euc_means = np.array(euc_means)
    euc_errs = np.array(euc_errs)
    kl_means = np.array(kl_means)
    kl_errs = np.array(kl_errs)
    ratios = np.array(ratios)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left panel: raw alignment values
    ax1.plot(ratios, euc_means, 'o-', color='#4292c6', linewidth=2,
             label='Euclidean alignment')
    ax1.fill_between(ratios, euc_means - euc_errs, euc_means + euc_errs,
                     color='#4292c6', alpha=0.2)
    ax1.plot(ratios, kl_means, 's-', color='#e6550d', linewidth=2,
             label='KL alignment')
    ax1.fill_between(ratios, kl_means - kl_errs, kl_means + kl_errs,
                     color='#e6550d', alpha=0.2)

    ax1.axvline(1.0, color='gray', linestyle='--', alpha=0.6,
                label='$|S| = d_{\\mathrm{model}}$')
    ax1.set_xscale('log')
    ax1.set_xlabel('Compression ratio ($|S| / d_{\\mathrm{model}}$)')
    ax1.set_ylabel('Spearman $\\rho$ (activation dist. vs belief dist.)')
    ax1.set_title('What geometry does the transformer encode?')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right panel: the gap (KL - Euclidean alignment)
    gap_means = kl_means - euc_means
    gap_errs = np.sqrt(kl_errs**2 + euc_errs**2)

    ax2.plot(ratios, gap_means, 'D-', color='#2ca02c', linewidth=2)
    ax2.fill_between(ratios, gap_means - gap_errs, gap_means + gap_errs,
                     color='#2ca02c', alpha=0.2)
    ax2.axhline(0, color='gray', linestyle='-', alpha=0.4)
    ax2.axvline(1.0, color='gray', linestyle='--', alpha=0.6,
                label='$|S| = d_{\\mathrm{model}}$')
    ax2.set_xscale('log')
    ax2.set_xlabel('Compression ratio ($|S| / d_{\\mathrm{model}}$)')
    ax2.set_ylabel('KL alignment $-$ Euclidean alignment')
    ax2.set_title('Information-geometric advantage')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Probe-Free Geometry: Transformers Prefer Information Geometry '
                 'Under Compression', fontsize=12, y=1.02)
    plt.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_probe_comparison_sweep(summary: dict, save_path: str = None):
    """Supporting evidence: probe-based metrics across compression ratios.

    Compares KL probe's pairwise_rho_kl vs MSE probe's pairwise_r2_euclidean.
    """
    n_states_vals = sorted(summary.keys())
    ratios = [summary[n]['compression_ratio'] for n in n_states_vals]

    mse_euc_means, mse_euc_errs = [], []
    kl_kl_means, kl_kl_errs = [], []

    for n in n_states_vals:
        mse_stats, _ = _get_best_layer_metrics(
            summary[n], 'probe_metrics.mse_pairwise_r2_euclidean')
        kl_stats, _ = _get_best_layer_metrics(
            summary[n], 'probe_metrics.kl_pairwise_rho_kl')
        mse_euc_means.append(mse_stats['mean'])
        mse_euc_errs.append(mse_stats['stderr'])
        kl_kl_means.append(kl_stats['mean'])
        kl_kl_errs.append(kl_stats['stderr'])

    mse_euc_means = np.array(mse_euc_means)
    mse_euc_errs = np.array(mse_euc_errs)
    kl_kl_means = np.array(kl_kl_means)
    kl_kl_errs = np.array(kl_kl_errs)
    ratios = np.array(ratios)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(ratios, mse_euc_means, 'o-', color='#4292c6', linewidth=2,
            label='MSE probe: pairwise R$^2$ (Euclidean)')
    ax.fill_between(ratios, mse_euc_means - mse_euc_errs,
                    mse_euc_means + mse_euc_errs, color='#4292c6', alpha=0.2)
    ax.plot(ratios, kl_kl_means, 's-', color='#e6550d', linewidth=2,
            label='KL probe: pairwise $\\rho$ (KL)')
    ax.fill_between(ratios, kl_kl_means - kl_kl_errs,
                    kl_kl_means + kl_kl_errs, color='#e6550d', alpha=0.2)

    ax.axvline(1.0, color='gray', linestyle='--', alpha=0.6,
               label='$|S| = d_{\\mathrm{model}}$')
    ax.set_xscale('log')
    ax.set_xlabel('Compression ratio ($|S| / d_{\\mathrm{model}}$)')
    ax.set_ylabel('Structure preservation')
    ax.set_title('Probe Comparison: Each Probe with Its Natural Metric')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_convergence_diagnostics(runs: dict, save_path: str = None):
    """Training loss curves for all n_states values with entropy rate baselines."""
    n_states_vals = sorted(runs.keys())
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(n_states_vals)))

    fig, ax = plt.subplots(figsize=(10, 5))

    for idx, n in enumerate(n_states_vals):
        for run in runs[n]:
            losses = run['train_loss']
            if losses:
                ax.plot(range(len(losses)), losses, color=colors[idx],
                        alpha=0.5, linewidth=0.8)
        # Plot entropy rate as horizontal line
        entropy = runs[n][0]['entropy_rate_nats']
        ax.axhline(entropy, color=colors[idx], linestyle='--', alpha=0.6,
                   linewidth=1)
        # Label the last point
        ax.text(len(runs[n][0]['train_loss']) + 1, entropy,
                f'$|S|$={n}', fontsize=7, color=colors[idx], va='center')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss (nats)')
    ax.set_title('Training Convergence (solid=loss, dashed=entropy rate)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_layer_structure(summary: dict, selected_n_states: list = None,
                         save_path: str = None):
    """Layer-wise KL vs Euclidean alignment for selected n_states values."""
    if selected_n_states is None:
        all_n = sorted(summary.keys())
        # Pick ~4 representative values spanning the range
        if len(all_n) <= 4:
            selected_n_states = all_n
        else:
            indices = np.linspace(0, len(all_n) - 1, 4, dtype=int)
            selected_n_states = [all_n[i] for i in indices]

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(selected_n_states)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for idx, n in enumerate(selected_n_states):
        if n not in summary:
            continue
        per_layer = summary[n]['per_layer']
        layer_keys = _get_ordered_layer_keys(per_layer)

        euc_vals = []
        kl_vals = []
        for lk in layer_keys:
            rg = per_layer[lk].get('raw_geometry', {})
            euc_vals.append(rg.get('euclidean_alignment', {}).get('mean', float('nan')))
            kl_vals.append(rg.get('kl_alignment', {}).get('mean', float('nan')))

        ratio = summary[n]['compression_ratio']
        label = f'$|S|$={n} (ratio={ratio:.2f})'

        ax1.plot(range(len(layer_keys)), euc_vals, 'o-', color=colors[idx],
                 label=label)
        ax2.plot(range(len(layer_keys)), kl_vals, 's-', color=colors[idx],
                 label=label)

    for ax, title in [(ax1, 'Euclidean alignment'), (ax2, 'KL alignment')]:
        if layer_keys:
            ax.set_xticks(range(len(layer_keys)))
            ax.set_xticklabels([k.replace('_', '\n') for k in layer_keys],
                               fontsize=7)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Spearman $\\rho$')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Layer-Wise Geometry Alignment', fontsize=12, y=1.02)
    plt.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_effective_dimensionality(summary: dict, save_path: str = None):
    """Effective rank vs compression ratio."""
    n_states_vals = sorted(summary.keys())
    ratios = [summary[n]['compression_ratio'] for n in n_states_vals]

    eff_rank_means = []
    eff_rank_errs = []

    for n in n_states_vals:
        stats, _ = _get_best_layer_metrics(summary[n], 'effective_rank')
        eff_rank_means.append(stats['mean'])
        eff_rank_errs.append(stats['stderr'])

    eff_rank_means = np.array(eff_rank_means)
    eff_rank_errs = np.array(eff_rank_errs)
    ratios = np.array(ratios)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(ratios, eff_rank_means, 'o-', color='#756bb1', linewidth=2)
    ax.fill_between(ratios, eff_rank_means - eff_rank_errs,
                    eff_rank_means + eff_rank_errs, color='#756bb1', alpha=0.2)

    ax.axvline(1.0, color='gray', linestyle='--', alpha=0.6,
               label='$|S| = d_{\\mathrm{model}}$')
    ax.axhline(64, color='gray', linestyle=':', alpha=0.4,
               label='$d_{\\mathrm{model}} = 64$')

    ax.set_xscale('log')
    ax.set_xlabel('Compression ratio ($|S| / d_{\\mathrm{model}}$)')
    ax.set_ylabel('Effective rank')
    ax.set_title('How Much of the Residual Stream Does the Model Use?')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_fig(fig, save_path)
    return fig
