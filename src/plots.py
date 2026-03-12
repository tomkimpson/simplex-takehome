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
