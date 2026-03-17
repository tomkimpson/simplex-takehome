from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    # Mess3 component parameters: (alpha, x)
    # A: sparse fractal, highly informative tokens
    # B: medium fractal, moderate informativeness
    # C: dense fractal, weakly informative tokens
    components: list = field(default_factory=lambda: [
        {"name": "A", "alpha": 0.90, "x": 0.05},
        {"name": "B", "alpha": 0.70, "x": 0.10},
        {"name": "C", "alpha": 0.55, "x": 0.05},
    ])

    # Data generation
    seq_length: int = 16
    n_train_sequences: int = 300_000   # 100k per component
    n_eval_sequences: int = 30_000     # 10k per component
    n_analysis_sequences: int = 10_000

    # Architecture
    vocab_size: int = 3
    d_model: int = 64
    n_layers: int = 3
    n_heads: int = 4
    d_mlp: int = 256
    dropout: float = 0.0

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    n_epochs: int = 200
    eval_every: int = 10

    # Analysis
    pca_components: int = 6

    # KL probe hyperparameters
    kl_probe_lr: float = 1e-3
    kl_probe_epochs: int = 1000
    kl_probe_batch_size: int = 4096
    kl_probe_warm_start: bool = True

    # PCA compression sweep (Experiment 2)
    pca_sweep_k_values: list = field(default_factory=lambda: [1, 2, 3, 4, 8, 16, 32, 64])
    pca_sweep_layer: str = 'layer_2'

    # Reproducibility
    seed: int = 42

    @property
    def d_head(self):
        return self.d_model // self.n_heads


@dataclass
class CompressionExperimentConfig:
    """Config for Experiment 3: Genuine Compression Regime."""
    # HMM parameters
    n_states: int = 50
    n_tokens: int = 3
    hmm_sparsity: float = 0.5
    hmm_dirichlet_alpha: float = 0.3

    # Data generation
    seq_length: int = 32
    n_train_sequences: int = 300_000
    n_eval_sequences: int = 30_000

    # Architecture (d_model=64 is fixed — the whole point of the experiment)
    vocab_size: int = 3
    d_model: int = 64
    n_layers: int = 3
    n_heads: int = 4
    d_mlp: int = 256
    dropout: float = 0.0

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    n_epochs: int = 200
    eval_every: int = 10

    # KL probe
    kl_probe_lr: float = 1e-3
    kl_probe_epochs: int = 1000

    # Sweep
    n_states_sweep: list = field(default_factory=lambda: [10, 20, 50, 100, 200])
    n_analysis: int = 5000

    # Reproducibility
    seed: int = 42

    @property
    def d_head(self):
        return self.d_model // self.n_heads
