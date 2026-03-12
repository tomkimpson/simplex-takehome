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

    # Reproducibility
    seed: int = 42

    @property
    def d_head(self):
        return self.d_model // self.n_heads
