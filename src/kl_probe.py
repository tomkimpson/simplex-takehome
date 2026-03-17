"""KL-divergence (softmax-affine) probe for belief state recovery.

The probe maps residual stream activations to belief states via:
    b_hat = softmax(W @ a + c)
trained by minimizing forward KL divergence D_KL(b_true || b_hat).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression


class SoftmaxAffineProbe(nn.Module):
    """Softmax-affine probe: b_hat = softmax(W @ a + c).

    Parameters:
        W: (n_states, d_resid) weight matrix
        c: (n_states,) bias vector
    """

    def __init__(self, d_resid: int, n_states: int):
        super().__init__()
        self.linear = nn.Linear(d_resid, n_states)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """Return log-probabilities (log-softmax of affine transform)."""
        logits = self.linear(activations)
        return F.log_softmax(logits, dim=-1)

    def predict_probs(self, activations: torch.Tensor) -> torch.Tensor:
        """Return probabilities (softmax of affine transform)."""
        logits = self.linear(activations)
        return F.softmax(logits, dim=-1)


def initialize_from_mse(probe: SoftmaxAffineProbe,
                        activations: np.ndarray,
                        beliefs: np.ndarray,
                        eps: float = 1e-8) -> None:
    """Initialize KL probe weights from MSE linear regression solution.

    Fits LinearRegression(activations -> beliefs), copies weights/bias
    into the probe as a warm-start. The MSE weights are in probability space,
    not logit space, but provide a much better starting point than random init.
    """
    reg = LinearRegression()
    reg.fit(activations, beliefs)

    with torch.no_grad():
        probe.linear.weight.copy_(torch.from_numpy(reg.coef_).float())
        probe.linear.bias.copy_(torch.from_numpy(reg.intercept_).float())


def train_kl_probe(activations: np.ndarray, beliefs: np.ndarray,
                   n_states: int = 3,
                   lr: float = 1e-3, n_epochs: int = 1000,
                   batch_size: int = 4096,
                   warm_start: bool = True,
                   eps: float = 1e-8,
                   device: str = 'cpu',
                   verbose: bool = False) -> tuple:
    """Train a softmax-affine probe via KL divergence minimization.

    Loss: -sum_j b_j * log_softmax(logits)_j  (forward KL / cross-entropy)

    Args:
        activations: (N, d_resid) activation vectors
        beliefs: (N, n_states) target belief distributions
        n_states: number of hidden states
        lr: learning rate for Adam
        n_epochs: number of training epochs
        batch_size: mini-batch size
        warm_start: initialize from MSE solution
        eps: smoothing constant for target beliefs
        device: torch device string
        verbose: print loss periodically

    Returns:
        probe: trained SoftmaxAffineProbe
        history: dict with 'loss' list
    """
    N, D = activations.shape
    probe = SoftmaxAffineProbe(D, n_states).to(device)

    if warm_start:
        initialize_from_mse(probe, activations, beliefs, eps=eps)
        probe = probe.to(device)

    X = torch.from_numpy(activations).float().to(device)
    B = torch.from_numpy(beliefs).float().to(device)
    B = B.clamp(min=eps)
    B = B / B.sum(dim=-1, keepdim=True)

    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    history = {'loss': []}

    for epoch in range(n_epochs):
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            x_batch = X[idx]
            b_batch = B[idx]

            log_probs = probe(x_batch)
            loss = -(b_batch * log_probs).sum(dim=-1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        history['loss'].append(avg_loss)

        if verbose and (epoch % 100 == 0 or epoch == n_epochs - 1):
            print(f"  KL probe epoch {epoch:4d}/{n_epochs}: loss={avg_loss:.6f}")

    probe.eval()
    return probe, history


def predict_beliefs_kl(probe: SoftmaxAffineProbe,
                       activations: np.ndarray,
                       device: str = 'cpu',
                       batch_size: int = 4096) -> np.ndarray:
    """Generate belief predictions from a trained KL probe.

    Returns:
        predictions: (N, n_states) predicted belief distributions
    """
    probe.eval()
    predictions = []
    X = torch.from_numpy(activations).float().to(device)

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = X[start:start + batch_size]
            probs = probe.predict_probs(batch)
            predictions.append(probs.cpu().numpy())

    return np.concatenate(predictions, axis=0)
