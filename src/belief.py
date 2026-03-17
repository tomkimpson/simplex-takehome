import numpy as np
from .mess3 import Mess3Process


def compute_oracle_beliefs(token_sequences: np.ndarray,
                           component_labels: np.ndarray,
                           processes: list[Mess3Process]) -> np.ndarray:
    """Compute Bayesian belief states assuming the component is known.

    For each sequence, given its component label and observed tokens,
    compute the posterior over hidden states at each position.

    Args:
        token_sequences: int array (N, L) of observed tokens
        component_labels: int array (N,) identifying which Mess3 component
        processes: list of Mess3Process instances (one per component)

    Returns:
        beliefs: float array (N, L, 3) where beliefs[i, t, :] is the
                 belief distribution over hidden states for sequence i
                 at position t (AFTER observing tokens 0..t-1, i.e.
                 the predictive belief at position t)
    """
    N, L = token_sequences.shape
    beliefs = np.zeros((N, L, 3))

    # Uniform prior at position 0 (before any tokens observed)
    beliefs[:, 0, :] = 1.0 / 3.0

    for t in range(1, L):
        token = token_sequences[:, t - 1]  # token observed at previous position

        for k, proc in enumerate(processes):
            mask = component_labels == k
            if not mask.any():
                continue

            # eta_{t} = eta_{t-1} @ T^(x_{t-1}) / normalizer
            eta_prev = beliefs[mask, t - 1, :]  # (n_k, 3)
            tok = token[mask]  # (n_k,)

            # Batch matrix multiply: each sequence uses T^(tok[i])
            # T[tok] has shape (n_k, 3, 3)
            T_tok = proc.T[tok]  # (n_k, 3, 3)

            # eta_new[i] = eta_prev[i] @ T_tok[i]
            eta_new = np.einsum('ij,ijk->ik', eta_prev, T_tok)

            # Normalize
            normalizer = eta_new.sum(axis=1, keepdims=True)
            eta_new = eta_new / normalizer

            beliefs[mask, t, :] = eta_new

    return beliefs


def compute_mixture_beliefs(token_sequences: np.ndarray,
                            processes: list[Mess3Process],
                            component_priors: np.ndarray = None) -> tuple:
    """Compute beliefs when the component identity is unknown.

    Maintains a joint posterior over (component, hidden_state) pairs.

    Args:
        token_sequences: int array (N, L)
        processes: list of K Mess3Process instances
        component_priors: float array (K,) prior over components.
                         Defaults to uniform.

    Returns:
        joint_beliefs: float array (N, L, K, 3) - full joint posterior
        component_posteriors: float array (N, L, K) - marginal P(component | history)
    """
    N, L = token_sequences.shape
    K = len(processes)

    if component_priors is None:
        component_priors = np.ones(K) / K

    # Joint belief: P(component=k, state=s | history)
    joint = np.zeros((N, L, K, 3))

    # Initialize: prior_k * uniform over states
    for k in range(K):
        joint[:, 0, k, :] = component_priors[k] / 3.0

    for t in range(1, L):
        token = token_sequences[:, t - 1]  # (N,)

        for k, proc in enumerate(processes):
            # Previous joint belief for component k
            eta_prev = joint[:, t - 1, k, :]  # (N, 3)

            # Update: eta_new = eta_prev @ T^(token)
            T_tok = proc.T[token]  # (N, 3, 3)
            eta_new = np.einsum('ij,ijk->ik', eta_prev, T_tok)

            joint[:, t, k, :] = eta_new

        # Normalize across all (component, state) pairs
        normalizer = joint[:, t, :, :].sum(axis=(1, 2), keepdims=True)
        joint[:, t, :, :] = joint[:, t, :, :] / normalizer.reshape(N, 1, 1)

    # Marginal component posteriors
    component_posteriors = joint.sum(axis=3)  # (N, L, K)

    return joint, component_posteriors


def compute_beliefs_general(token_sequences: np.ndarray, hmm) -> np.ndarray:
    """Compute Bayesian belief states for a single general HMM.

    Args:
        token_sequences: int array (N, L) of observed tokens
        hmm: a GeneralHMM instance (or any object with .T, .n_states, .stationary)

    Returns:
        beliefs: float array (N, L, n_states) where beliefs[i, t, :] is the
                 belief distribution over hidden states for sequence i
                 at position t (AFTER observing tokens 0..t-1)
    """
    N, L = token_sequences.shape
    S = hmm.n_states
    beliefs = np.zeros((N, L, S))

    # Initialize from stationary distribution
    beliefs[:, 0, :] = hmm.stationary[None, :]

    for t in range(1, L):
        token = token_sequences[:, t - 1]  # token observed at previous position
        eta_prev = beliefs[:, t - 1, :]  # (N, S)

        # T[tok] has shape (N, S, S) — one transition matrix per sequence
        T_tok = hmm.T[token]  # (N, S, S)

        # eta_new[i] = eta_prev[i] @ T_tok[i]
        eta_new = np.einsum('ij,ijk->ik', eta_prev, T_tok)

        # Normalize
        normalizer = eta_new.sum(axis=1, keepdims=True)
        normalizer = np.where(normalizer > 0, normalizer, 1.0)
        eta_new = eta_new / normalizer

        beliefs[:, t, :] = eta_new

    return beliefs
