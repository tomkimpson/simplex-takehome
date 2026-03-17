"""General-purpose HMM with arbitrary number of states and tokens."""

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix


class GeneralHMM:
    """Hidden Markov Model with arbitrary state and token counts.

    Attributes:
        T: (n_tokens, n_states, n_states) token-labeled transition matrices.
           T[x, i, j] = P(emit x, transition to j | state i)
        T_total: (n_states, n_states) total transition matrix (sum over tokens)
        E: (n_states, n_tokens) emission matrix. E[s, x] = P(emit x | state s)
        stationary: (n_states,) stationary distribution
        n_states: int
        n_tokens: int
    """

    def __init__(self, T: np.ndarray, validate: bool = True):
        self.n_tokens, self.n_states, _ = T.shape
        self.T = T.astype(np.float64)
        self.T_total = self.T.sum(axis=0)  # (n_states, n_states)

        # Emission matrix: E[s, x] = sum_j T[x, s, j] = P(emit x | state s)
        self.E = np.zeros((self.n_states, self.n_tokens))
        for x in range(self.n_tokens):
            self.E[:, x] = self.T[x].sum(axis=1)

        self.stationary = self._compute_stationary()
        if validate:
            self._validate()

    def _compute_stationary(self) -> np.ndarray:
        """Compute stationary distribution as left eigenvector of T_total."""
        eigenvalues, eigenvectors = np.linalg.eig(self.T_total.T)
        # Find eigenvector for eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = np.real(eigenvectors[:, idx])
        pi = np.abs(pi)
        pi = pi / pi.sum()
        return pi

    def _validate(self):
        """Check that T defines a valid HMM."""
        # Row sums of T_total should be 1
        row_sums = self.T_total.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-8), \
            f"T_total row sums not 1: {row_sums}"

        # Emission rows should sum to 1
        emission_sums = self.E.sum(axis=1)
        assert np.allclose(emission_sums, 1.0, atol=1e-8), \
            f"Emission row sums not 1: {emission_sums}"

        # All entries non-negative
        assert np.all(self.T >= -1e-10), "Negative entries in T"

        # Stationary distribution should be valid
        assert np.allclose(self.stationary.sum(), 1.0, atol=1e-8), \
            f"Stationary distribution doesn't sum to 1: {self.stationary.sum()}"

    def entropy_rate(self) -> float:
        """Compute entropy rate in nats."""
        H = 0.0
        for s in range(self.n_states):
            for x in range(self.n_tokens):
                p = self.E[s, x]
                if p > 0:
                    H -= self.stationary[s] * p * np.log(p)
        return H

    def generate_sequences(self, n_sequences: int, seq_length: int,
                           rng: np.random.Generator) -> tuple:
        """Generate sequences from the HMM.

        Returns:
            tokens: int64 array (n_sequences, seq_length)
            hidden_states: int64 array (n_sequences, seq_length)
        """
        tokens = np.zeros((n_sequences, seq_length), dtype=np.int64)
        hidden_states = np.zeros((n_sequences, seq_length), dtype=np.int64)

        # Initialize from stationary distribution
        states = rng.choice(self.n_states, size=n_sequences, p=self.stationary)
        hidden_states[:, 0] = states

        # Emit first token
        emission_probs = self.E[states]  # (n_sequences, n_tokens)
        tokens[:, 0] = _sample_categorical(emission_probs, rng)

        for t in range(1, seq_length):
            prev_tokens = tokens[:, t - 1]
            prev_states = hidden_states[:, t - 1]

            # Transition: given state s and emitted token x, next state distribution
            # is T[x, s, :] / E[s, x] (conditional on having emitted x)
            # But actually, generate jointly: P(next_state=j, token=x | state=s) = T[x,s,j]
            # We already emitted the token, so for generation we do:
            # 1. Sample token from E[s, :]
            # 2. Sample next state from T[token, s, :] / sum_j T[token, s, j]

            # Actually, for HMM generation the standard approach:
            # Given current state s:
            #   Sample next state j from T_total[s, :]
            #   Sample token x from P(x | s, j) = T[x, s, j] / T_total[s, j]
            # OR equivalently:
            #   Sample (x, j) jointly from T[:, s, :] (flattened then unflattened)

            # Simplest: sample next state from T_total, then token given (s, j)
            next_states = np.zeros(n_sequences, dtype=np.int64)
            emitted = np.zeros(n_sequences, dtype=np.int64)

            for s in range(self.n_states):
                mask = prev_states == s
                n_s = mask.sum()
                if n_s == 0:
                    continue

                # Sample next state
                next_states[mask] = rng.choice(
                    self.n_states, size=n_s, p=self.T_total[s])

                # Sample token given (current_state=s, next_state=j)
                js = next_states[mask]
                for j in range(self.n_states):
                    j_mask_local = js == j
                    n_sj = j_mask_local.sum()
                    if n_sj == 0:
                        continue
                    # P(token=x | s, j) = T[x, s, j] / T_total[s, j]
                    t_total_sj = self.T_total[s, j]
                    if t_total_sj > 0:
                        token_probs = self.T[:, s, j] / t_total_sj
                    else:
                        token_probs = np.ones(self.n_tokens) / self.n_tokens

                    idx_global = np.where(mask)[0][j_mask_local]
                    emitted[idx_global] = rng.choice(
                        self.n_tokens, size=n_sj, p=token_probs)

            hidden_states[:, t] = next_states
            tokens[:, t] = emitted

        return tokens, hidden_states


def _sample_categorical(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample from categorical distributions (one per row)."""
    n = probs.shape[0]
    cumulative = np.cumsum(probs, axis=1)
    u = rng.random(n)[:, None]
    return (u >= cumulative).sum(axis=1).astype(np.int64)


def random_sparse_hmm(n_states: int, n_tokens: int = 3,
                      sparsity: float = 0.5,
                      dirichlet_alpha: float = 0.3,
                      rng: np.random.Generator = None,
                      max_attempts: int = 100) -> GeneralHMM:
    """Generate a random sparse ergodic HMM.

    Args:
        n_states: number of hidden states
        n_tokens: number of observable tokens
        sparsity: fraction of transition entries to zero out (0 = dense, 0.9 = very sparse)
        dirichlet_alpha: Dirichlet concentration parameter (lower = sparser/peakier)
        rng: random number generator
        max_attempts: max retries for ergodicity

    Returns:
        GeneralHMM instance
    """
    if rng is None:
        rng = np.random.default_rng(42)

    for attempt in range(max_attempts):
        # Step 1: Generate total transition matrix
        T_total = np.zeros((n_states, n_states))
        for i in range(n_states):
            row = rng.dirichlet(np.full(n_states, dirichlet_alpha))

            # Apply sparsity: zero out smallest entries
            if sparsity > 0:
                n_zero = int(sparsity * n_states)
                if n_zero > 0 and n_zero < n_states:
                    sorted_idx = np.argsort(row)
                    row[sorted_idx[:n_zero]] = 0.0

            # Add small self-loop for aperiodicity
            row[i] = max(row[i], 0.01)
            row = row / row.sum()
            T_total[i] = row

        # Step 2: Check ergodicity (strong connectivity)
        adj = (T_total > 1e-10).astype(np.float64)
        n_components, labels = connected_components(
            csr_matrix(adj), directed=True, connection='strong')

        if n_components > 1:
            # Add small bridges between components
            for comp in range(1, n_components):
                i = np.where(labels == comp)[0][0]
                j = np.where(labels == comp - 1)[0][0]
                T_total[i, j] = max(T_total[i, j], 0.01)
                T_total[j, i] = max(T_total[j, i], 0.01)
            # Renormalize
            T_total = T_total / T_total.sum(axis=1, keepdims=True)

        # Step 3: Split T_total across tokens
        T = np.zeros((n_tokens, n_states, n_states))
        for i in range(n_states):
            for j in range(n_states):
                if T_total[i, j] > 1e-15:
                    # Split this transition probability across tokens
                    token_fracs = rng.dirichlet(np.ones(n_tokens))
                    T[:, i, j] = token_fracs * T_total[i, j]

        # Step 4: Verify emission non-degeneracy
        E = np.zeros((n_states, n_tokens))
        for x in range(n_tokens):
            E[:, x] = T[x].sum(axis=1)

        # Check that emissions vary across states
        emission_std = E.std(axis=0).mean()
        if emission_std < 0.01:
            continue  # Too degenerate, retry

        try:
            hmm = GeneralHMM(T, validate=True)
            return hmm
        except AssertionError:
            continue

    raise RuntimeError(f"Failed to generate valid HMM after {max_attempts} attempts")
