import numpy as np


class Mess3Process:
    """Mess3 Hidden Markov Model with 3 hidden states and 3 observable tokens.

    Parameterized by (alpha, x) where:
        beta = (1 - alpha) / 2
        y = 1 - 2x

    The token-labeled transition matrices T^(k) encode the joint probability
    of emitting token k and transitioning to a new state.
    """

    def __init__(self, alpha: float, x: float):
        self.alpha = alpha
        self.x = x
        self.beta = (1 - alpha) / 2
        self.y = 1 - 2 * x
        self.n_states = 3
        self.n_tokens = 3

        # Build token-labeled transition matrices T^(0), T^(1), T^(2)
        # T^(k)_{ij} = P(emit k, go to j | in state i)
        self.T = self._build_transition_matrices()

        # Total transition matrix: T_total = sum_k T^(k)
        self.T_total = self.T.sum(axis=0)

        # Emission matrix: E[i, k] = P(emit k | state i) = sum_j T^(k)_{ij}
        self.E = np.array([self.T[k].sum(axis=1) for k in range(3)]).T

    def _build_transition_matrices(self):
        """Build the three token-labeled transition matrices.

        T^(k) is constructed by cyclic permutation: token k is the
        'informative' token for state k (appears with probability proportional
        to alpha), while other tokens appear with probability proportional to beta.
        """
        a, b = self.alpha, self.beta
        xv, yv = self.x, self.y

        # Base matrix for token 0: state 0 is the 'matching' state
        T0 = np.array([
            [a * yv, b * xv, b * xv],
            [a * xv, b * yv, b * xv],
            [a * xv, b * xv, b * yv],
        ])

        # Token 1: state 1 is the 'matching' state (cyclic shift of columns)
        T1 = np.array([
            [b * yv, a * xv, b * xv],
            [b * xv, a * yv, b * xv],
            [b * xv, a * xv, b * yv],
        ])

        # Token 2: state 2 is the 'matching' state
        T2 = np.array([
            [b * yv, b * xv, a * xv],
            [b * xv, b * yv, a * xv],
            [b * xv, b * xv, a * yv],
        ])

        return np.stack([T0, T1, T2], axis=0)  # shape (3, 3, 3)

    def entropy_rate(self) -> float:
        """Compute the entropy rate H of the process.

        H = -sum_s pi(s) sum_k E[s,k] log2(E[s,k])

        Since the stationary distribution is uniform, this simplifies to
        the average entropy of the emission distribution.
        """
        # Stationary distribution is uniform [1/3, 1/3, 1/3]
        H = 0.0
        for s in range(self.n_states):
            for k in range(self.n_tokens):
                p = self.E[s, k]
                if p > 0:
                    H -= (1.0 / 3.0) * p * np.log2(p)
        return H

    def generate_sequences(self, n_sequences: int, seq_length: int,
                           rng: np.random.Generator) -> tuple:
        """Generate sequences in batch.

        Args:
            n_sequences: number of sequences to generate
            seq_length: length of each sequence
            rng: numpy random generator

        Returns:
            tokens: int array of shape (n_sequences, seq_length)
            hidden_states: int array of shape (n_sequences, seq_length)
        """
        tokens = np.zeros((n_sequences, seq_length), dtype=np.int64)
        hidden_states = np.zeros((n_sequences, seq_length), dtype=np.int64)

        # Initialize hidden states from stationary (uniform) distribution
        states = rng.integers(0, self.n_states, size=n_sequences)

        for t in range(seq_length):
            hidden_states[:, t] = states

            # Emit tokens: P(token | state) from emission matrix
            # E[state, :] gives emission probs for each sequence
            emission_probs = self.E[states]  # (n_sequences, 3)
            # Sample tokens
            cumprobs = emission_probs.cumsum(axis=1)
            u = rng.random(n_sequences)[:, None]
            tokens[:, t] = (u >= cumprobs).sum(axis=1)

            # Transition: P(next_state | current_state, emitted_token)
            # For each sequence, get T^(token)[state, :] and normalize
            emitted = tokens[:, t]
            # Gather transition rows: T[token, state, :]
            trans_probs = self.T[emitted, states, :]  # (n_sequences, 3)
            # Normalize (rows should already sum to emission prob, so divide)
            row_sums = trans_probs.sum(axis=1, keepdims=True)
            trans_probs = trans_probs / row_sums

            # Sample next states
            cumprobs = trans_probs.cumsum(axis=1)
            u = rng.random(n_sequences)[:, None]
            states = (u >= cumprobs).sum(axis=1)

        return tokens, hidden_states

    def validate(self):
        """Run sanity checks on the transition matrices."""
        # Each T^(k) row should sum to E[s, k]
        for k in range(3):
            row_sums = self.T[k].sum(axis=1)
            expected = self.E[:, k]
            assert np.allclose(row_sums, expected), \
                f"T^({k}) row sums {row_sums} != emission probs {expected}"

        # Total transition matrix rows should sum to 1
        total_row_sums = self.T_total.sum(axis=1)
        assert np.allclose(total_row_sums, 1.0), \
            f"Total transition matrix row sums {total_row_sums} != 1"

        # Stationary distribution should be uniform
        eigenvalues, eigenvectors = np.linalg.eig(self.T_total.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = np.real(eigenvectors[:, idx])
        pi = pi / pi.sum()
        assert np.allclose(pi, 1.0 / 3.0), \
            f"Stationary distribution {pi} is not uniform"

        return True
