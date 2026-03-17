import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .mess3 import Mess3Process
from .config import ExperimentConfig


class NonErgodicMess3Dataset(Dataset):
    """Dataset of sequences from a non-ergodic mixture of Mess3 processes.

    Each sequence is generated entirely by one component. The component
    label is stored for analysis but NOT provided to the model.
    """

    def __init__(self, processes: list[Mess3Process], n_sequences: int,
                 seq_length: int, rng: np.random.Generator):
        n_per_component = n_sequences // len(processes)
        all_tokens = []
        all_hidden = []
        all_labels = []

        for k, proc in enumerate(processes):
            tokens, hidden = proc.generate_sequences(n_per_component, seq_length, rng)
            all_tokens.append(tokens)
            all_hidden.append(hidden)
            all_labels.append(np.full(n_per_component, k, dtype=np.int64))

        tokens = np.concatenate(all_tokens, axis=0)
        hidden = np.concatenate(all_hidden, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        # Shuffle
        perm = rng.permutation(len(tokens))
        self.tokens = torch.from_numpy(tokens[perm])
        self.hidden_states = torch.from_numpy(hidden[perm])
        self.component_labels = torch.from_numpy(labels[perm])

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        return {
            'input_tokens': tokens[:-1],     # (L-1,)
            'target_tokens': tokens[1:],     # (L-1,)
            'component_label': self.component_labels[idx],
            'hidden_states': self.hidden_states[idx],
            'full_tokens': tokens,           # (L,) for belief computation
        }


class GeneralHMMDataset(Dataset):
    """Dataset of sequences from a single ergodic HMM with arbitrary n_states."""

    def __init__(self, hmm, n_sequences: int, seq_length: int,
                 rng: np.random.Generator):
        tokens, hidden_states = hmm.generate_sequences(n_sequences, seq_length, rng)
        perm = rng.permutation(n_sequences)
        self.tokens = torch.from_numpy(tokens[perm])
        self.hidden_states = torch.from_numpy(hidden_states[perm])
        self.n_states = hmm.n_states
        self.n_tokens = hmm.n_tokens

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        return {
            'input_tokens': tokens[:-1],
            'target_tokens': tokens[1:],
            'hidden_states': self.hidden_states[idx],
            'full_tokens': tokens,
        }


def create_dataloaders(config: ExperimentConfig):
    """Create train and eval dataloaders from config."""
    processes = [
        Mess3Process(c['alpha'], c['x']) for c in config.components
    ]

    rng_train = np.random.default_rng(config.seed)
    rng_eval = np.random.default_rng(config.seed + 1)

    train_dataset = NonErgodicMess3Dataset(
        processes, config.n_train_sequences, config.seq_length, rng_train)
    eval_dataset = NonErgodicMess3Dataset(
        processes, config.n_eval_sequences, config.seq_length, rng_eval)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=0)
    eval_loader = DataLoader(
        eval_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=0)

    return train_loader, eval_loader, processes
