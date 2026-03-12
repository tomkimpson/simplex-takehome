import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path

from .transformer import Mess3Transformer
from .dataset import create_dataloaders
from .config import ExperimentConfig


def evaluate(model, eval_loader, device, n_components=3):
    """Evaluate model, returning overall and per-component loss."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    component_loss = np.zeros(n_components)
    component_tokens = np.zeros(n_components)

    with torch.no_grad():
        for batch in eval_loader:
            inputs = batch['input_tokens'].to(device)
            targets = batch['target_tokens'].to(device)
            labels = batch['component_label'].numpy()

            logits = model(inputs)
            # Per-sequence loss
            B, L, V = logits.shape
            loss_per_token = F.cross_entropy(
                logits.reshape(-1, V), targets.reshape(-1), reduction='none'
            ).reshape(B, L)
            loss_per_seq = loss_per_token.mean(dim=1).cpu().numpy()

            total_loss += loss_per_token.sum().item()
            total_tokens += B * L

            for k in range(n_components):
                mask = labels == k
                if mask.any():
                    component_loss[k] += loss_per_seq[mask].sum()
                    component_tokens[k] += mask.sum()

    avg_loss = total_loss / total_tokens
    component_avg = component_loss / np.maximum(component_tokens, 1)
    model.train()
    return avg_loss, component_avg


def train(config: ExperimentConfig = None, device_str: str = None,
          resume_from: str = None):
    """Main training loop."""
    if config is None:
        config = ExperimentConfig()

    # Device selection
    if device_str:
        device = torch.device(device_str)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    # Data
    train_loader, eval_loader, processes = create_dataloaders(config)
    print(f'Train: {len(train_loader.dataset)} sequences, '
          f'Eval: {len(eval_loader.dataset)} sequences')

    # Theoretical entropy rates (in nats for comparison with cross-entropy)
    entropy_rates_bits = [p.entropy_rate() for p in processes]
    entropy_rates_nats = [h * np.log(2) for h in entropy_rates_bits]
    for i, (bits, nats) in enumerate(zip(entropy_rates_bits, entropy_rates_nats)):
        name = config.components[i]['name']
        print(f'Component {name}: H = {bits:.4f} bits = {nats:.4f} nats')

    # Model
    model = Mess3Transformer.from_config(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {n_params:,}')

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Resume from checkpoint
    start_epoch = 0
    if resume_from:
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f'Resumed from {resume_from}, starting at epoch {start_epoch}')

    # Logging
    log_dir = Path('results/logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path('results/checkpoints')
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history = {
        'train_loss': [],
        'eval_loss': [],
        'component_loss': [],
        'entropy_rates_nats': entropy_rates_nats,
    }

    # Training
    for epoch in range(start_epoch, config.n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            inputs = batch['input_tokens'].to(device)
            targets = batch['target_tokens'].to(device)

            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, config.vocab_size), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches
        history['train_loss'].append(avg_train_loss)

        # Evaluate periodically
        if epoch % config.eval_every == 0 or epoch == config.n_epochs - 1:
            eval_loss, comp_loss = evaluate(model, eval_loader, device)
            history['eval_loss'].append((epoch, eval_loss))
            history['component_loss'].append((epoch, comp_loss.tolist()))

            comp_str = '  '.join(
                f'{config.components[i]["name"]}={comp_loss[i]:.4f}'
                for i in range(len(processes)))
            print(f'Epoch {epoch:3d}/{config.n_epochs}  '
                  f'train={avg_train_loss:.4f}  eval={eval_loss:.4f}  '
                  f'[{comp_str}]')

            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
            }, ckpt_dir / f'checkpoint_epoch_{epoch:04d}.pt')

    # Save final model and history
    torch.save({
        'epoch': config.n_epochs - 1,
        'model_state_dict': model.state_dict(),
        'config': config,
    }, ckpt_dir / 'model_final.pt')

    with open(log_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f'\nTraining complete. Final eval loss: {history["eval_loss"][-1][1]:.4f}')
    return model, history


if __name__ == '__main__':
    train()
