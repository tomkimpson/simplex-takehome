import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int,
                 dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Causal mask
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer('mask', mask)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scale = math.sqrt(self.d_head)
        attn = (q @ k.transpose(-2, -1)) / scale  # (B, H, L, L)
        attn = attn.masked_fill(self.mask[:L, :L], float('-inf'))
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v  # (B, H, L, d_head)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out), attn_weights


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_mlp: int,
                 max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_out, attn_weights = self.attn(self.ln1(x))
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, attn_weights


class Mess3Transformer(nn.Module):
    """Small causal decoder-only transformer for Mess3 next-token prediction.

    Exposes residual stream activations at every layer for geometry analysis.
    """

    def __init__(self, vocab_size: int, d_model: int, n_layers: int,
                 n_heads: int, d_mlp: int, max_seq_len: int,
                 dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_mlp, max_seq_len, dropout)
            for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, x, return_activations=False):
        """Forward pass.

        Args:
            x: input token indices, shape (B, L)
            return_activations: if True, also return dict of residual stream
                activations and attention weights at each layer

        Returns:
            logits: shape (B, L, vocab_size)
            activations: dict (only if return_activations=True)
                'embedding': (B, L, d_model) - post-embedding
                'layer_0', ..., 'layer_{n-1}': (B, L, d_model) - post each block
                'final': (B, L, d_model) - post final layernorm
                'attn_weights_0', ...: (B, H, L, L) - attention weights
        """
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0)

        h = self.token_emb(x) + self.pos_emb(positions)

        activations = {}
        if return_activations:
            activations['embedding'] = h.detach()

        for i, block in enumerate(self.blocks):
            h, attn_w = block(h)
            if return_activations:
                activations[f'layer_{i}'] = h.detach()
                activations[f'attn_weights_{i}'] = attn_w.detach()

        h = self.ln_final(h)
        if return_activations:
            activations['final'] = h.detach()

        logits = self.head(h)

        if return_activations:
            return logits, activations
        return logits

    @classmethod
    def from_config(cls, config):
        return cls(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_mlp=config.d_mlp,
            max_seq_len=config.seq_length,
            dropout=config.dropout,
        )
