"""Model configuration."""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    """GPT model configuration."""
    vocab_size: int = 50257
    max_seq_len: int = 1024
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    bias: bool = False
    tie_weights: bool = True
    layer_norm_eps: float = 1e-5