"""Attention mechanisms.

Implement in guide/03-attention/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def scaled_dot_product_attention(
    q: torch.Tensor,      # [B, H, T, D_head]
    k: torch.Tensor,      # [B, H, T, D_head]
    v: torch.Tensor,      # [B, H, T, D_head]
    mask: torch.Tensor = None,  # [B, 1, T, T] or [T, T]
    dropout: float = 0.0,
    training: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled dot-product attention.
    
    Returns: (output, attention_weights)
    output: [B, H, T, D_head]
    attention_weights: [B, H, T, T]
    """
    raise NotImplementedError("Implement in guide/03-attention/")


class MultiHeadAttention(nn.Module):
    """Multi-head attention with combined QKV projection."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, bias: bool = False):
        super().__init__()
        raise NotImplementedError("Implement in guide/03-attention/")
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError("Implement in guide/03-attention/")


class CausalSelfAttention(MultiHeadAttention):
    """Multi-head attention with built-in causal mask."""
    
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int,
                 dropout: float = 0.1, bias: bool = False):
        super().__init__(d_model, n_heads, dropout, bias)
        raise NotImplementedError("Implement in guide/03-attention/")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement in guide/03-attention/")