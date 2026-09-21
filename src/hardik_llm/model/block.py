"""Transformer block and MLP.

Implement in guide/04-transformer/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hardik_llm.attention.scaled_dot_product import CausalSelfAttention
from hardik_llm.model.config import GPTConfig


class MLP(nn.Module):
    """MLP block with GeLU activation (4x expansion)."""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GELU with tanh approximation (matches GPT-2)
        return self.fc2(self.dropout(F.gelu(self.fc1(x), approximate='tanh')))


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer Block."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(
            config.d_model, config.n_heads, config.max_seq_len,
            config.dropout, config.qkv_bias
        )
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config.d_model, config.dropout)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm: LayerNorm -> Attention -> Residual
        x = x + self.dropout(self.attn(self.ln1(x)))
        # Pre-Norm: LayerNorm -> MLP -> Residual
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x