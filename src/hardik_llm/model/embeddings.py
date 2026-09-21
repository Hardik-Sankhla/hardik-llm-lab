"""Model components: embeddings, blocks, config, GPT model.

Implement in guide/02-embeddings/ and guide/04-transformer/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass, field


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    max_seq_len: int = 1024
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    dropout: float = 0.1
    qkv_bias: bool = False
    tie_weights: bool = True
    pad_token_id: int = 50257
    
    d_head: int = field(init=False)
    
    def __post_init__(self):
        self.d_head = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"


class TokenEmbedding(nn.Module):
    """Token embeddings with optional padding index."""
    
    def __init__(self, vocab_size: int, d_model: int, pad_token_id: int = None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, T]
        # output: [B, T, D]
        return self.embedding(input_ids)


class SinusoidalPositionalEmbedding(nn.Module):
    """Fixed sinusoidal positional encodings."""
    
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, D]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        return x + self.pe[:, :x.size(1)]


class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embeddings (GPT-style)."""
    
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pe.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)  # [1, T]
        return x + self.pe(positions)


class GPTEmbeddings(nn.Module):
    """Combined token + positional embeddings."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.token_emb = TokenEmbedding(config.vocab_size, config.d_model, config.pad_token_id)
        self.pos_emb = LearnedPositionalEmbedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(input_ids)
        x = self.pos_emb(x)
        return self.dropout(x)