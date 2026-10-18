# Phase 2: Transformer Architecture — Chapter 3

> **Book**: Ch 3 "Transformer Anatomy" (pp. 81–108)  
> **Core**: Build transformer from components — encoder, decoder, self-attention, FFN, layer norm, positional embeddings

## Book Implementation Checklist

| Section | Book Task | Key Code | Extension Ideas |
|---------|-----------|----------|-----------------|
| 3.1 | Transformer architecture overview | `nn.MultiheadAttention`, `nn.Transformer` | Custom implementation from scratch |
| 3.2 | Encoder block | Self-attention + FFN + LN + Residual | Pre-LN vs Post-LN |
| 3.3 | Self-attention math | `QK^T/√d` → softmax → `V` | Multi-head, masked, cross-attention |
| 3.4 | Feed-forward layer | `Linear → GELU → Linear` | GLU, SwiGLU, MoE |
| 3.5 | Layer normalization | `nn.LayerNorm` | RMSNorm, ScaleNorm |
| 3.6 | Positional embeddings | Sinusoidal vs learned vs RoPE | ALiBi, T5 relative bias |
| 3.7 | Classification head | `[CLS]` token + Linear | Pooling strategies |
| 3.8 | Decoder block | Causal mask + cross-attention | GPT vs Llama vs T5 decoder |
| 3.9 | Transformer variants | BERT, GPT, T5, encoder-decoder | Model family tree |

## Recommended Experiments

| Exp ID | Variant | Implementation | Target |
|--------|---------|----------------|--------|
| PH2-001 | Minimal encoder | From-scratch `nn.Module` | Match BERT-base params |
| PH2-002 | Minimal decoder | From-scratch causal LM | Match GPT-2 small params |
| PH2-003 | Attention ablation | No attention / no FFN / no LN | Component importance |
| PH2-004 | Positional embedding | Sinusoidal vs Learned vs RoPE | Convergence, extrapolation |
| PH2-005 | Normalization | LayerNorm vs RMSNorm vs None | Training stability |
| PH2-006 | Model variants | BERT / GPT / T5 from config | Architecture comparison |

## Quick Start

```bash
# Build encoder from scratch (Ch 3.2)
python -m phase2_architecture.run --component encoder --exp-id PH2-001

# Build decoder from scratch (Ch 3.8)
python -m phase2_architecture.run --component decoder --exp-id PH2-002

# Self-attention visualization (Ch 3.3)
python -m phase2_architecture.run --demo attention --exp-id PH2-003

# Positional embedding comparison (Ch 3.6)
python -m phase2_architecture.run --demo pos-embed --exp-id PH2-004

# Train tiny transformer (Ch 3.9)
python -m phase2_architecture.run --train tiny --exp-id PH2-006
```

## Key Book Code Patterns to Reproduce

```python
# 1. Self-attention from scratch (Ch 3.3)
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch, seq, _ = x.shape
        q = self.q_proj(x).view(batch, seq, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.d_model)
        return self.out_proj(out)

# 2. Encoder block (Ch 3.2)
class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = SelfAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-LN (modern) vs Post-LN (original)
        x = x + self.dropout(self.attention(self.ln1(x), mask))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x

# 3. Positional embeddings (Ch 3.6)
def sinusoidal_pos_embed(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# RoPE (modern, not in book)
def apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

# 4. Full encoder stack
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(sinusoidal_pos_embed(max_seq_len, d_model))
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.embedding(x) + self.pos_embed[:x.size(1)]
        for layer in self.layers:
            x = layer(x, mask)
        return self.ln_f(x)
```

## Architecture Comparison (Book Ch 3.7)

| Model | Type | Layers | Heads | d_model | d_ff | Params | Key Feature |
|-------|------|--------|-------|---------|------|--------|-------------|
| **BERT-base** | Encoder | 12 | 12 | 768 | 3072 | 110M | Bidirectional, [CLS] |
| **BERT-large** | Encoder | 24 | 16 | 1024 | 4096 | 340M | Deeper, wider |
| **GPT-2 small** | Decoder | 12 | 12 | 768 | 3072 | 124M | Causal, no [CLS] |
| **GPT-2 medium** | Decoder | 24 | 16 | 1024 | 4096 | 355M | Deeper, wider |
| **T5-base** | Enc-Dec | 12/12 | 12 | 768 | 3072 | 220M | Encoder-decoder |
| **T5-large** | Enc-Dec | 24/24 | 16 | 1024 | 4096 | 770M | Deeper, wider |

## Modern Architecture Improvements (Beyond Book)

| Improvement | Models | Benefit |
|-------------|--------|---------|
| **RoPE** | Llama, PaLM, GPT-NeoX | Better extrapolation |
| **SwiGLU** | Llama, PaLM, Gemma | Better FFN |
| **RMSNorm** | Llama, Gemma | Faster, stable |
| **Grouped Query Attention** | Llama-2-70B, Gemma | Faster inference |
| **Sliding Window** | Mistral, Longformer | Long context |
| **ALiBi** | BLOOM, MPT | No positional embeddings |

## Next Phase Dependencies

- Phase 1 (Classification): Uses encoder architecture
- Phase 3 (NER): Uses encoder with token classification head
- Phase 4 (Generation): Uses decoder architecture
- Phase 5 (Summarization): Uses encoder-decoder
- Phase 9 (From Scratch): Scales these components
- Tracks/PEFT: Injects adapters into these blocks