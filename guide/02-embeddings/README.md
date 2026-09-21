# Guide: Embeddings & Positional Encoding (Chapter 2-3)

**Objective**: Understand how tokens become vectors. Implement token embeddings, positional embeddings, and experiment with alternatives.

---

## Prerequisites

- Read: `reference/raschka/ch02/01_main-chapter-code/ch02.ipynb` (embedding section)
- Read: `reference/raschka/ch03/01_main-chapter-code/ch03.ipynb` (positional encoding)
- Key concepts: embedding lookup, positional encoding (sinusoidal vs learned), embedding dimension

---

## Your Implementation Checklist

### 1. Token Embeddings (`src/hardik_llm/model/embeddings.py`)

```python
import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, pad_token_id: int = None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        # Initialize properly
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, T]
        # output: [B, T, D]
        return self.embedding(input_ids)
```

### 2. Positional Embeddings - Two Variants

```python
class SinusoidalPositionalEmbedding(nn.Module):
    """Fixed sinusoidal positional encodings (Vaswani et al.)"""
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        # Create positional encoding matrix
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
    """Learned positional embeddings (GPT-style)"""
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pe.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)  # [1, T]
        return x + self.pe(positions)
```

### 3. Combined Embedding Module

```python
class GPTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_emb = TokenEmbedding(config.vocab_size, config.d_model, config.pad_token_id)
        self.pos_emb = LearnedPositionalEmbedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(input_ids)
        x = self.pos_emb(x)
        return self.dropout(x)
```

---

## Experiments to Run (Guide/02-embeddings/)

### EXP-004: Sinusoidal vs Learned Positional Embeddings

**Question**: Does learned or fixed positional encoding work better for small models?

**Variables**:
- Sinusoidal (fixed) vs Learned (trainable)
- Same model config otherwise

**Metrics**:
- Training loss curve
- Validation perplexity
- Convergence speed (steps to reach loss X)
- Generalization to longer sequences (extrapolation)

**Procedure**:
1. Train two identical small models (124M config, short run)
2. One with sinusoidal, one with learned
3. Compare loss curves
4. Test extrapolation: generate with context > trained max_len

**Expected**: Learned typically better for trained lengths; sinusoidal extrapolates better

---

### EXP-005: Embedding Dimension vs Model Quality

**Question**: How does d_model affect model capacity under fixed parameter budget?

**Variables**:
- d_model: [128, 256, 384, 512, 768]
- Adjust n_layers to keep ~constant params
- Fixed: total params ≈ 10M

**Metrics**:
- Validation loss
- Parameters
- Training throughput (tokens/sec)
- Memory usage

---

### EXP-006: Weight Tying (Input/Output Embeddings)

**Question**: Does tying input and output embeddings help?

**Variables**:
- Tied: `lm_head.weight = token_emb.embedding.weight`
- Untied: Separate parameters

**Metrics**:
- Validation perplexity
- Parameter count
- Training stability

**Note**: This is a classic GPT design choice - test it!

---

## What to Observe & Note

### During Implementation
- [ ] Embedding gradient flow: `embedding.weight.grad` shape?
- [ ] Positional embedding gradients - do they get large?
- [ ] Dropout on embeddings - where exactly to apply?
- [ ] Padding token - should it get gradient updates?

### During Experiments
- [ ] Plot loss curves for sinusoidal vs learned
- [ ] Test sequence length extrapolation (generate at 2x trained length)
- [ ] Measure parameter savings from weight tying
- [ ] Note: Does weight tying hurt or help at small scale?

### Failure Modes
- [ ] Positional embedding out of range (seq_len > max_len)
- [ ] Embedding dimension not divisible by n_heads
- [ ] Padding token affecting loss (should be masked)

---

## Validation Tests (`tests/test_embeddings.py`)

```python
def test_token_embedding_shape():
    """TokenEmbedding: [B, T] -> [B, T, D]"""
    pass

def test_sinusoidal_deterministic():
    """Sinusoidal PE is deterministic, no gradients."""
    pass

def test_learned_pe_gradients():
    """Learned PE receives gradients."""
    pass

def test_weight_tying():
    """Tied embeddings share exact same tensor."""
    pass

def test_embedding_dropout():
    """Dropout applied to combined embeddings."""
    pass

def test_padding_idx_no_grad():
    """Padding token embedding receives no gradient."""
    pass
```

---

## Documentation to Write

1. `docs/foundations/embeddings.md` - Embedding types, positional encoding math
2. `guide/02-embeddings/EXPERIMENT_LOG.md` - Running log
3. `experiments/configs/EXP-004-pos-encoding.yaml`
4. Update `docs/research/experiments.md`

---

## Next Step

After embeddings: `guide/03-attention/`