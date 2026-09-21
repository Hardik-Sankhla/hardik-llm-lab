# Guide: Transformer Block & GPT Model (Chapter 4)

**Objective**: Assemble embeddings + attention + MLP into transformer blocks. Build complete GPT model. Implement weight tying, gradient checkpointing, and generation.

---

## Prerequisites

- Read: `reference/raschka/ch04/01_main-chapter-code/ch04.ipynb`
- Reference: `reference/raschka/ch04/01_main-chapter-code/gpt.py`
- Key concepts: Pre-Norm transformer block, residual connections, MLP (GeLU), LM head, generation loop

---

## Your Implementation Checklist

### 1. MLP Block (`src/hardik_llm/model/block.py`)

```python
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)  # Expansion 4x
        self.fc2 = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GELU activation (approximate='tanh' matches GPT-2)
        return self.fc2(self.dropout(F.gelu(self.fc1(x), approximate='tanh')))
```

### 2. Transformer Block (Pre-Norm) (`src/hardik_llm/model/block.py`)

```python
class TransformerBlock(nn.Module):
    def __init__(self, config):
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
```

### 3. GPT Config (`src/hardik_llm/model/config.py`)

```python
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
    
    # Derived
    d_head: int = field(init=False)
    
    def __post_init__(self):
        self.d_head = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
```

### 4. Complete GPT Model (`src/hardik_llm/model/gpt.py`)

```python
class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.embeddings = GPTEmbeddings(config)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.d_model)
        
        # LM Head (tied with token embeddings if config.tie_weights)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        if config.tie_weights:
            self.lm_head.weight = self.embeddings.token_emb.embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None):
        # input_ids: [B, T]
        x = self.embeddings(input_ids)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, T, V]
        
        loss = None
        if targets is not None:
            # Shift for next-token prediction
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        return logits, loss
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, 
                 temperature: float = 1.0, top_k: int = None):
        """Autoregressive generation."""
        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            idx_cond = input_ids[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
```

---

## Experiments to Run (Guide/04-transformer/)

### EXP-011: Pre-Norm vs Post-Norm

**Question**: Does Pre-Norm (used in GPT) train more stably than Post-Norm (original Transformer)?

**Variables**:
- Pre-Norm: LN -> Attention -> Residual
- Post-Norm: Attention -> Residual -> LN

**Metrics**:
- Training loss stability (variance)
- Gradient norm at different depths
- Max trainable depth before divergence

**Procedure**:
1. Implement both variants
2. Train deep models (24+ layers) with same config
3. Compare gradient norms per layer

---

### EXP-012: Model Size Scaling (10M → 124M → 355M)

**Question**: How does loss scale with parameters?

**Configs**:
| Size | d_model | n_layers | n_heads | Params |
|------|---------|----------|---------|--------|
| Tiny | 256 | 4 | 4 | ~5M |
| Small | 384 | 6 | 6 | ~15M |
| Medium | 512 | 8 | 8 | ~40M |
| 124M | 768 | 12 | 12 | ~124M |

**Metrics**:
- Validation perplexity
- Training throughput (tokens/sec/GPU)
- Memory usage

---

### EXP-013: Weight Tying Effect

**Question**: Does tying input/output embeddings help at small scale?

**Variables**: Tied vs Untied (same config)

**Metrics**: Validation perplexity, param count, training stability

---

### EXP-014: Gradient Checkpointing

**Question**: Memory vs compute tradeoff for activation checkpointing?

**Variables**:
- No checkpointing
- Checkpoint every block
- Checkpoint every 2 blocks

**Metrics**:
- Peak memory
- Training time per step
- Max batch size at OOM boundary

---

## What to Observe & Note

### During Implementation
- [ ] Residual stream: does `x + attention(ln(x))` preserve gradient flow?
- [ ] LayerNorm placement: Pre-Norm means LN is *inside* residual path
- [ ] Weight tying: `lm_head.weight is embeddings.token_emb.embedding.weight`
- [ ] Generation: KV-cache not implemented yet - recomputes full attention each step

### During Experiments
- [ ] Plot model size vs validation perplexity (scaling law)
- [ ] Plot depth vs gradient norm (Pre-Norm should be flatter)
- [ ] Measure generation speed (tokens/sec) at different sizes
- [ ] Note: At what size does overfitting appear on small dataset?

### Failure Modes
- [ ] NaN loss → check initialization, gradient clipping
- [ ] OOM → reduce batch size, enable gradient checkpointing
- [ ] Generation repeats → temperature too low, top_k too small
- [ ] Loss not decreasing → check LR, weight decay, data pipeline

---

## Validation Tests (`tests/test_model.py`)

```python
def test_gpt_forward_shape():
    """GPTModel: [B, T] -> logits [B, T, V]"""
    pass

def test_gpt_loss_computation():
    """Loss computed correctly with ignore_index."""
    pass

def test_weight_tying():
    """lm_head.weight is token_emb.embedding.weight (same tensor)."""
    pass

def test_generation_shape():
    """generate: [B, T] -> [B, T + max_new_tokens]"""
    pass

def test_generation_deterministic():
    """Same seed + temp=0 -> same output."""
    pass

def test_transformer_block_residual():
    """Residual connection preserves gradient flow."""
    pass

def test_parameter_count():
    """Config matches expected parameter count."""
    pass
```

---

## Documentation to Write

1. `docs/foundations/transformers.md` - Complete architecture diagram + explanation
2. `guide/04-transformer/EXPERIMENT_LOG.md`
3. `experiments/configs/EXP-011-pre-vs-post-norm.yaml`, `EXP-012-scaling.yaml`
4. Update `docs/research/experiments.md`

---

## Visualization to Build

`notebooks/foundations/04_model_architecture.ipynb`:
- Model diagram with tensor shapes
- Parameter breakdown table
- Generation walkthrough

---

## Next Step

After GPT model: `guide/05-pretraining/`