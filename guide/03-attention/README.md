# Guide: Attention Mechanisms (Chapter 3)

**Objective**: Implement self-attention from matrix multiplication up. Understand every tensor shape. Run ablations on head count, causal masking, attention variants.

---

## Prerequisites

- Read: `reference/raschka/ch03/01_main-chapter-code/ch03.ipynb`
- Reference: `reference/raschka/ch03/01_main-chapter-code/multihead-attention.ipynb`
- Key concepts: Q, K, V projections, scaled dot-product, causal mask, multi-head

---

## Your Implementation Checklist

### 1. Scaled Dot-Product Attention (`src/hardik_llm/attention/scaled_dot_product.py`)

```python
import torch
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
    Returns: (output, attention_weights)
    output: [B, H, T, D_head]
    attention_weights: [B, H, T, T]
    """
    d_head = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_head)  # [B, H, T, T]
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attn_weights = F.softmax(scores, dim=-1)
    attn_weights = F.dropout(attn_weights, p=dropout, training=training)
    
    output = torch.matmul(attn_weights, v)
    return output, attn_weights
```

### 2. Multi-Head Attention (`src/hardik_llm/attention/multi_head.py`)

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, bias: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Combined QKV projection (more efficient)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = dropout
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: [B, T, D]
        B, T, D = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # [B, T, 3D]
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, D_head]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        out, attn_weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.dropout, training=self.training
        )
        
        # Combine heads
        out = out.transpose(1, 2).reshape(B, T, D)  # [B, T, D]
        out = self.out_proj(out)
        return out
```

### 3. Causal Self-Attention (`src/hardik_llm/attention/causal.py`)

```python
class CausalSelfAttention(MultiHeadAttention):
    """Multi-head attention with causal mask built-in"""
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, 
                 dropout: float = 0.1, bias: bool = False):
        super().__init__(d_model, n_heads, dropout, bias)
        # Pre-compute causal mask
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer('causal_mask', mask.view(1, 1, max_seq_len, max_seq_len))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        mask = self.causal_mask[:, :, :T, :T]
        return super().forward(x, mask=mask)
```

---

## Experiments to Run (Guide/03-attention/)

### EXP-007: Head Count vs Quality (Fixed Parameter Budget)

**Question**: Under fixed total params, is it better to have more heads (smaller d_head) or fewer heads (larger d_head)?

**Variables**:
- Config: d_model=384, vary n_heads=[2, 3, 4, 6, 8, 12]
- Adjust n_layers to keep ~constant params
- Fixed: total params ≈ 10M

**Metrics**:
- Validation loss
- Training throughput (tokens/sec)
- Memory usage
- Attention pattern diversity (entropy of attn weights)

**Procedure**:
1. For each head count, compute required n_layers for ~10M params
2. Train for fixed steps (e.g., 5000)
3. Compare validation loss curves

---

### EXP-008: Causal vs Full Attention

**Question**: How much does causal masking hurt modeling capacity?

**Variables**:
- Causal (GPT) vs Full (BERT-style) attention
- Same model config

**Metrics**:
- Validation perplexity
- Training loss
- Can full attention model learn bidirectional dependencies?

**Procedure**:
1. Train two models on same data
2. Test on next-token prediction (causal task)
3. Test on masked prediction (bidirectional task)

---

### EXP-009: Attention Implementation Variants

**Question**: Manual vs PyTorch `F.scaled_dot_product_attention` vs Flash Attention

**Implement**:
1. Your manual `scaled_dot_product_attention`
2. PyTorch `F.scaled_dot_product_attention` (with `is_causal=True`)
3. Compare: forward output, gradients, speed, memory

**Metrics**:
- Numerical agreement (max abs diff)
- Forward/backward time
- Peak memory

---

### EXP-010: Removing Attention (MLP-only Baseline)

**Question**: How much does attention actually contribute?

**Variables**:
- Standard GPT block (Attention + MLP)
- MLP-only block (replace attention with residual + MLP)
- Same parameter count (increase MLP width)

**Metrics**:
- Validation loss
- In-context learning ability (few-shot)

---

## What to Observe & Note

### During Implementation
- [ ] Tensor shapes at every step: Q/K/V → scores → weights → output
- [ ] How does `transpose(-2, -1)` work for K?
- [ ] Why `masked_fill(mask == 0, -inf)` before softmax?
- [ ] Gradient flow: `q.grad`, `k.grad`, `v.grad` shapes?
- [ ] Combined QKV projection vs separate - memory/speed difference?

### During Experiments
- [ ] Plot n_heads vs val_loss (expect U-shape: too few = bottleneck, too many = noise)
- [ ] Visualize attention heads: do different heads learn different patterns?
- [ ] Causal vs full: does full attention overfit on next-token task?
- [ ] Manual vs PyTorch SDPA: numerical differences? (should be ~1e-6)

### Failure Modes
- [ ] `d_model % n_heads != 0` → clear error
- [ ] Causal mask wrong shape → silent wrong results
- [ ] Dropout on attention weights vs output - different effects
- [ ] Large sequence length → OOM on attention matrix [B, H, T, T]

---

## Validation Tests (`tests/test_attention.py`)

```python
def test_scaled_dot_product_shapes():
    """Q[B,H,T,D], K[B,H,T,D], V[B,H,T,D] -> out[B,H,T,D], weights[B,H,T,T]"""
    pass

def test_causal_mask():
    """Causal mask prevents attending to future positions."""
    pass

def test_multihead_attention_shape():
    """MultiHeadAttention: [B, T, D] -> [B, T, D]"""
    pass

def test_attention_deterministic():
    """Same input + same weights = same output (no dropout)."""
    pass

def test_manual_vs_pytorch_sdpa():
    """Your implementation matches F.scaled_dot_product_attention numerically."""
    pass

def test_causal_self_attention():
    """CausalSelfAttention produces valid causal mask."""
    pass

def test_gradient_flow():
    """Gradients flow through Q, K, V projections."""
    pass
```

---

## Documentation to Write

1. `docs/foundations/attention.md` - Attention math, multi-head, causal masking
2. `guide/03-attention/EXPERIMENT_LOG.md`
3. `experiments/configs/EXP-007-head-count.yaml`, `EXP-008-causal-vs-full.yaml`
4. Update `docs/research/experiments.md`

---

## Visualization to Build

Create attention visualization in `notebooks/foundations/03_attention_visualization.ipynb`:
- Heatmap of attention weights for sample sentence
- Per-head patterns
- Compare causal vs full attention matrices

---

## Next Step

After attention: `guide/04-transformer/`