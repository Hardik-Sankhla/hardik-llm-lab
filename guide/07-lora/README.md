# Guide: LoRA / PEFT (Appendix E)

**Objective**: Implement Low-Rank Adaptation from scratch. Understand the math, run rank ablations, compare with full fine-tuning.

---

## Prerequisites

- Read: `reference/raschka/appendix-E/01_main-chapter-code/appendix-E.ipynb`
- Key concepts: Low-rank decomposition, adapter modules, rank, alpha scaling, target modules

---

## Your Implementation Checklist

### 1. LoRA Linear Layer (`src/hardik_llm/finetuning/lora.py`)

```python
class LoRALinear(nn.Module):
    """Low-Rank Adaptation applied to a linear layer."""
    def __init__(self, base_layer: nn.Linear, rank: int = 4, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Freeze base layer
        for param in base_layer.parameters():
            param.requires_grad = False
        
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        # LoRA adapters: A (in_features x rank), B (rank x out_features)
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Initialize A with normal, B with zeros (so initial output = base only)
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base output
        base_out = self.base_layer(x)
        
        # LoRA path: x @ A @ B * scaling
        # x: [B, T, D] or [B, D]
        lora_out = (self.dropout(x) @ self.lora_A @ self.lora_B) * self.scaling
        
        return base_out + lora_out
    
    def merge_weights(self):
        """Merge LoRA into base layer for inference (no runtime overhead)."""
        with torch.no_grad():
            self.base_layer.weight += (self.lora_B @ self.lora_A.T).T * self.scaling
        # Remove LoRA params
        del self.lora_A, self.lora_B
```

### 2. LoRA Config & Injection (`src/hardik_llm/finetuning/lora.py`)

```python
@dataclass
class LoRAConfig:
    rank: int = 4
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: [
        "qkv_proj", "out_proj", "fc1", "fc2"  # Attention + MLP
    ])

def inject_lora(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """Replace target linear layers with LoRALinear."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this module should get LoRA
            if any(target in name for target in config.target_modules):
                parent = get_parent_module(model, name)
                child_name = name.split('.')[-1]
                setattr(parent, child_name, LoRALinear(module, config.rank, config.alpha, config.dropout))
    return model

def get_parent_module(model: nn.Module, name: str) -> nn.Module:
    parts = name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent
```

### 3. LoRA Utilities

```python
def count_lora_params(model: nn.Module) -> tuple[int, int]:
    """Returns (lora_params, total_params)"""
    lora_params = sum(p.numel() for n, p in model.named_parameters() 
                      if 'lora_' in n and p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return lora_params, total_params

def save_lora_weights(model: nn.Module, path: str):
    """Save only LoRA parameters (tiny checkpoint)."""
    lora_state = {n: p for n, p in model.named_parameters() if 'lora_' in n}
    torch.save(lora_state, path)

def load_lora_weights(model: nn.Module, path: str):
    """Load LoRA parameters into model."""
    lora_state = torch.load(path)
    model.load_state_dict(lora_state, strict=False)
```

---

## Experiments to Run (Guide/07-lora/)

### EXP-022: LoRA Rank Ablation

**Question**: What's the minimal rank for effective adaptation?

**Variables**:
- Rank: [1, 2, 4, 8, 16, 32, 64]
- Alpha: rank * 4 (standard scaling)
- Task: Instruction fine-tuning on small dataset (1k examples)
- Base: Pretrained 124M model

**Metrics**:
- Validation loss
- Generation quality (qualitative)
- Trainable params
- Training time

**Expected**: Rank 4-8 often sufficient for small tasks

---

### EXP-023: LoRA vs Full Fine-tuning

**Question**: How close does LoRA get to full FT?

**Variables**:
- LoRA (rank=8, alpha=32) on all linear layers
- Full FT (all params)
- Same dataset, same steps

**Metrics**:
- Final val loss
- Pretrain perplexity retention (forgetting)
- Training time
- Memory usage

---

### EXP-024: Target Module Selection

**Question**: Which layers benefit most from LoRA?

**Variables**:
- Attention only (qkv_proj, out_proj)
- MLP only (fc1, fc2)
- All linear layers
- No LoRA (baseline)

**Metrics**: Val loss, params

---

### EXP-025: Alpha Scaling

**Question**: How does alpha affect training?

**Variables**: Alpha = [rank, rank*2, rank*4, rank*8, rank*16] with rank=8

**Metrics**: Training stability, final loss

---

## What to Observe & Note

### During Implementation
- [ ] LoRA init: A=normal, B=zeros → initial output = base model exactly
- [ ] Scaling: `alpha / rank` - why this formula?
- [ ] Dropout on LoRA path only, not base
- [ ] Merge weights for inference: removes runtime overhead

### During Experiments
- [ ] Plot rank vs val_loss (expect plateau after rank 8)
- [ ] Plot rank vs trainable_params (linear)
- [ ] Compare LoRA vs full FT forgetting (measure pretrain perplexity)
- [ ] Note: LoRA rank 1 might still work for very simple tasks

### Failure Modes
- [ ] Rank > min(in_features, out_features) → error
- [ ] Alpha too large → instability
- [ ] Target modules wrong → no trainable params
- [ ] Forgetting to freeze base layer → double training

---

## Validation Tests (`tests/test_lora.py`)

```python
def test_lora_initial_output_equals_base():
    """Before training, LoRA output == base output exactly."""
    pass

def test_lora_rank_parameter_count():
    """Rank r adds r*(in+out) params per layer."""
    pass

def test_merge_weights():
    """Merged weights produce identical output to unmerged."""
    pass

def test_lora_save_load():
    """Save/load preserves LoRA params only."""
    pass

def test_lora_freeze_base():
    """Base layer params frozen, only LoRA trainable."""
    pass

def test_inject_lora_target_modules():
    """Only specified modules get LoRA."""
    pass
```

---

## Documentation to Write

1. `docs/adaptation/lora.md` - LoRA math, implementation, best practices
2. `guide/07-lora/EXPERIMENT_LOG.md`
3. `experiments/configs/EXP-022-lora-rank.yaml`, etc.
4. Update `docs/research/experiments.md`

---

## Next Step

After LoRA: `guide/08-evaluation/`