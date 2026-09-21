# Mistakes & Debugging Log

*Documented as they happen - this is where the real learning lives.*

---

## Mistake #001: Assuming Lower Training Loss = Better Model

**Context**: Early pretraining runs, watching training loss drop.

**What Happened**: Training loss kept decreasing, but validation perplexity started increasing after step 5000.

**Why**: Overfitting on small dataset. The model memorized training sequences.

**Fix**: 
- Add weight decay (0.1)
- Increase dropout (0.1 → 0.2)
- Monitor validation loss, not training loss
- Implement early stopping

**Lesson**: Training loss is a measure of optimization, not generalization. Always track validation metrics.

---

## Mistake #002: Forgetting Causal Mask in Attention

**Context**: Implementing multi-head attention from scratch.

**What Happened**: Model could "see the future" - attended to all positions including future tokens.

**Why**: Used `F.scaled_dot_product_attention` without `is_causal=True` and didn't pass manual mask.

**Fix**: 
- Always pass causal mask for autoregressive modeling
- Write test: `assert attn_weights[i, j] == 0 for j > i`
- Verify with gradient check: future tokens should have zero gradient

**Lesson**: Silent correctness bugs are the most dangerous. Test the constraint, not just the output.

---

## Mistake #003: Weight Tying Without Bias Handling

**Context**: Tying input/output embeddings to save parameters.

**What Happened**: `lm_head.weight = token_emb.embedding.weight` worked, but `lm_head.bias` was still created and trained separately.

**Why**: `nn.Linear` creates bias by default. Tying weights doesn't tie bias.

**Fix**: `nn.Linear(..., bias=False)` for LM head when tying weights.

**Lesson**: Parameter sharing requires exact tensor sharing. Check `id(tensor)` to verify.

---

## Mistake #004: Gradient Accumulation Loss Scaling

**Context**: Implementing gradient accumulation for effective larger batch size.

**What Happened**: Loss didn't decrease properly with accumulation steps > 1.

**Why**: Forgot to divide loss by `accumulation_steps` before backward. Gradients were `accum_steps` times too large.

**Fix**: `loss = loss / config.gradient_accumulation_steps` before `backward()`.

**Lesson**: Gradient accumulation requires loss scaling. Verify by comparing `accum=1, batch=32` vs `accum=4, batch=8`.

---

## Mistake #005: Mixed Precision Without Gradient Scaling

**Context**: Enabling FP16 training for speed.

**What Happened**: Gradients underflowed to zero, training stalled.

**Why**: FP16 has smaller dynamic range. Small gradients become zero.

**Fix**: Use `GradScaler`:
```python
scaler = GradScaler()
with autocast():
    loss = model(x, y)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Lesson**: FP16 requires loss scaling. Always use `GradScaler` or switch to BF16 if hardware supports.

---

## Mistake #006: Padding Token Getting Gradients

**Context**: Training with padded sequences.

**What Happened**: Padding token embedding received gradients and drifted.

**Why**: `padding_idx` in `nn.Embedding` only zeroes output, not gradients.

**Fix**: 
- Mask loss for padding tokens: `loss_fn(..., ignore_index=pad_token_id)`
- Or: `embedding.weight.grad[pad_token_id].zero_()` after backward

**Lesson**: `padding_idx` ≠ gradient masking. Handle in loss or explicitly zero gradients.

---

## Mistake #007: LoRA Initialization - Non-Zero B Matrix

**Context**: Implementing LoRA adapters.

**What Happened**: Initial forward pass with LoRA != base model output.

**Why**: Initialized both A and B with normal distribution.

**Fix**: Initialize A ~ N(0, 0.02), B = zeros. This ensures `x @ A @ B = 0` initially.

**Lesson**: LoRA must be identity at initialization. Zero initialization for B is critical.

---

## Mistake #008: Not Setting Seeds for Reproducibility

**Context**: Running experiments, getting different results each run.

**What Happened**: Same config, different validation loss across runs.

**Why**: Forgot to set `torch.manual_seed`, `numpy.random.seed`, `random.seed`, and `torch.cuda.deterministic = True`.

**Fix**: 
```python
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Lesson**: Reproducibility requires explicit seeding everywhere. Make it a config parameter.

---

## Debugging Checklist (When Things Go Wrong)

1. **Loss NaN?**
   - Check gradient clipping
   - Check mixed precision scaling
   - Check LR too high
   - Check weight initialization

2. **Loss Not Decreasing?**
   - Verify data pipeline (are inputs/targets correct?)
   - Check LR schedule (is LR actually changing?)
   - Check optimizer (are params actually updating?)
   - Check model forward pass (shapes correct?)

3. **OOM?**
   - Reduce batch size
   - Enable gradient checkpointing
   - Reduce sequence length
   - Use gradient accumulation

4. **Validation Loss >> Training Loss?**
   - Overfitting: increase dropout, weight decay
   - Data leakage check
   - Model too large for data

5. **Generation Quality Poor?**
   - Check temperature/top-k
   - Check prompt formatting
   - Verify tokenizer roundtrip
   - Check model was actually trained (not random weights)

---

*Add new mistakes as you discover them. Each one is a lesson learned.*