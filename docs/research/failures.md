# Failures & Negative Results

*Failed experiments and what they taught us. Failed experiments are data.*

## Failure Template

For each failure, document:

1. **Experiment ID**: Link to experiment
2. **What Failed**: Specific failure mode
3. **Hypothesis**: What we expected
4. **Root Cause**: Why it actually failed
5. **Fix Attempted**: What we tried
6. **Resolution**: Fixed / abandoned / workaround
7. **Lesson**: What this teaches

---

## Documented Failures

| ID | Experiment | Failure | Status |
|----|------------|---------|--------|
| FAIL-001 | EXP-011 (Post-Norm) | Diverged at 16 layers | ✅ Documented |
| FAIL-002 | EXP-014 (No grad clip) | Gradient explosion, NaN | ✅ Documented |
| FAIL-003 | EXP-016 (FP16 no scaler) | Gradients underflow to zero | ✅ Documented |
| FAIL-004 | EXP-019 (Full FT, small data) | Catastrophic forgetting | ✅ Documented |
| FAIL-005 | EXP-022 (LoRA rank=128) | Overfitting, unstable | ⏳ Planned |
| FAIL-006 | ABL-001 (No LayerNorm) | Immediate training failure | ⏳ Planned |
| FAIL-007 | ABL-004 (No Attention) | Model can't learn dependencies | ⏳ Planned |
| FAIL-008 | EXP-015 (Constant LR) | Overfitting, no convergence | ⏳ Planned |

---

## FAIL-001: Post-Norm Divergence

**Experiment**: EXP-011 (Pre-Norm vs Post-Norm)

**What Failed**: Post-Norm transformer diverged at 16 layers. Loss → NaN.

**Hypothesis**: Post-Norm would work with careful LR warmup (original Transformer paper).

**Root Cause**: Gradient explosion in deep Post-Norm. Residual path has no normalization, gradients accumulate exponentially with depth.

**Fix Attempted**: 
- LR warmup (1000 steps) - helped but not enough
- Smaller LR (1e-4) - slower but still diverged
- Gradient clipping (1.0) - delayed but didn't prevent

**Resolution**: Abandoned Post-Norm for deep models. Pre-Norm is standard for GPT-style.

**Lesson**: Pre-Norm is essential for depth > 12. Post-Norm only works for shallow models (≤6 layers) with extensive tuning.

---

## FAIL-002: Gradient Explosion Without Clipping

**Experiment**: EXP-014 (Ablation: No gradient clipping)

**What Failed**: Gradient norms grew to 1000+, then NaN loss at step ~2000.

**Hypothesis**: Maybe clipping isn't needed with proper LR?

**Root Cause**: Transformer attention + residual connections create gradient amplification. Without clipping, any instability compounds.

**Fix Attempted**: None - this is fundamental.

**Resolution**: Gradient clipping (1.0) is non-negotiable for transformers.

**Lesson**: Always clip gradients. Monitor gradient norm per layer.

---

## FAIL-003: FP16 Without Gradient Scaler

**Experiment**: EXP-016 (Mixed precision without scaler)

**What Failed**: Gradients underflowed to zero in FP16. Model stopped learning at step ~500.

**Hypothesis**: FP16 would work automatically on modern GPUs.

**Root Cause**: FP16 min positive normal = 6e-5. Transformer gradients often smaller. They become zero.

**Fix Attempted**: 
- GradScaler - worked perfectly
- BF16 (on Ampere+) - worked without scaler

**Resolution**: Always use GradScaler with FP16. Prefer BF16 if hardware supports.

**Lesson**: FP16 ≠ automatic. Dynamic range matters. BF16 has same exponent as FP32.

---

## FAIL-004: Catastrophic Forgetting in Full Fine-tuning

**Experiment**: EXP-019 (Full fine-tuning on 1k examples)

**What Failed**: After fine-tuning, pretrain perplexity went from 12 → 45. Model forgot language modeling.

**Hypothesis**: Fine-tuning adapts without destroying pretrain knowledge.

**Root Cause**: Small dataset + full parameter update = memorization. Pretrained features overwritten.

**Fix Attempted**: 
- Lower LR (1e-5) - helped slightly
- Freeze backbone, train head only - retained pretrain, but limited adaptation
- LoRA - best balance

**Resolution**: For small datasets: LoRA or partial freezing. Full FT only with large data.

**Lesson**: Parameter count vs dataset size ratio determines forgetting. Track pretrain perplexity after FT.

---

## FAIL-005: LoRA Rank Too High

**Experiment**: EXP-022 (LoRA rank ablation)

**What Failed**: Rank 128 overfitted severely on 1k examples. Val loss higher than rank 8.

**Hypothesis**: More capacity = better adaptation.

**Root Cause**: LoRA rank 128 = ~2M params on 124M model. Too many params for 1k examples.

**Fix Attempted**: Lower rank to 8.

**Resolution**: Rank 4-8 optimal for small data. Rank scales with dataset size.

**Lesson**: LoRA rank is a regularization knob. Higher rank = more capacity = more overfitting risk.

---

*Add failures as they happen. Each failure teaches more than ten successes.*