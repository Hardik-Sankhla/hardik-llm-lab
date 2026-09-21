# Lessons Learned

*Consolidated insights from the entire journey.*

---

## Architecture Insights

### 1. Pre-Norm > Post-Norm for Deep Models
Pre-Norm (LayerNorm inside residual) allows training much deeper models without gradient explosion/vanishing. Post-Norm (original Transformer) requires careful LR warmup and still struggles past ~12 layers.

**Evidence**: EXP-011 showed Pre-Norm stable at 24 layers, Post-Norm diverged at 16.

### 2. Weight Tying is Free Lunch
Tying input/output embeddings saves ~30% params (for vocab=50k, d_model=768: 76M params) with no quality loss and sometimes slight improvement.

**Evidence**: EXP-006/013 showed identical or better perplexity with tying.

### 3. Learned Positional Embeddings > Sinusoidal for Fixed Length
For models trained at fixed max length, learned positional embeddings consistently outperform sinusoidal. Sinusoidal only wins for extrapolation beyond trained length.

**Evidence**: EXP-004 confirmed learned better within trained range.

### 4. Attention Head Count: Diminishing Returns
Under fixed parameter budget, more heads (smaller d_head) helps up to a point, then hurts. Optimal around 8-12 heads for 768 d_model.

**Evidence**: EXP-007 showed U-shaped curve: 4 heads < 8 heads = 12 heads > 16 heads.

---

## Training Insights

### 5. Gradient Clipping is Non-Negotiable
Without gradient clipping (1.0), training inevitably hits gradient explosion in transformer models, especially with FP16.

### 6. Mixed Precision (BF16) = Free Speedup
On Ampere+ GPUs, BF16 gives ~1.5x speedup with no quality loss. FP16 requires GradScaler and can be unstable.

### 7. Cosine Decay with Warmup is Standard for a Reason
Constant LR → overfitting. Linear decay → suboptimal. Cosine with warmup consistently best across experiments.

**Evidence**: EXP-015 compared 5 schedules, cosine+ warmup won.

### 8. Weight Decay 0.1 is Strong Default
For GPT-style models, weight decay of 0.1 (with proper param grouping: no decay on bias/LN/embeddings) consistently prevents overfitting.

### 9. Gradient Accumulation Works Exactly Like Large Batch
`accum=4, batch=8` produces identical results to `batch=32` (within numerical precision). Verified in EXP-017.

---

## Fine-tuning Insights

### 10. LoRA Rank 4-8 is Often Sufficient
For instruction fine-tuning on small datasets (<10k examples), rank 4-8 matches full fine-tuning with 1000x fewer trainable params.

**Evidence**: EXP-022 showed plateau after rank 8.

### 11. Freezing Backbone Prevents Catastrophic Forgetting
Full fine-tuning on small datasets destroys pretrained knowledge. Freezing all but last 2 layers + head retains 95%+ pretrain perplexity.

**Evidence**: EXP-019 measured pretrain perplexity before/after FT.

### 12. Instruction Tuning Needs Quality > Quantity
1000 high-quality diverse instructions > 10000 low-quality repetitive ones. Data curation matters more than volume.

**Evidence**: EXP-020 showed quality plateau after ~2k good examples.

---

## Experimentation Insights

### 13. Failed Experiments Are More Valuable Than Successful Ones
Knowing *what doesn't work* and *why* builds deeper understanding than confirming what works.

**Example**: Removing LayerNorm entirely → complete training failure. Taught us LN is essential for gradient flow.

### 14. Always Measure Multiple Metrics
Loss alone is misleading. Track: throughput (tokens/sec), memory, gradient norm, validation perplexity, generation quality.

### 15. Reproducibility Requires Explicit Seeding
`torch.manual_seed` alone is insufficient. Need: Python random, NumPy, CUDA deterministic, cuDNN benchmark off.

### 16. Config Hash = Experiment Identity
Hash the full config (model + training + data) to uniquely identify experiments. Prevents "which config was this run?" confusion.

---

## Engineering Insights

### 17. Tests Catch 80% of Bugs Before Training
Shape tests, gradient tests, numerical comparison tests catch most implementation bugs before you waste GPU hours.

### 18. Visualization Reveals What Numbers Hide
Attention heatmaps, loss curves, gradient norm plots reveal patterns invisible in scalar metrics.

### 19. Checkpoint Everything, Resume Exactly
Save: model, optimizer, scheduler, scaler, step, RNG states. Resume should be bit-identical.

### 20. Documentation Written During > Documentation Written After
Writing `EXPERIMENT_LOG.md` during the experiment captures reasoning that's lost afterward.

---

## Meta-Lessons

### 21. Depth > Breadth
10 deeply understood components > 50 superficially implemented techniques.

### 22. "From Scratch" Means Understanding, Not Reimplementation
Using `nn.Linear` is fine. Not understanding what `nn.Linear` does is not.

### 23. The Reference is a Curriculum, Not a Target
Goal: understand well enough to implement independently, not to replicate exactly.

### 24. Portfolio Value = Process Documentation
The experiments, failures, and reasoning are more valuable than the final model weights.

### 25. Compute Constraints Force Better Science
Limited GPU forces you to design efficient, informative experiments rather than brute-force scaling.

---

*This document grows as the project evolves. Each lesson should reference the experiment that taught it.*