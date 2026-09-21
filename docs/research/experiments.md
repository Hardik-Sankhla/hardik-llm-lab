# Experiment Registry

*Master index of all experiments. Update as experiments are planned, running, and completed.*

## Status Legend

- ⏳ **Planned** - Designed, not started
- 🔄 **Running** - In progress
- ✅ **Completed** - Results documented
- ❌ **Failed** - Documented with lessons

## Foundations

| ID | Question | Variable | Metrics | Status | Results Link |
|----|----------|----------|---------|--------|--------------|
| EXP-001 | Vocab size vs compression/efficiency | vocab_size | compression_ratio, seq_len, val_ppl | ⏳ Planned | - |
| EXP-002 | Custom BPE vs tiktoken | implementation | token_agreement, speed, memory | ⏳ Planned | - |
| EXP-003 | Special tokens impact | special_tokens | boundary_loss, generation_quality | ⏳ Planned | - |
| EXP-004 | Sinusoidal vs learned PE | pos_encoding | val_loss, extrapolation | ⏳ Planned | - |
| EXP-005 | Embedding dim vs quality | d_model | val_loss, params, throughput | ⏳ Planned | - |
| EXP-006 | Weight tying effect | tie_weights | val_ppl, params | ⏳ Planned | - |

## Attention

| ID | Question | Variable | Metrics | Status | Results Link |
|----|----------|----------|---------|--------|--------------|
| EXP-007 | Head count vs quality (fixed params) | n_heads | val_loss, throughput, memory | ⏳ Planned | - |
| EXP-008 | Causal vs full attention | attention_type | val_ppl, bidirectional_task | ⏳ Planned | - |
| EXP-009 | Manual vs PyTorch SDPA | implementation | speed, memory, numerical_diff | ⏳ Planned | - |
| EXP-010 | Attention vs MLP-only | architecture | val_loss, in_context_learning | ⏳ Planned | - |

## Transformer

| ID | Question | Variable | Metrics | Status | Results Link |
|----|----------|----------|---------|--------|--------------|
| EXP-011 | Pre-Norm vs Post-Norm | norm_position | gradient_norm, max_depth | ⏳ Planned | - |
| EXP-012 | Model scaling laws | model_size | val_ppl, params, throughput | ⏳ Planned | - |
| EXP-013 | Weight tying effect | tie_weights | val_ppl | ⏳ Planned | - |
| EXP-014 | Gradient checkpointing | checkpoint_freq | memory, speed | ⏳ Planned | - |

## Training

| ID | Question | Variable | Metrics | Status | Results Link |
|----|----------|----------|---------|--------|--------------|
| EXP-015 | LR schedule comparison | scheduler | val_loss, stability | ⏳ Planned | - |
| EXP-016 | Mixed precision effect | precision | speed, memory, loss | ⏳ Planned | - |
| EXP-017 | Gradient accumulation | accum_steps | speed, memory, loss | ⏳ Planned | - |
| EXP-018 | Weight decay values | weight_decay | val_loss, overfitting_gap | ⏳ Planned | - |

## Fine-tuning

| ID | Question | Variable | Metrics | Status | Results Link |
|----|----------|----------|---------|--------|--------------|
| EXP-019 | Freezing strategy | freeze_strategy | accuracy, forgetting | ⏳ Planned | - |
| EXP-020 | Instruction data scaling | dataset_size | quality, perplexity | ⏳ Planned | - |
| EXP-021 | Classification vs instruction | task_type | respective_metrics | ⏳ Planned | - |

## LoRA

| ID | Question | Variable | Metrics | Status | Results Link |
|----|----------|----------|---------|--------|--------------|
| EXP-022 | LoRA rank ablation | rank | val_loss, params, time | ⏳ Planned | - |
| EXP-023 | LoRA vs full FT | method | val_loss, forgetting | ⏳ Planned | - |
| EXP-024 | LoRA target modules | target_modules | val_loss, params | ⏳ Planned | - |
| EXP-025 | Alpha scaling | alpha | stability, loss | ⏳ Planned | - |

## Evaluation

| ID | Question | Variable | Metrics | Status | Results Link |
|----|----------|----------|---------|--------|--------------|
| EXP-026 | Perplexity vs generation quality | checkpoint | ppl, human_score, distinct-n | ⏳ Planned | - |
| EXP-027 | Sampling params (temp, top-k) | temp, top_k | quality, diversity, repetition | ⏳ Planned | - |
| EXP-028 | Few-shot evaluation | n_shots | accuracy | ⏳ Planned | - |

---

## How to Update

1. When planning: Add row with ⏳ Planned
2. When starting: Change to 🔄 Running, note start date
3. When complete: Change to ✅ Completed, link results file
4. If failed: Change to ❌ Failed, document in `research/failures.md`

Results files: `experiments/results/EXP-XXX.json`
Plots: `experiments/plots/EXP-XXX/`