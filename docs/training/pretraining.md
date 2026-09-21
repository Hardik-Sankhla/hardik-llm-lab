# Pretraining Pipeline

*Under construction - see `guide/05-pretraining/README.md` for implementation guide and experiments.*

## Key Concepts

- Data loading & tokenization
- AdamW optimizer with param groups
- Cosine decay with warmup
- Mixed precision (FP16/BF16)
- Gradient accumulation
- Gradient clipping
- Checkpointing & resuming

## Experiments

- EXP-015: Learning rate schedule comparison
- EXP-016: Mixed precision effect
- EXP-017: Gradient accumulation vs large batch
- EXP-018: Weight decay values