# Optimization Details

*Under construction - see `guide/05-pretraining/README.md` for implementation guide and experiments.*

## Key Concepts

- AdamW with proper weight decay grouping
- Learning rate schedules (cosine, linear, constant)
- Mixed precision training
- Gradient accumulation
- Gradient clipping strategies

## Experiments

- EXP-015: LR schedule comparison
- EXP-016: Mixed precision effect
- EXP-017: Gradient accumulation
- EXP-018: Weight decay values