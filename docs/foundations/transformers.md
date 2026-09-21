# Transformer Architecture

*Under construction - see `guide/04-transformer/README.md` for implementation guide and experiments.*

## Key Concepts

- Pre-Norm transformer block
- Residual connections
- MLP (GeLU, 4x expansion)
- LayerNorm
- LM head with weight tying
- Autoregressive generation

## Experiments

- EXP-011: Pre-Norm vs Post-Norm
- EXP-012: Model scaling laws
- EXP-013: Weight tying effect
- EXP-014: Gradient checkpointing