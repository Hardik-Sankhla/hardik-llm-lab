# Methodology

## The Loop

Every major component in this project follows:

```
Learn → Rebuild → Verify → Experiment → Break → Explain → Extend
```

## Learn

- Read the reference material (Sebastian Raschka's book/code)
- Understand the mathematical formulation
- Identify the key design decisions

## Rebuild

- Close the reference
- Implement from mathematical understanding
- Use only PyTorch primitives (nn.Linear, nn.LayerNorm, F.softmax, etc.)
- No peeking at reference implementation

## Verify

- Write unit tests for shapes, gradients, numerical correctness
- Compare against reference implementation on same inputs
- Benchmark speed and memory

## Experiment

- Design controlled experiments with clear hypotheses
- Define independent/dependent variables
- Measure multiple metrics
- Run multiple seeds for statistical significance

## Break

- Push configurations to failure
- Remove components entirely
- Use extreme hyperparameters
- Document what breaks and why

## Explain

- Write findings in documentation
- Create visualizations
- Link to experiment results
- Note limitations

## Extend

- Go beyond the curriculum
- Add new features
- Integrate with other tools
- Build on top of the foundation

## Experiment Design Principles

1. **One variable at a time** - Controlled comparisons
2. **Fixed compute budget** - Fair comparisons
3. **Multiple metrics** - Not just loss
4. **Reproducible configs** - Every experiment has a YAML config
5. **Track everything** - MLflow for params, metrics, artifacts
6. **Document failures** - Failed experiments are data