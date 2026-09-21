# Benchmarks

*Standard benchmarks for evaluating language model quality.*

## Benchmarks to Run

| Benchmark | Task Type | Metric | Dataset Size |
|-----------|-----------|--------|--------------|
| Lambada | Next-word prediction | Accuracy | 5,153 |
| HellaSwag | Commonsense reasoning | Accuracy | 10,042 |
| PIQA | Physical reasoning | Accuracy | 1,838 |
| WinoGrande | Winograd schemas | Accuracy | 1,267 |
| ARC-Easy | Science QA | Accuracy | 2,376 |
| ARC-Challenge | Hard science QA | Accuracy | 1,172 |
| BoolQ | Boolean QA | Accuracy | 3,270 |
| CB | CommitmentBank | Accuracy/F1 | 56 |
| COPA | Causal reasoning | Accuracy | 100 |
| MultiRC | Multi-sentence RC | F1/EM | 1,800 |
| ReCoRD | Reading comprehension | F1/EM | 1,800 |
| RTE | Textual entailment | Accuracy | 277 |
| WiC | Word sense disambiguation | Accuracy | 638 |
| WSC | Winograd schema | Accuracy | 104 |

## Evaluation Protocol

1. **Zero-shot**: Direct prompt, no examples
2. **Few-shot**: 1, 3, 5 examples in context
3. **Metrics**: Accuracy for classification, F1/EM for QA

## Results Template

```markdown
## Benchmark Results - [Model Name/Checkpoint]

| Benchmark | Zero-shot | 1-shot | 3-shot | 5-shot |
|-----------|-----------|--------|--------|--------|
| Lambada | 42.1% | 45.3% | 47.8% | 48.2% |
| HellaSwag | 31.5% | 34.2% | 36.1% | 37.0% |
| ... | ... | ... | ... | ... |

**Model**: [Config]
**Checkpoint**: [Step/Path]
**Date**: [YYYY-MM-DD]
```

## Implementation Notes

- Use `lm-evaluation-harness` for standardized evaluation
- Cache datasets locally
- Run on same hardware for comparability
- Report confidence intervals (multiple seeds)