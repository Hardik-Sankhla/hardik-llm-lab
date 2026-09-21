# Ablation Studies

*Documenting controlled ablations to understand component importance.*

## Ablation Template

For each ablation, document:

1. **Component**: What was removed/changed
2. **Hypothesis**: What we expect to happen
3. **Setup**: Config, dataset, compute
4. **Results**: Metrics comparison
5. **Interpretation**: Why did this happen?
6. **Surprise**: Anything unexpected?

---

## Planned Ablations

| ID | Component | Hypothesis | Status |
|----|-----------|------------|--------|
| ABL-001 | LayerNorm (remove entirely) | Training fails completely | ⏳ Planned |
| ABL-002 | Residual connections (remove) | Gradient flow breaks, no training | ⏳ Planned |
| ABL-003 | MLP in transformer block | Attention alone insufficient | ⏳ Planned |
| ABL-004 | Attention (replace with MLP) | Severe quality drop | ⏳ Planned |
| ABL-005 | Positional encoding (remove) | Position info lost, poor quality | ⏳ Planned |
| ABL-006 | Dropout (set to 0) | Overfitting increases | ⏳ Planned |
| ABL-007 | Weight decay (set to 0) | Overfitting increases | ⏳ Planned |
| ABL-008 | Gradient clipping (remove) | Training instability | ⏳ Planned |
| ABL-009 | Causal mask (remove) | Cheats on next-token task | ⏳ Planned |
| ABL-010 | Weight tying (untie) | More params, similar quality | ⏳ Planned |

---

## Results Format

When completed, each ablation gets a section:

```markdown
## ABL-XXX: [Component]

**Hypothesis**: [What we expected]

**Setup**: [Config details]

**Results**:
| Metric | Baseline | Ablated | Change |
|--------|----------|---------|--------|
| Val Loss | 2.45 | 3.12 | +27% |
| ... | ... | ... | ... |

**Interpretation**: [Why this happened]

**Surprise**: [Anything unexpected]
```