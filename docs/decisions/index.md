# Architecture Decision Records (ADRs)

*Each ADR documents a significant architectural decision.*

## ADR Index

| ID | Title | Status | Date |
|----|-------|--------|------|
| ADR-001 | Use PyTorch as DL Framework | Accepted | 2025-01-15 |
| ADR-002 | Use BPE Tokenization | Accepted | 2025-01-15 |
| ADR-003 | Pre-Norm Transformer Architecture | Accepted | 2025-01-15 |
| ADR-004 | MLflow for Experiment Tracking | Accepted | 2025-01-15 |
| ADR-005 | MkDocs Material for Documentation | Accepted | 2025-01-15 |
| ADR-006 | MIT License for Original Code | Accepted | 2025-01-15 |
| ADR-007 | Package Name: hardik_llm | Accepted | 2025-01-15 |
| ADR-008 | Repository Name: hardik-llm-lab | Accepted | 2025-01-15 |

---

## ADR Template

```markdown
# ADR-XXX: [Title]

## Status
[Proposed | Accepted | Rejected | Deprecated | Superseded]

## Context
[What is the issue that motivates this decision? What forces are at play?]

## Options
### Option 1: [Name]
- Pros:
- Cons:

### Option 2: [Name]
- Pros:
- Cons:

## Decision
[What we decided. Full sentence.]

## Reason
[Why this option was chosen over others.]

## Consequences
### Positive
- [Benefits]

### Negative
- [Drawbacks/costs]

### Neutral
- [Things that just are different]

## Related
- [Links to related ADRs, issues, PRs]
```

---

## ADR-001: Use PyTorch as DL Framework

**Status**: Accepted

**Context**: Need a deep learning framework for implementing transformer models from scratch.

**Options**:
- PyTorch: Dynamic graphs, strong ecosystem, research-friendly
- TensorFlow/Keras: Static graphs, production deployment focus
- JAX: Functional, high-performance, steeper learning curve

**Decision**: Use PyTorch.

**Reason**: Best balance of research flexibility, ecosystem (torch.compile, FSDP, etc.), and community support. Dynamic graphs make debugging and experimentation easier.

**Consequences**:
- Positive: Easy debugging, rich ecosystem, standard in research
- Negative: Deployment requires extra work vs TensorFlow Serving
- Neutral: Must learn PyTorch idioms

---

## ADR-002: Use BPE Tokenization

**Status**: Accepted

**Context**: Need tokenization method for English text.

**Options**:
- BPE (Byte-Pair Encoding): Subword, used by GPT, good compression
- WordPiece: Similar to BPE, used by BERT
- Unigram: Probabilistic, used by T5
- Character-level: No vocab limit, very long sequences

**Decision**: Use BPE.

**Reason**: Standard for GPT-style models. Good balance of vocab size and sequence length. tiktoken provides production reference.

**Consequences**:
- Positive: Compatible with GPT ecosystem, well-understood
- Negative: Requires training, vocab size hyperparameter
- Neutral: Must handle special tokens explicitly

---

## ADR-003: Pre-Norm Transformer Architecture

**Status**: Accepted

**Context**: Choose between Pre-Norm (GPT-2/3) and Post-Norm (original Transformer).

**Options**:
- Pre-Norm: LN → Attention → Residual
- Post-Norm: Attention → Residual → LN

**Decision**: Use Pre-Norm.

**Reason**: Enables training deeper models (>12 layers) without gradient instability. Standard in modern LLMs (GPT, LLaMA, etc.). Post-Norm requires extensive LR warmup and still struggles at depth.

**Consequences**:
- Positive: Stable deep training, standard practice
- Negative: Slightly different gradient flow than original paper
- Neutral: Must adjust initialization for residual scaling

---

## ADR-004: MLflow for Experiment Tracking

**Status**: Accepted

**Context**: Need reproducible experiment tracking for 30+ planned experiments.

**Options**:
- MLflow: Open source, local/server, good UI, integrates with many tools
- Weights & Biases: SaaS, excellent UI, free for academic
- TensorBoard: Built into PyTorch, basic experiment tracking
- Sacred + Omniboard: Less maintained

**Decision**: Use MLflow (local file backend initially).

**Reason**: Open source, self-hosted, good Python API, supports artifacts/models, can migrate to server later. No external dependency.

**Consequences**:
- Positive: Full control, no cost, works offline
- Negative: UI less polished than W&B, manual setup for server
- Neutral: Can add W&B later as optional dependency

---

## ADR-005: MkDocs Material for Documentation

**Status**: Accepted

**Context**: Need story-mode documentation site for portfolio.

**Options**:
- MkDocs Material: Beautiful, navigation, search, widely used
- Sphinx: Python standard, more complex
- Docusaurus: React-based, more customization
- GitBook: SaaS, not self-hosted

**Decision**: Use MkDocs Material.

**Reason**: Best balance of beauty, simplicity, and features for technical documentation. GitHub Pages deployment is trivial. Material theme is professional.

**Consequences**:
- Positive: Professional output, easy to write in Markdown, search built-in
- Negative: Less interactive than React-based sites
- Neutral: Must learn MkDocs config

---

## ADR-006: MIT License for Original Code

**Status**: Accepted

**Context**: Choose license for original implementations in this repo.

**Options**:
- MIT: Simple, permissive, widely used
- Apache-2.0: Patent grant, matches reference
- BSD-3-Clause: Similar to MIT
- GPL-3.0: Copyleft, restricts commercial use

**Decision**: MIT License.

**Reason**: Maximum permissibility for portfolio project. Simple, well-understood. Reference material is Apache-2.0 but our independent code can be MIT.

**Consequences**:
- Positive: Anyone can use/modify/distribute commercially
- Negative: No patent protection (unlike Apache-2.0)
- Neutral: Must maintain attribution for reference material

---

## ADR-007: Package Name: hardik_llm

**Status**: Accepted

**Context**: Python package name for `src/` module.

**Options**:
- hardik_llm: Personal, clear ownership
- minillm: Descriptive, generic
- scratch_llm: Descriptive
- llm_lab: Matches repo name

**Decision**: hardik_llm

**Reason**: Personal branding for portfolio. Clear this is *my* implementation. Avoids generic names that could conflict.

**Consequences**:
- Positive: Personal brand, unique on PyPI
- Negative: Less descriptive of function
- Neutral: Import is `from hardik_llm import ...`

---

## ADR-008: Repository Name: hardik-llm-lab

**Status**: Accepted

**Context**: GitHub repository name.

**Options**:
- hardik-llm-lab: Personal + "lab" = research environment
- hardik-llm: Simpler
- llm-lab: Generic
- minillm: Descriptive

**Decision**: hardik-llm-lab

**Reason**: "Lab" emphasizes research/experimentation over single implementation. Personal name for portfolio identity. Available on GitHub.

**Consequences**:
- Positive: Clear identity, "lab" sets right expectations
- Negative: Longer to type
- Neutral: GitHub URL: github.com/Hardik-Sankhla/hardik-llm-lab
```