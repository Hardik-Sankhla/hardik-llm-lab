# Hardik LLM Lab

> Understanding language models by building, breaking, measuring, and rebuilding them.

I started this project while studying *Build a Large Language Model (From Scratch)* by Sebastian Raschka.

Rather than stopping at reproducing the book's examples, I turned the learning process into a research-engineering laboratory.

The goal is simple:

    Understand → Implement → Measure → Break → Explain → Improve

## What I Built

- ✓ BPE tokenizer from first principles
- ✓ Self-attention & multi-head attention
- ✓ Transformer blocks (Pre-Norm)
- ✓ GPT-style language model
- ✓ Pretraining pipeline with mixed precision
- ✓ Fine-tuning pipeline (classification + instruction)
- ✓ LoRA / PEFT implementation
- ✓ Evaluation framework (perplexity, benchmarks, generation)
- ✓ Experiment tracking & reproducibility infrastructure
- ✓ CLI for training, generation, evaluation

## Research Questions

- How does vocabulary size affect training efficiency?
- How does context length affect memory & quality?
- How does model depth affect convergence?
- How does attention-head count affect performance?
- How does LoRA rank affect adaptation efficiency?
- What causes training instability?

## Quick Start

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Start JupyterLab for exploration
jupyter lab

# Train a model (see configs/)
python -m hardik_llm.training.train --config configs/pretrain_small.yaml

# Generate text
python -m hardik_llm.generation.generate --checkpoint checkpoints/best.pt --prompt "The transformer"

# Run an experiment
python -m hardik_llm.experiments.run --exp-id EXP-001
```

## Repository Structure

```
hardik-llm-lab/
├── reference/raschka/          # Sebastian's original code (learning reference)
├── src/hardik_llm/             # My independent implementations
│   ├── tokenizer/              # BPE tokenizer
│   ├── attention/              # Attention mechanisms
│   ├── model/                  # GPT model & components
│   ├── training/               # Training loop, optimizers, schedulers
│   ├── finetuning/             # Classification, instruction tuning, LoRA
│   ├── evaluation/             # Perplexity, benchmarks, generation
│   └── utils/                  # Logging, seeding, reproducibility
├── notebooks/                  # Research notebooks (one per concept)
├── experiments/                # Structured experiment registry
│   ├── configs/                # Experiment configurations
│   ├── runs/                   # Run artifacts
│   ├── results/                # Results JSON
│   └── plots/                  # Visualizations
├── guide/                      # Step-by-step experiment guides
├── docs/                       # Story-mode documentation (MkDocs)
├── tests/                      # Unit & integration tests
├── scripts/                    # Utility scripts
├── configs/                    # Training configs
├── checkpoints/                # Model checkpoints (gitignored)
├── assets/                     # Diagrams, figures
└── pyproject.toml              # Package config
```

## Documentation

Build the story-mode site:

```bash
pip install -e ".[dev]"
mkdocs serve
```

The documentation is deployed to: https://hardik-sankhla.gitlab.io/hardik-llm-lab

## Repository

GitLab: https://gitlab.com/hardik-sankhla/hardik-llm-lab

## Attribution

Learning foundation: This project was initially developed while studying *Build a Large Language Model (From Scratch)* by Sebastian Raschka (Manning Publications). The book served as a conceptual and educational reference.

The `reference/` directory contains the reference material where permitted. The implementations under `src/`, `experiments/`, `docs/`, and extensions represent my independent work.

Original work: https://github.com/rasbt/LLMs-from-scratch
Book: https://www.manning.com/books/build-a-large-language-model-from-scratch

## License

MIT License - see [LICENSE](LICENSE) for details.