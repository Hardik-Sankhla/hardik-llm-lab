# Contributing to Hardik LLM Lab

Thank you for your interest in contributing! This project is a personal research laboratory, but contributions that improve code quality, documentation, or experimental rigor are welcome.

## Ways to Contribute

### 🐛 Bug Reports
- Use the [issue tracker](https://github.com/Hardik-Sankhla/hardik-llm-lab/issues)
- Include: Python version, OS, steps to reproduce, expected vs actual behavior
- Minimal reproducible example if possible

### 💡 Feature Requests
- Open an issue with the `enhancement` label
- Describe the use case and proposed API
- Link to relevant papers or prior art if applicable

### 📝 Documentation
- Fix typos, clarify explanations, add examples
- Documentation lives in `docs/` (MkDocs/Material theme)
- Run locally: `mkdocs serve`

### 🧪 Experiments & Research
- Add new experiment configs in `experiments/configs/`
- Document findings in `docs/research/`
- Follow the existing experiment structure

### 🔧 Code Contributions
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run quality checks: `make check` or `just check`
5. Run tests: `make test` or `just test`
6. Submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/hardik-llm-lab.git
cd hardik-llm-lab

# Install in development mode with all dependencies
pip install -e ".[dev,experiments]"

# Install pre-commit hooks
pre-commit install

# Verify setup
pytest
ruff check src/
black --check src/
mypy src/
```

## Code Style

- **Formatter**: Black (line length 100)
- **Linter**: Ruff (with strict rules)
- **Type Checker**: MyPy (strict mode)
- **Tests**: pytest with coverage

Run all checks:
```bash
make check    # lint + format + typecheck
make test     # pytest with coverage
```

Or with Just:
```bash
just check
just test
```

## Project Structure

```
src/hardik_llm/          # Main package (import name: hardik_llm)
├── tokenizer/           # BPE tokenizer
├── attention/           # Attention mechanisms
├── model/               # GPT model & components
├── training/            # Training loop, optimizers, schedulers
├── finetuning/          # Classification, instruction tuning, LoRA
├── evaluation/          # Perplexity, benchmarks, generation
├── experiments/         # Experiment runners
└── utils/               # Logging, seeding, reproducibility

experiments/             # Experiment registry
├── configs/             # YAML configs
├── runs/                # Artifacts
├── results/             # JSON results
└── plots/               # Visualizations

notebooks/               # Research notebooks
guide/                   # Step-by-step guides
docs/                    # MkDocs documentation
tests/                   # Unit & integration tests
configs/                 # Training configs
scripts/                 # Utility scripts
```

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new attention variant
fix: resolve gradient explosion in deep models
docs: update installation guide
refactor: simplify training loop
test: add coverage for tokenizer
chore: update dependencies
```

## Pull Request Process

1. Ensure all CI checks pass
2. Update CHANGELOG.md under `[Unreleased]`
3. Add tests for new functionality
4. Update documentation if API changes
5. Request review from maintainer

## Questions?

Open a [discussion](https://github.com/Hardik-Sankhla/hardik-llm-lab/discussions) or [issue](https://github.com/Hardik-Sankhla/hardik-llm-lab/issues).

## License

By contributing, you agree that your contributions will be licensed under the MIT License.