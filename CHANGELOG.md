# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Professional repository configuration for GitHub
- GitHub Actions CI/CD pipeline with lint, test, build, publish, and docs deploy
- Codecov integration for coverage reporting
- PyPI publishing workflow on release
- GitHub Pages documentation deployment
- Professional badges in README
- Project URLs in pyproject.toml

### Changed
- Repository name standardized as `hardik-llm-lab` on GitHub
- Documentation site configured for GitHub Pages
- CI pipeline updated from GitLab CI to GitHub Actions
- README updated with professional badges and GitHub links

### Removed
- GitLab CI configuration (`.gitlab-ci.yml`)

## [0.1.0] - 2025-01-XX

### Added
- BPE tokenizer from first principles
- Self-attention & multi-head attention implementations
- Transformer blocks (Pre-Norm architecture)
- GPT-style language model
- Pretraining pipeline with mixed precision
- Fine-tuning pipeline (classification + instruction tuning)
- LoRA / PEFT implementation
- Evaluation framework (perplexity, benchmarks, generation)
- Experiment tracking & reproducibility infrastructure
- CLI for training, generation, evaluation
- Structured experiment registry
- Story-mode documentation (MkDocs)
- Unit & integration tests
- Development tooling (Ruff, Black, MyPy, pytest, pre-commit)

### Research Foundation
- Initial development while studying *Build a Large Language Model (From Scratch)* by Sebastian Raschka
- Reference implementations in `reference/raschka/`
- Independent implementations in `src/hardik_llm/`