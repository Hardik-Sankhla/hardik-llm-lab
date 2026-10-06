# Hardik LLM Lab - Development Commands
# Usage: make <target>

.PHONY: help install test lint format typecheck build-docs serve-docs clean

# Default target
help:
	@echo "Hardik LLM Lab - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install package in development mode with all deps"
	@echo "  make install-dev    Install dev dependencies only"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           Run ruff linter"
	@echo "  make format         Format code with black + ruff"
	@echo "  make typecheck      Run mypy type checking"
	@echo "  make check          Run all checks (lint + format + typecheck)"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run pytest with coverage"
	@echo "  make test-fast      Run pytest without coverage"
	@echo "  make test-watch     Run pytest in watch mode (requires pytest-watch)"
	@echo ""
	@echo "Documentation:"
	@echo "  make build-docs     Build MkDocs site to ./site/"
	@echo "  make serve-docs     Serve MkDocs with live reload (http://localhost:8000)"
	@echo "  make deploy-docs    Build and deploy to GitHub Pages (requires gh-pages branch)"
	@echo ""
	@echo "Building:"
	@echo "  make build          Build Python package (sdist + wheel)"
	@echo "  make build-check    Verify built package with twine"
	@echo ""
	@echo "Experiments:"
	@echo "  make exp-001        Run EXP-001 (tokenizer vocab size)"
	@echo "  make train-small    Train small model (configs/pretrain_small.yaml)"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean          Remove build artifacts, cache, generated files"
	@echo "  make clean-all      Clean + remove .venv, site, dist"

# ============================================================
# Setup
# ============================================================
install:
	pip install --upgrade pip
	pip install -e ".[dev,experiments]"

install-dev:
	pip install --upgrade pip
	pip install -e ".[dev]"

# ============================================================
# Code Quality
# ============================================================
lint:
	ruff check src/

format:
	ruff check --fix src/
	black src/

typecheck:
	mypy src/

check: lint format typecheck

# ============================================================
# Testing
# ============================================================
test:
	pytest -v --cov=hardik_llm --cov-report=term-missing --cov-report=html

test-fast:
	pytest -v

test-watch:
	ptw tests/ src/ -- -v

# ============================================================
# Documentation
# ============================================================
build-docs:
	mkdocs build --strict

serve-docs:
	mkdocs serve --dev-addr=0.0.0.0:8000

deploy-docs:
	mkdocs gh-deploy --force

# ============================================================
# Building
# ============================================================
build:
	pip install build
	python -m build

build-check:
	pip install twine
	twine check dist/*

# ============================================================
# Experiments (adjust configs as needed)
# ============================================================
exp-001:
	python -m hardik_llm.experiments.run --exp-id EXP-001

train-small:
	python -m hardik_llm.training.train --config configs/pretrain_small.yaml

# ============================================================
# Maintenance
# ============================================================
clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	rm -rf site public
	rm -rf checkpoints/*.pt checkpoints/*.pth
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

clean-all: clean
	rm -rf .venv
	rm -rf .cache