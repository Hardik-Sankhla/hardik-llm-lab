# Hardik LLM Lab - Justfile (modern command runner)
# Install: cargo install just  OR  pip install just
# Usage: just <recipe>

# Default recipe
default: help

# ============================================================
# Help
# ============================================================
help:
	@just --list --unsorted

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

check:
	just lint
	just format
	just typecheck

# ============================================================
# Testing
# ============================================================
test:
	pytest -v --cov=hardik_llm --cov-report=term-missing --cov-report=html

test-fast:
	pytest -v

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
# Experiments
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

clean-all:
	just clean
	rm -rf .venv
	rm -rf .cache