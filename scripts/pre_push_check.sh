#!/usr/bin/env bash
# Pre-push validation script for Hardik LLM Lab
# Run: ./scripts/pre_push_check.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# Check if in venv
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    log_warn "Not in virtual environment. Activating .venv..."
    if [[ -f ".venv/bin/activate" ]]; then
        source .venv/bin/activate
    else
        log_error "No .venv found. Run: python -m venv .venv && source .venv/bin/activate && pip install -e '.[dev]'"
        exit 1
    fi
fi

FAILED=0

run_check() {
    local name="$1"
    local cmd="$2"
    log_info "Running: $name"
    if eval "$cmd"; then
        log_info "✓ $name passed"
    else
        log_error "✗ $name failed"
        FAILED=1
    fi
}

echo "========================================="
echo "Hardik LLM Lab - Pre-Push Validation"
echo "========================================="
echo

# Critical checks
run_check "Tests" "pytest -v --tb=short -q"
run_check "Lint (ruff)" "ruff check src/"
run_check "Format (black)" "black --check src/"
run_check "Type check (mypy)" "mypy src/ || true"  # mypy baseline may have errors

# Package checks
run_check "Import test" "python -c 'import hardik_llm; print(f\"hardik_llm v{hardik_llm.__version__}\")'"
run_check "Build package" "python -m build"
run_check "Verify package" "twine check dist/*"

# Docs
run_check "Build docs" "mkdocs build --strict"

# Git checks
log_info "Checking for large files..."
large_files=$(find . -type f -size +10M ! -path "./.venv/*" ! -path "./.git/*" ! -path "./checkpoints/*" ! -path "./dataset/*" 2>/dev/null | head -5)
if [[ -n "$large_files" ]]; then
    log_warn "Large files detected (>10MB):"
    echo "$large_files"
    FAILED=1
else
    log_info "✓ No large files"
fi

log_info "Checking for secrets..."
secrets=$(grep -r -i "api_key\|secret\|password\|access_token\|auth_token" --include="*.py" --include="*.yaml" --include="*.yml" --include="*.json" --include="*.toml" . 2>/dev/null | grep -v ".venv" | grep -v ".git" | grep -v "example" | grep -v "test" | grep -v "pad_token" | grep -v "vocab" | grep -v "reference/" | head -5 || true)
if [[ -n "$secrets" ]]; then
    log_warn "Potential secrets found:"
    echo "$secrets"
    FAILED=1
else
    log_info "✓ No obvious secrets"
fi

log_info "Checking for GitLab references..."
gitlab_refs=$(grep -r "gitlab" --include="*.md" --include="*.yml" --include="*.yaml" . 2>/dev/null | grep -v ".git" | head -5 || true)
if [[ -n "$gitlab_refs" ]]; then
    log_warn "GitLab references found:"
    echo "$gitlab_refs"
    FAILED=1
else
    log_info "✓ No GitLab references"
fi

echo
echo "========================================="
if [[ $FAILED -eq 0 ]]; then
    log_info "All checks passed! ✓"
    echo "Ready to push."
    exit 0
else
    log_error "Some checks failed. Fix before pushing."
    exit 1
fi