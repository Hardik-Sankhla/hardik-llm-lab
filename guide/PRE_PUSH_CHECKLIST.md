# Pre-Push Checklist for Hardik LLM Lab

> Run this checklist before every push to maintain a professional, working repository.

---

## 🔴 Critical (Must Pass)

### Tests & Code Quality
- [ ] **Tests pass**: `pytest -v` → all green (or expected skips)
- [ ] **Lint clean**: `ruff check src/` → no errors (warnings OK)
- [ ] **Format clean**: `black --check src/` → no changes needed
- [ ] **Type check**: `mypy src/` → no new errors (baseline exists)

### Package Integrity
- [ ] **Import works**: `python -c "import hardik_llm; print(hardik_llm.__version__)"`
- [ ] **CLI works**: `hardik-llm --help` (or `python -m hardik_llm.cli --help`)
- [ ] **Build works**: `python -m build` → creates `dist/*.whl` and `dist/*.tar.gz`
- [ ] **Package verify**: `twine check dist/*` → no errors

---

## 🟡 Important (Should Pass)

### Documentation
- [ ] **MkDocs builds**: `mkdocs build --strict` → no warnings/errors
- [ ] **MkDocs serves**: `mkdocs serve` → loads locally at http://localhost:8000
- [ ] **README renders**: Check GitHub preview after push
- [ ] **CHANGELOG updated**: Add entries under `[Unreleased]` for notable changes

### Configuration Consistency
- [ ] **pyproject.toml version** matches `__version__` in `src/hardik_llm/__init__.py`
- [ ] **Project URLs** in pyproject.toml point to correct GitHub repo
- [ ] **mkdocs.yml** `repo_url` / `site_url` match GitHub repo
- [ ] **README badges** use correct repo name (`Hardik-Sankhla/hardik-llm-lab`)
- [ ] **CI workflow** references correct package name (`hardik_llm` for coverage)

---

## 🟢 Good Practice (Nice to Have)

### Git Hygiene
- [ ] **Branch naming**: `feature/`, `fix/`, `docs/`, `refactor/`, `chore/`
- [ ] **Commit messages**: Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`)
- [ ] **No large files**: No `.pt`, `.pth`, `.zip`, `.csv`, `.parquet` committed
- [ ] **No secrets**: No API keys, tokens, `.env` files committed
- [ ] **Clean history**: Squash fixup commits before PR merge

### Dependency Management
- [ ] **Dependencies pinned**: `pyproject.toml` uses `>=` not `==` (except upper bounds for conflicts)
- [ ] **Lock file considered**: `uv lock` or `pip-tools` if using lock files
- [ ] **Security scan**: `pip-audit` or GitHub Dependabot alerts checked

### Experiment Tracking
- [ ] **Experiment configs** in `experiments/configs/` are versioned
- [ ] **Results** in `experiments/results/` are JSON (not pickle)
- [ ] **Plots** in `experiments/plots/` are committed for key results

---

## 🔧 Automated Checks (Run in CI)

These run automatically on push/PR — fix locally first:

| Check | Command | CI Job |
|-------|---------|--------|
| Lint | `ruff check src/` | `lint-and-typecheck` |
| Format | `black --check src/` | `lint-and-typecheck` |
| Types | `mypy src/` | `lint-and-typecheck` |
| Tests | `pytest -v --cov=hardik_llm` | `test` |
| Build | `python -m build && twine check dist/*` | `build` |
| Docs | `mkdocs build --strict` | `deploy-docs` |

---

## 📋 Release Checklist (When Tagging v0.x.x)

- [ ] Update version in `pyproject.toml` AND `src/hardik_llm/__init__.py`
- [ ] Update `CHANGELOG.md` with release date and sections
- [ ] Run full test suite: `pytest -v --cov=hardik_llm --cov-report=term-missing`
- [ ] Build and verify: `python -m build && twine check dist/*`
- [ ] Create GitHub Release (triggers PyPI publish via trusted publishing)
- [ ] Verify PyPI package: `pip install hardik-llm-lab==X.Y.Z`
- [ ] Verify docs deploy: Check https://hardik-sankhla.github.io/hardik-llm-lab

---

## 🚀 Quick Local Commands

```bash
# Full pre-push validation (run from repo root)
cd /mnt/Storage/hardik-llm-lab

# 1. Activate venv
source .venv/bin/activate

# 2. Quick check (tests + lint + format)
make check && make test-fast

# 3. Full check (includes typecheck + build + docs)
make check && make test && make build && make build-docs

# 4. Or with just
just check && just test && just build && just build-docs
```

---

## 📁 File Reference

| File | Purpose | Update When |
|------|---------|-------------|
| `pyproject.toml` | Package metadata, deps, tool config | Version, deps, URLs, tool settings |
| `src/hardik_llm/__init__.py` | Package version, exports | Version, new public APIs |
| `mkdocs.yml` | Docs config | Site/repo URLs, nav structure |
| `.github/workflows/ci.yml` | CI/CD pipeline | New jobs, Python versions, secrets |
| `README.md` | Project landing | Badges, links, quickstart, structure |
| `docs/index.md` | Docs home page | Sync with README |
| `CHANGELOG.md` | Release history | Every PR with user-facing changes |
| `CONTRIBUTING.md` | Contributor guide | Process changes, new guidelines |
| `SECURITY.md` | Vulnerability policy | New security considerations |
| `CODE_OF_CONDUCT.md` | Community standards | Rarely |

---

## ⚠️ Common Pitfalls to Avoid

| Pitfall | Prevention |
|---------|------------|
| Version mismatch | Single source: `pyproject.toml` → read by `src/hardik_llm/__init__.py` via `importlib.metadata` |
| Import path drift | Always use `hardik_llm.*` (not `llm_lab.*`) |
| GitLab refs | Search for `gitlab` before push: `grep -r gitlab --include="*.md" --include="*.yml" .` |
| Missing newlines | `black` + `ruff --fix` handle this |
| Untyped code | Add type hints; use `# type: ignore` sparingly |
| Large files | Use `.gitignore` patterns; use Git LFS if needed |
| Secret leaks | Never commit `.env`; use GitHub Secrets for CI |

---

## 🔗 Useful Links

- **Repo**: https://github.com/Hardik-Sankhla/hardik-llm-lab
- **Docs**: https://hardik-sankhla.github.io/hardik-llm-lab
- **CI**: https://github.com/Hardik-Sankhla/hardik-llm-lab/actions
- **PyPI**: https://pypi.org/project/hardik-llm-lab
- **Codecov**: https://codecov.io/gh/Hardik-Sankhla/hardik-llm-lab

---

*Keep this checklist updated as the project evolves. Last updated: 2025-10-18*