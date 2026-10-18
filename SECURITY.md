# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly:

**Do not open a public issue.** Instead, email the maintainer directly at:

**datascientist.hardiksankhla@gmail.com**

Include the following information:
- Description of the vulnerability
- Steps to reproduce (if applicable)
- Potential impact
- Suggested fix (if you have one)

## Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Fix Development**: Depends on severity and complexity
- **Disclosure**: Coordinated with reporter after fix is ready

## Security Considerations

This is a research laboratory for LLM implementations. Key security notes:

### Model Checkpoints
- Model weights in `checkpoints/` are **not versioned** (gitignored)
- Only download checkpoints from trusted sources
- Verify checksums before loading untrusted models

### Dependencies
- Dependencies are pinned in `pyproject.toml`
- Run `pip-audit` or `pip check` regularly
- GitHub Actions runs dependency scanning

### Data Handling
- Training data in `dataset/` is **not versioned** (gitignored)
- Never commit personal data, API keys, or secrets
- `.env` files are gitignored

### Experiment Tracking
- MLflow, Weights & Biases, TensorBoard configs may contain credentials
- Never log secrets to experiment trackers
- Use environment variables for API keys

### CI/CD
- GitHub Actions workflows use minimal permissions
- PyPI publishing uses trusted publishing (OIDC), not API tokens
- Codecov token stored as repository secret

## Disclosure Policy

Security fixes will be released as patch versions. Vulnerability details will be disclosed after a fix is available, typically within 30 days of coordinated disclosure.

## Acknowledgments

Security researchers who report vulnerabilities responsibly will be acknowledged in the CHANGELOG (unless they prefer anonymity).