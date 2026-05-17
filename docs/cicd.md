# CI/CD

BERTokenScope uses GitHub Actions for quality, tests, Docker validation, and release scaffolding.

## CI Workflow

`.github/workflows/ci.yml` runs:

- Ruff lint
- Ruff format check
- mypy type check
- Bandit security scan
- pip-audit dependency audit
- pytest
- contract tests
- regression tests
- compile check
- Docker Compose config validation
- Docker build validation

## Release Workflow

`.github/workflows/release.yml` runs on version tags:

```text
v*.*.*
```

It builds and pushes a container image to GitHub Container Registry.

## Staging Deploy Workflow

`.github/workflows/deploy-staging.yml` is a manual workflow scaffold. It is intentionally provider-neutral until cloud credentials are configured.

## Local CI

Run:

```powershell
.\scripts\ci_local.ps1
```

Optional lint/type tools run only if installed locally.
