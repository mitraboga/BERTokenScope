# Portfolio Deployment Route

The recommended portfolio route is Streamlit-first:

1. Deploy the dashboard publicly on Streamlit Community Cloud.
2. Keep the public app in offline/fallback mode for reliability.
3. Link the GitHub repository to show the enterprise backend architecture.
4. Deploy the FastAPI backend separately later if live transformer inference is needed.

This gives reviewers a live product surface immediately while still demonstrating that the codebase is engineered beyond a simple Streamlit prototype.

## What Reviewers Can See Live

- Masked-token prediction workflow.
- Attention heatmaps and token relationships.
- Explainability reports with token attribution.
- Financial NLP signals and transcript drift.
- Embedding maps and document similarity.
- Multi-model comparison interface.
- Run history view.

## What Reviewers Can Inspect in GitHub

- FastAPI service with versioned routes.
- API-key authentication and role-aware access control.
- Persistent run repository and artifact tracking.
- Model registry, cache policy, warmup, and async job boundaries.
- Observability, structured logs, metrics, and request IDs.
- CORS, rate limiting, body-size limits, and security headers.
- Data governance, redaction, audit logging, and retention cleanup.
- Docker Compose, CI/CD workflows, tests, smoke checks, and load-probe scaffolding.

## Why Offline Mode Is Correct for the Public Demo

Streamlit Community Cloud is best used here as a stable portfolio experience. Running large transformer downloads during public app startup can create avoidable cold-start and memory risk. BERTokenScope therefore keeps the public app deterministic while preserving live model-serving code for separate infrastructure.
