# BERTokenScope
### Visualizing How Transformers Understand Language, Context, and Financial Text

BERTokenScope is a transformer explainability platform that extends the CS50AI Attention project into a production-style NLP intelligence system. It visualizes self-attention, masked-token predictions, token relationships, embeddings, and financial language signals so analysts and ML engineers can inspect how BERT-style models understand context.

At its core, BERTokenScope answers one question:

> How does a transformer decide what words matter?

## What It Does

- **Attention Explorer**: layer-by-layer and head-by-head attention inspection.
- **Masked Word Lab**: BERT-style `[MASK]` prediction with top-k probabilities.
- **Token Relationship Analysis**: strongest token-to-token attention links.
- **Financial NLP Intelligence**: sentiment, risk language, uncertainty, and executive tone signals.
- **Transcript Drift Analysis**: compare financial tone across quarters or reporting periods.
- **Explainability Lab**: token importance and attention-based attribution scaffolding.
- **Counterfactual Explanations**: remove important tokens and measure score impact.
- **Embedding Explorer**: sentence/document embeddings with dimensionality reduction hooks.
- **Company Similarity Maps**: cluster transcript excerpts, companies, or filings in semantic space.
- **Model Comparison**: compare BERT, DistilBERT, RoBERTa, and FinBERT-compatible models.
- **Runtime Benchmarking**: compare confidence, latency, and outputs across model families.
- **Run Tracking**: persist local artifacts for API analyses and experiment history.
- **FastAPI Service**: API endpoints for masked-token prediction, financial text analysis, and health checks.

## Why This Project Exists

The original CS50AI Attention assignment predicts masked words and generates static attention diagrams. BERTokenScope keeps that foundation, then scales it into an industry-grade interpretability platform:

- reusable Python services instead of one-off scripts
- a Streamlit dashboard instead of static outputs only
- typed domain models and deterministic fallbacks
- finance-specific text intelligence
- transcript chunking and period-over-period tone analytics
- embedding maps, similarity scoring, and clustering for document intelligence
- multi-model comparison across masked language, finance, and embedding tasks
- explainability reports with token attribution, rationales, and counterfactual impacts
- local run history, JSON artifacts, Docker Compose, and CI checks
- SQLite-backed run metadata repository with JSON artifact payloads
- API-key authentication with role-aware route protection
- multi-service deployment topology with API, dashboard, worker, Redis, and optional gateway
- model registry, cache policy, warmup endpoints, and async serving job boundaries
- request IDs, structured JSON request logs, and Prometheus-style metrics
- CORS, security headers, request size limits, rate limiting, and safe error responses
- `/api/v1` versioned routes, idempotency keys, pagination, and standard error envelopes
- API contract tests, NLP regression fixtures, smoke scripts, and load-probe scaffolding
- CI/CD with linting, formatting checks, type checks, security scans, Docker builds, release images, and staging deploy scaffold
- data governance with redaction, audit logs, retention cleanup, and financial-use disclaimers
- API-first architecture for future integration
- testable components that do not require model downloads

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Public Portfolio Deployment

BERTokenScope is prepared for the recommended portfolio route:

- deploy the Streamlit dashboard publicly on Streamlit Community Cloud
- keep the hosted dashboard in offline/fallback mode for reliability
- link the GitHub repository to show the enterprise FastAPI backend architecture
- deploy the FastAPI backend separately later if live transformer serving is needed

Streamlit Community Cloud settings:

```text
Main file path: app/streamlit_app.py
Required secrets: none
Recommended mode: offline-safe demo
```

See [docs/streamlit-cloud.md](docs/streamlit-cloud.md) and [docs/portfolio-deployment.md](docs/portfolio-deployment.md).

Run the API:

```powershell
$env:BERTSCOPE_API_KEY="replace-with-a-long-random-secret"
uvicorn api.main:app --reload
```

Run tests:

```powershell
pytest
```

Run both services with Docker Compose:

```powershell
docker compose up --build
```

Use the optional gateway profile:

```powershell
docker compose --profile gateway up --build
```

## Model Downloads

BERTokenScope lazy-loads Hugging Face models only when live serving is enabled. The public dashboard uses deterministic fallback behavior so it does not require model downloads, GPU access, or Hugging Face cache state.

For live transformer inference, install the optional model dependencies:

```powershell
pip install -r requirements-models.txt
```

Then configure:

```text
BERTSCOPE_ALLOW_MODEL_DOWNLOADS=true
```

## Project Structure

```text
BERTokenScope/
├── api/                    FastAPI service
├── app/                    Streamlit dashboard
├── attention/              Attention extraction, rollout, heatmaps, token links
├── ber_tokenscope/         Shared settings, schemas, model adapters
├── embeddings/             Embedding generation and reduction helpers
├── explainability/         Token importance and attribution helpers
├── financial_nlp/          Finance-aware NLP analytics
├── tests/                  Unit tests
├── configs/                Runtime config
└── docs/                   Architecture and CS50AI extension notes
```

## Portfolio Summary

Engineered **BERTokenScope**, a transformer intelligence and explainability platform extending CS50AI attention mechanisms into interactive NLP systems for attention visualization, masked-token prediction, financial sentiment analysis, embedding analytics, and multi-model transformer interpretability using BERT, FinBERT, Streamlit, FastAPI, and PyTorch.
