# BERTokenScope Roadmap

## Phase 1: Platform Foundation

- Streamlit dashboard with masked-token, attention, finance, and comparison sections.
- FastAPI endpoints for service integration.
- Lazy Hugging Face model loading with deterministic fallbacks.
- Unit tests for core attention and financial NLP logic.
- CI and Docker baseline.

## Phase 2: Transformer Interpretability Depth

- Full layer/head selector in the Attention Explorer.
- Attention rollout and attention entropy charts.
- CLS-token tracking across layers.
- Exportable attention artifacts for reproducible analysis.
- CS50AI-compatible static diagram export.

## Phase 3: Financial Intelligence

- FinBERT live sentiment adapter.
- Earnings-call transcript parser.
- Quarter-over-quarter tone drift.
- Risk, uncertainty, and optimism time series.
- Transcript-level risk and tone diagnostics.

## Phase 4: Embedding Intelligence

- Sentence and document embedding generation.
- Company and transcript similarity maps.
- 2D semantic visualization.
- Clustering for topic/company grouping.
- API and dashboard embedding explorer.

## Phase 5: Multi-Model Comparison

- Masked-token comparison across BERT-family profiles.
- Financial sentiment comparison with lexical and FinBERT-compatible paths.
- Embedding comparison across deterministic and transformer-compatible encoders.
- Latency and confidence dashboards.
- API endpoint for comparison reports.

## Phase 6: Explainability Lab

- Token attribution for attention and financial language signals.
- Counterfactual token removal for prediction impact analysis.
- Analyst-facing rationale generation.
- API endpoint for explanation reports.
- Dashboard explainability workflow.

## Phase 7: MLOps and Portfolio Polish

- MLflow experiment tracking.
- Model benchmark table for latency, memory, and confidence.
- GitHub Actions with linting and test matrix.
- Docker Compose for API plus dashboard.
- Deployed demo with cached lightweight models.
- Local JSON artifact tracking and run history.
