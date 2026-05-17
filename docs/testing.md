# Testing Strategy

BERTokenScope now uses layered testing.

## Default Tests

```powershell
pytest
```

Covers:

- API endpoints
- auth and roles
- security middleware
- persistence
- observability
- model serving
- explainability
- financial NLP
- embeddings
- model comparison

## Contract Tests

Contract tests validate OpenAPI and response-shape stability:

```powershell
pytest -m contract
```

## Regression Fixtures

Regression fixtures lock down expected NLP behavior for deterministic fallback paths:

```powershell
pytest -m regression
```

## API Smoke Test

Run after starting the API:

```powershell
.\scripts\smoke_api.ps1 -BaseUrl http://127.0.0.1:8788 -ApiKey $env:BERTSCOPE_API_KEY
```

## Load Probe

Run after starting the API:

```powershell
python scripts/load_probe.py --base-url http://127.0.0.1:8788 --api-key $env:BERTSCOPE_API_KEY --requests 25
```

This is not a substitute for full load testing, but it verifies basic throughput and latency behavior.
