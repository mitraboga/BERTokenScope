# Operations

## Local Services

Set an API key before running the protected API:

```powershell
$env:BERTSCOPE_API_KEY="replace-with-a-long-random-secret"
$env:BERTSCOPE_API_KEY_ID="local-admin"
$env:BERTSCOPE_API_KEY_ROLE="admin"
```

Run the dashboard:

```powershell
.\scripts\run_app.ps1
```

Run the API:

```powershell
.\scripts\run_api.ps1 -Port 8788
```

Run with Docker Compose:

```powershell
docker compose up --build
```

Run through the optional gateway:

```powershell
docker compose --profile gateway up --build
```

## Artifacts

BERTokenScope stores run metadata in SQLite at `artifacts/bertscope.db` and full run artifacts in `artifacts/runs` by default. Each API workflow writes:

- request payload
- response payload
- task name
- duration
- timestamp
- summary metadata

Configure the database location with:

```powershell
$env:BERTSCOPE_DATABASE_URL="sqlite:///artifacts/bertscope.db"
```

Use the API to inspect runs:

```text
GET /runs
GET /runs/{run_id}
```

Protected API endpoints require:

```text
X-API-Key: <your-api-key>
```

## Production Notes

The current deployment profile is multi-service and local-production ready. For a hosted production environment, add rate limiting, centralized logging, managed object storage for artifacts, and a model cache volume.

## Model Serving

BERTokenScope uses a pinned model registry with offline-safe fallback behavior. By default, live model downloads are disabled and local cache status is exposed through:

```text
GET /models/registry
GET /models/status
POST /models/warmup
POST /models/warmup {"model_id": "...", "async_job": true}
GET /jobs/{job_id}
```

Configure model serving with:

```powershell
$env:BERTSCOPE_MODEL_CACHE_DIR="artifacts/model-cache"
$env:BERTSCOPE_ALLOW_MODEL_DOWNLOADS="false"
$env:BERTSCOPE_INFERENCE_TIMEOUT_SECONDS="30"
```

## Observability

Every API response includes `X-Request-ID`. Clients may provide one, or BERTokenScope generates it.

```text
GET /observability/summary
GET /metrics
```

`/metrics` exposes Prometheus-style text metrics for request counts, status codes, routes, errors, and total latency.

## API Robustness

Versioned routes are available under:

```text
/api/v1
```

For idempotent POST retries, send:

```text
X-Idempotency-Key: <unique-key>
```

Run history supports pagination:

```text
GET /api/v1/runs?limit=20&offset=0
```

Errors use a standard envelope:

```json
{"error": {"code": "http_error", "message": "...", "request_id": "..."}}
```

## Security Controls

Configure CORS, request size, and rate limiting:

```powershell
$env:BERTSCOPE_ALLOWED_ORIGINS="http://localhost:8501"
$env:BERTSCOPE_MAX_REQUEST_BYTES="1000000"
$env:BERTSCOPE_RATE_LIMIT_REQUESTS="120"
$env:BERTSCOPE_RATE_LIMIT_WINDOW_SECONDS="60"
```

## Data Governance

Configure artifact redaction, retention, and audit logging:

```powershell
$env:BERTSCOPE_REDACT_ARTIFACTS="true"
$env:BERTSCOPE_RETENTION_DAYS="30"
$env:BERTSCOPE_AUDIT_LOG_PATH="artifacts/audit/audit.jsonl"
```

Governance endpoints:

```text
GET /governance/status
GET /governance/audit
POST /governance/retention/cleanup
```

## Testing

See [testing.md](testing.md) for contract tests, regression fixtures, API smoke tests, and the local load probe.

## CI/CD

See [cicd.md](cicd.md) for linting, type checks, dependency audit, Docker validation, release image publishing, and staging deployment scaffolding.
