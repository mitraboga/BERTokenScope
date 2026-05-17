# Deployment Architecture

BERTokenScope is structured as a deployable multi-service AI platform.

## Services

```text
gateway      nginx reverse proxy, optional profile
dashboard    Streamlit analyst interface
api          FastAPI inference and orchestration service
worker       background worker placeholder for async model jobs
redis        queue/cache foundation for future async workloads
artifacts    mounted artifact volume for run history
database     SQLite metadata store locally, Postgres-compatible boundary next
model-cache  mounted Hugging Face model cache volume
```

## Request Flow

```text
Browser -> Gateway -> Dashboard
Client  -> Gateway -> API -> Analysis Engines -> SQLite Metadata + Artifacts
Worker  -> Redis -> Long-running model jobs
API     -> Model Registry -> Cache Status / Warmup / Fallbacks
API     -> Metrics Registry -> /metrics + structured request logs
API     -> /api/v1 -> versioned routes + idempotency + error envelopes
```

## Local Production Profile

Copy `.env.example` to `.env`, set `BERTSCOPE_API_KEY`, then run:

```powershell
docker compose up --build
```

Optional gateway profile:

```powershell
docker compose --profile gateway up --build
```

Gateway routes:

```text
http://localhost:8080/      dashboard
http://localhost:8080/api/  API
```

## Health Checks

```text
GET /health  liveness and version metadata
GET /ready   readiness probe
```

## Production Mapping

For cloud deployment:

- run `api` behind an application load balancer
- run `dashboard` as a separate internal app or analyst console
- run `worker` on CPU/GPU nodes depending on workload
- replace local artifact volumes with object storage
- replace Redis container with managed Redis
- replace local SQLite with managed Postgres for multi-instance deployments
- mount or prebuild model-cache volumes for live transformer serving
- terminate TLS at the gateway/load balancer
