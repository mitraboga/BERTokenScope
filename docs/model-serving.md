# Model Serving

BERTokenScope separates model-serving concerns from analysis logic.

## Registry

The model registry lives in `model_serving/registry.py` and pins the supported model IDs:

- `bert-base-uncased`
- `distilbert-base-uncased`
- `roberta-base`
- `ProsusAI/finbert`
- `sentence-transformers/all-MiniLM-L6-v2`

Each registry entry declares its task and fallback.

## Cache Policy

Live downloads are disabled by default:

```text
BERTSCOPE_ALLOW_MODEL_DOWNLOADS=false
```

This keeps deployments deterministic. To enable live Hugging Face loading, set:

```text
BERTSCOPE_ALLOW_MODEL_DOWNLOADS=true
```

and mount:

```text
BERTSCOPE_MODEL_CACHE_DIR=artifacts/model-cache
```

Live model serving also requires the optional dependency set:

```powershell
pip install -r requirements-models.txt
```

The public Streamlit deployment intentionally does not install these dependencies.

## Warmup

Use warmup endpoints to inspect readiness before routing traffic:

```text
GET  /models/status
POST /models/warmup
```

Warmup can be submitted asynchronously:

```json
{"model_id": "bert-base-uncased", "async_job": true}
```

Then poll:

```text
GET /jobs/{job_id}
```

## Timeouts

Inference and warmup operations use:

```text
BERTSCOPE_INFERENCE_TIMEOUT_SECONDS=30
```
