# Streamlit Community Cloud Deployment

BERTokenScope is prepared for a practical portfolio deployment path:

- Public Streamlit dashboard for recruiters and reviewers.
- Offline-safe deterministic model behavior for reliable demos.
- GitHub repository showing the enterprise FastAPI, model-serving, governance, and CI architecture.
- Optional FastAPI deployment later when live model-serving infrastructure is available.

## Recommended Public Demo Setup

Use Streamlit Community Cloud with:

```text
Repository: your public GitHub BERTokenScope repository
Branch: main
Main file path: app/streamlit_app.py
```

The dashboard intentionally initializes its interactive workflows in offline-safe mode. That means the public app does not need Hugging Face model downloads, GPU access, or a separate API service to demonstrate the core product experience.

## Required Secrets

No secrets are required for the public dashboard-only deployment.

Optional Streamlit secrets or environment variables:

```text
BERTSCOPE_ENVIRONMENT=demo
BERTSCOPE_ALLOW_MODEL_DOWNLOADS=false
BERTSCOPE_REDACT_ARTIFACTS=true
```

## Dependency Strategy

`requirements.txt` is optimized for the public dashboard and API scaffolding. Heavy live-model dependencies are intentionally isolated in:

```text
requirements-models.txt
```

Install those only for deployments that need live Hugging Face inference:

```powershell
pip install -r requirements-models.txt
```

## Public Demo Positioning

Use this wording in the Streamlit app description or project portfolio:

```text
BERTokenScope is deployed as an offline-safe Streamlit demo for reliable public access. The repository includes the production FastAPI backend, model-serving boundaries, observability, security, governance, and CI/CD scaffolding for enterprise deployment.
```

## Optional Backend Deployment Later

Deploy the FastAPI backend separately when you want live model-serving:

```powershell
$env:BERTSCOPE_API_KEY="replace-with-a-long-random-secret"
$env:BERTSCOPE_ALLOW_MODEL_DOWNLOADS="true"
pip install -r requirements.txt
pip install -r requirements-models.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

For containerized deployments:

```powershell
docker compose up --build
```

## Deployment Checklist

- Push the repository to GitHub.
- Confirm Streamlit Cloud uses `app/streamlit_app.py`.
- Confirm `requirements.txt` is the dashboard install file.
- Keep `BERTSCOPE_ALLOW_MODEL_DOWNLOADS=false` for public reliability.
- Link the GitHub repository from the portfolio page and Streamlit app description.
- Add the deployed Streamlit URL to the README after deployment.
