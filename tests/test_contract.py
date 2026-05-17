import pytest
from fastapi.testclient import TestClient

from api.main import app

pytestmark = pytest.mark.contract


def test_openapi_includes_enterprise_endpoints():
    schema = TestClient(app).get("/openapi.json").json()
    paths = schema["paths"]

    for path in [
        "/api/v1/financial-analysis",
        "/api/v1/models/status",
        "/api/v1/observability/summary",
        "/api/v1/runs",
        "/metrics",
    ]:
        assert path in paths


def test_v1_financial_response_contract():
    response = TestClient(app).post(
        "/api/v1/financial-analysis",
        json={"text": "Revenue growth was strong, but inflation risk remains."},
        headers={"X-API-Key": "test-api-key"},
    )
    payload = response.json()

    assert response.status_code == 200
    assert {"text", "sentiment", "risk_score", "matched_terms", "model_name"} <= set(
        payload
    )
    assert {"label", "score", "explanation"} <= set(payload["sentiment"])


def test_error_envelope_contract():
    response = TestClient(app).post(
        "/api/v1/financial-analysis",
        json={},
        headers={"X-API-Key": "test-api-key"},
    )

    assert response.status_code == 422
    assert set(response.json()) == {"error"}
    assert {"code", "message", "request_id"} <= set(response.json()["error"])
