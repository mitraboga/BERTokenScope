import uuid

from fastapi.testclient import TestClient

from api.main import app

HEADERS = {"X-API-Key": "test-api-key"}


def test_health_endpoint():
    response = TestClient(app).get("/health", headers={"X-Request-ID": "req-test"})

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "environment" in response.json()
    assert response.headers["X-Request-ID"] == "req-test"


def test_ready_endpoint():
    response = TestClient(app).get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.headers["X-Frame-Options"] == "DENY"


def test_cors_preflight_allows_configured_origin():
    response = TestClient(app).options(
        "/financial-analysis",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200


def test_financial_analysis_endpoint():
    response = TestClient(app).post(
        "/financial-analysis",
        json={"text": "Revenue growth was strong, but inflation risk remains."},
        headers=HEADERS,
    )

    assert response.status_code == 200
    assert response.json()["matched_terms"]["risk"] == ["inflation", "risk"]


def test_attention_explore_endpoint():
    response = TestClient(app).post(
        "/attention/explore",
        json={
            "text": "The bank was muddy because the river overflowed near the [MASK].",
            "layer": 1,
            "head": 1,
        },
        headers=HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_layer"] == 1
    assert payload["selected_head"] == 1
    assert payload["head_summaries"]


def test_financial_transcript_endpoint():
    response = TestClient(app).post(
        "/financial-analysis/transcript",
        json={
            "company": "Example Corp",
            "documents": [
                {
                    "period": "Q1",
                    "text": "Revenue growth was strong and demand improved.",
                },
                {
                    "period": "Q2",
                    "text": "Inflation risk and uncertain guidance created headwinds.",
                },
            ],
            "max_words": 40,
        },
        headers=HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["company"] == "Example Corp"
    assert len(payload["tone_drift"]) == 2


def test_embeddings_endpoint():
    response = TestClient(app).post(
        "/embeddings/analyze",
        json={
            "documents": [
                {"label": "A", "text": "revenue growth demand"},
                {"label": "B", "text": "strong revenue growth"},
                {"label": "C", "text": "litigation risk recession"},
            ],
            "cluster_count": 2,
        },
        headers=HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["points"]) == 3
    assert payload["similarities"]


def test_model_comparison_endpoint():
    response = TestClient(app).post(
        "/models/compare",
        json={
            "masked_text": "The company reported record [MASK] this quarter.",
            "financial_text": "Revenue growth was strong, but inflation risk remains.",
            "documents": [
                {"label": "A", "text": "revenue growth demand"},
                {"label": "B", "text": "strong revenue growth"},
                {"label": "C", "text": "litigation risk recession"},
            ],
        },
        headers=HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["masked_language"]["runs"]) == 3
    assert len(payload["financial_sentiment"]["runs"]) == 2
    assert len(payload["embeddings"]["runs"]) == 2


def test_explainability_endpoint():
    response = TestClient(app).post(
        "/explainability/analyze",
        json={
            "task": "financial-sentiment",
            "text": "Revenue growth was strong, but inflation risk remains.",
        },
        headers=HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["task"] == "financial-sentiment"
    assert payload["token_attributions"]


def test_runs_endpoint_lists_tracked_runs():
    TestClient(app).post(
        "/financial-analysis",
        json={"text": "Revenue growth was strong, but inflation risk remains."},
        headers=HEADERS,
    )

    response = TestClient(app).get("/runs?limit=5", headers=HEADERS)

    assert response.status_code == 200
    assert response.json()


def test_model_registry_endpoint():
    response = TestClient(app).get("/models/registry", headers=HEADERS)

    assert response.status_code == 200
    assert response.json()


def test_model_status_endpoint():
    response = TestClient(app).get("/models/status", headers=HEADERS)

    assert response.status_code == 200
    assert response.json()


def test_model_warmup_endpoint():
    response = TestClient(app).post(
        "/models/warmup",
        json={"model_id": "bert-base-uncased"},
        headers=HEADERS,
    )

    assert response.status_code == 200
    assert response.json()["status"] in {"fallback-ready", "ready"}


def test_observability_summary_endpoint():
    TestClient(app).get("/health")

    response = TestClient(app).get("/observability/summary", headers=HEADERS)

    assert response.status_code == 200
    assert response.json()["request_count"] >= 1


def test_metrics_endpoint_exposes_prometheus_text():
    response = TestClient(app).get("/metrics")

    assert response.status_code == 200
    assert "bertscope_requests_total" in response.text


def test_governance_status_endpoint():
    response = TestClient(app).get("/api/v1/governance/status", headers=HEADERS)

    assert response.status_code == 200
    assert response.json()["redact_artifacts"] is True


def test_governance_audit_endpoint_requires_admin_and_returns_events():
    TestClient(app).post(
        "/api/v1/financial-analysis",
        json={"text": "Email jane@example.com and report revenue growth."},
        headers=HEADERS,
    )

    response = TestClient(app).get("/api/v1/governance/audit", headers=HEADERS)

    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_protected_endpoint_requires_api_key():
    response = TestClient(app).post(
        "/financial-analysis",
        json={"text": "Revenue growth was strong."},
    )

    assert response.status_code == 401


def test_protected_endpoint_rejects_invalid_api_key():
    response = TestClient(app).post(
        "/financial-analysis",
        json={"text": "Revenue growth was strong."},
        headers={"X-API-Key": "wrong"},
    )

    assert response.status_code == 401


def test_v1_financial_analysis_endpoint():
    response = TestClient(app).post(
        "/api/v1/financial-analysis",
        json={"text": "Revenue growth was strong, but inflation risk remains."},
        headers=HEADERS,
    )

    assert response.status_code == 200
    assert response.json()["sentiment"]["label"] in {"positive", "neutral", "negative"}


def test_v1_idempotency_replays_same_response():
    key = f"idem-{uuid.uuid4().hex}"
    headers = {**HEADERS, "X-Idempotency-Key": key}
    payload = {"text": "Revenue growth was strong, but inflation risk remains."}

    first = TestClient(app).post(
        "/api/v1/financial-analysis", json=payload, headers=headers
    )
    second = TestClient(app).post(
        "/api/v1/financial-analysis", json=payload, headers=headers
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()


def test_v1_idempotency_conflict_for_different_request():
    key = f"idem-{uuid.uuid4().hex}"
    headers = {**HEADERS, "X-Idempotency-Key": key}

    first = TestClient(app).post(
        "/api/v1/financial-analysis",
        json={"text": "Revenue growth was strong."},
        headers=headers,
    )
    second = TestClient(app).post(
        "/api/v1/financial-analysis",
        json={"text": "Inflation risk remains."},
        headers=headers,
    )

    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "http_error"


def test_v1_runs_endpoint_is_paginated():
    response = TestClient(app).get("/api/v1/runs?limit=2&offset=0", headers=HEADERS)

    assert response.status_code == 200
    payload = response.json()
    assert payload["limit"] == 2
    assert payload["offset"] == 0
    assert "items" in payload


def test_standard_error_envelope_for_auth_failure():
    response = TestClient(app).get("/api/v1/runs")

    assert response.status_code == 401
    assert response.json()["error"]["message"] == "Missing API key."


def test_standard_error_envelope_for_validation_failure():
    response = TestClient(app).post(
        "/api/v1/financial-analysis",
        json={},
        headers=HEADERS,
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
