from fastapi import FastAPI
from fastapi.testclient import TestClient

from ber_tokenscope.settings import SecuritySettings
from security.middleware import (
    RateLimitMiddleware,
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
)


def test_security_headers_are_applied():
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware, enabled=True)

    @app.get("/")
    def root():
        return {"ok": True}

    response = TestClient(app).get("/")

    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"


def test_request_size_limit_rejects_large_payload():
    app = FastAPI()
    app.add_middleware(RequestSizeLimitMiddleware, max_request_bytes=10)

    @app.post("/")
    def root():
        return {"ok": True}

    response = TestClient(app).post("/", content="x" * 20)

    assert response.status_code == 413


def test_rate_limit_rejects_after_limit():
    app = FastAPI()
    app.add_middleware(
        RateLimitMiddleware,
        settings=SecuritySettings(rate_limit_requests=1, rate_limit_window_seconds=60),
    )

    @app.get("/")
    def root():
        return {"ok": True}

    client = TestClient(app)
    assert client.get("/").status_code == 200
    assert client.get("/").status_code == 429
