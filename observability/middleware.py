from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from observability.metrics import MetricsRegistry


class ObservabilityMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, metrics: MetricsRegistry) -> None:
        super().__init__(app)
        self.metrics = metrics
        self.logger = logging.getLogger("bertscope.requests")

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
        request.state.request_id = request_id
        started_at = time.perf_counter()
        status_code = 500
        response = None
        try:
            response = await call_next(request)
            status_code = response.status_code
        finally:
            latency_ms = round((time.perf_counter() - started_at) * 1000, 3)
            route = request.url.path
            self.metrics.record_request(
                route=route,
                status_code=status_code,
                latency_ms=latency_ms,
            )
            self.logger.info(
                "http_request",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": route,
                    "status_code": status_code,
                    "latency_ms": latency_ms,
                },
            )
            if response is not None:
                response.headers["X-Request-ID"] = request_id
        return response
