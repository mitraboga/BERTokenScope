from __future__ import annotations

from collections import Counter, defaultdict
from threading import Lock

from ber_tokenscope.schemas import ObservabilitySummary


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self.request_count = 0
        self.error_count = 0
        self.total_latency_ms = 0.0
        self.routes: Counter[str] = Counter()
        self.statuses: Counter[str] = Counter()
        self.route_latency_ms: defaultdict[str, float] = defaultdict(float)

    def record_request(
        self, *, route: str, status_code: int, latency_ms: float
    ) -> None:
        with self._lock:
            self.request_count += 1
            self.total_latency_ms += latency_ms
            self.routes[route] += 1
            self.statuses[str(status_code)] += 1
            self.route_latency_ms[route] += latency_ms
            if status_code >= 500:
                self.error_count += 1

    def summary(self) -> ObservabilitySummary:
        with self._lock:
            average = (
                self.total_latency_ms / self.request_count
                if self.request_count
                else 0.0
            )
            return ObservabilitySummary(
                request_count=self.request_count,
                error_count=self.error_count,
                average_latency_ms=round(average, 3),
                routes=dict(self.routes),
                statuses=dict(self.statuses),
            )

    def prometheus_text(self) -> str:
        with self._lock:
            lines = [
                "# HELP bertscope_requests_total Total HTTP requests.",
                "# TYPE bertscope_requests_total counter",
                f"bertscope_requests_total {self.request_count}",
                "# HELP bertscope_errors_total Total HTTP 5xx responses.",
                "# TYPE bertscope_errors_total counter",
                f"bertscope_errors_total {self.error_count}",
                "# HELP bertscope_request_latency_ms_total Total request latency in milliseconds.",
                "# TYPE bertscope_request_latency_ms_total counter",
                f"bertscope_request_latency_ms_total {round(self.total_latency_ms, 3)}",
            ]
            for route, count in sorted(self.routes.items()):
                safe_route = route.replace("\\", "\\\\").replace('"', '\\"')
                lines.append(
                    f'bertscope_route_requests_total{{route="{safe_route}"}} {count}'
                )
            for status, count in sorted(self.statuses.items()):
                lines.append(
                    f'bertscope_status_responses_total{{status="{status}"}} {count}'
                )
            return "\n".join(lines) + "\n"
