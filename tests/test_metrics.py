from observability.metrics import MetricsRegistry


def test_metrics_registry_records_summary_and_prometheus_text():
    registry = MetricsRegistry()
    registry.record_request(route="/health", status_code=200, latency_ms=10.0)
    registry.record_request(route="/broken", status_code=500, latency_ms=20.0)

    summary = registry.summary()
    text = registry.prometheus_text()

    assert summary.request_count == 2
    assert summary.error_count == 1
    assert summary.average_latency_ms == 15.0
    assert "bertscope_requests_total 2" in text
