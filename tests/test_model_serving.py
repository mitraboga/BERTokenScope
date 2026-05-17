from model_serving.jobs import JobManager
from model_serving.registry import MODEL_REGISTRY, registry_by_id
from model_serving.runtime import ModelServingRuntime, run_with_timeout


def test_registry_contains_core_models():
    registry = registry_by_id()

    assert "bert-base-uncased" in registry
    assert "ProsusAI/finbert" in registry
    assert MODEL_REGISTRY


def test_model_runtime_reports_statuses():
    statuses = ModelServingRuntime().statuses()

    assert statuses
    assert all(status.model_id for status in statuses)


def test_model_runtime_warmup_uses_fallback_when_uncached():
    result = ModelServingRuntime().warmup("bert-base-uncased")

    assert result.status in {"fallback-ready", "ready"}


def test_run_with_timeout_returns_value():
    assert run_with_timeout(lambda: "ok", timeout_seconds=1) == "ok"


def test_job_manager_completes_job():
    manager = JobManager(max_workers=1)
    job = manager.submit("test", lambda: {"ok": True})

    for _ in range(100):
        current = manager.get(job.job_id)
        if current and current.status == "completed":
            break

    current = manager.get(job.job_id)
    assert current is not None
    assert current.status == "completed"
    assert current.result == {"ok": True}
