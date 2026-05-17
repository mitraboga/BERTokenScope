from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path

from ber_tokenscope.schemas import ModelStatus, ModelWarmupResult
from ber_tokenscope.settings import get_settings
from model_serving.registry import MODEL_REGISTRY, registry_by_id


class InferenceTimeoutError(RuntimeError):
    """Raised when an inference or warmup operation exceeds its configured timeout."""


def run_with_timeout(function, timeout_seconds: int):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(function)
        try:
            return future.result(timeout=timeout_seconds)
        except TimeoutError as exc:
            raise InferenceTimeoutError(
                f"Operation exceeded timeout of {timeout_seconds} seconds."
            ) from exc


class ModelServingRuntime:
    def __init__(self) -> None:
        self.settings = get_settings()

    @property
    def cache_dir(self) -> Path:
        path = Path(self.settings.model_serving.cache_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def statuses(self) -> list[ModelStatus]:
        return [self.status(entry.model_id) for entry in MODEL_REGISTRY]

    def status(self, model_id: str) -> ModelStatus:
        entry = registry_by_id().get(model_id)
        if entry is None:
            raise ValueError(f"Unknown model_id: {model_id}")
        cache_path = self.cache_dir / model_id.replace("/", "--")
        cached = cache_path.exists()
        notes = []
        if not cached:
            notes.append("Model weights are not present in the local cache.")
        if not self.settings.model_serving.allow_downloads:
            notes.append(
                "Live downloads are disabled; fallback mode will be used if needed."
            )
        return ModelStatus(
            model_id=entry.model_id,
            display_name=entry.display_name,
            task=entry.task,
            cached=cached,
            cache_path=str(cache_path),
            serving_mode="live" if cached else f"fallback:{entry.fallback}",
            notes=notes,
        )

    def warmup(self, model_id: str) -> ModelWarmupResult:
        started_at = time.perf_counter()
        status = self.status(model_id)
        if not status.cached and not self.settings.model_serving.allow_downloads:
            return ModelWarmupResult(
                model_id=model_id,
                status="fallback-ready",
                duration_ms=round((time.perf_counter() - started_at) * 1000, 3),
                message="Model is not cached and downloads are disabled; fallback is ready.",
            )

        def warmup_operation():
            # The actual heavy load remains delegated to task-specific adapters.
            return self.status(model_id)

        run_with_timeout(
            warmup_operation,
            timeout_seconds=self.settings.model_serving.inference_timeout_seconds,
        )
        return ModelWarmupResult(
            model_id=model_id,
            status="ready",
            duration_ms=round((time.perf_counter() - started_at) * 1000, 3),
            message="Model serving status checked successfully.",
        )
