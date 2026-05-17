from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ber_tokenscope.schemas import RunRecord
from ber_tokenscope.settings import get_settings
from governance.audit import AuditLogger
from governance.redaction import redact_payload
from persistence.run_repository import RunRepository


class RunTracker:
    """File-backed experiment and artifact tracker."""

    def __init__(
        self,
        run_dir: str | Path | None = None,
        max_history: int | None = None,
        database_url: str | None = None,
    ) -> None:
        settings = get_settings()
        self.settings = settings
        self.run_dir = Path(run_dir or settings.artifacts.run_dir)
        self.max_history = max_history or settings.artifacts.max_history
        self.repository = RunRepository(database_url=database_url)
        self.audit_logger = AuditLogger()
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def track(
        self,
        *,
        task: str,
        request: BaseModel | dict[str, Any],
        result: BaseModel | dict[str, Any],
        started_at: float,
        status: str = "completed",
        summary: dict[str, str | int | float | bool | None] | None = None,
    ) -> RunRecord:
        run_id = uuid.uuid4().hex
        duration_ms = round((time.perf_counter() - started_at) * 1000, 3)
        created_at = datetime.now(UTC).isoformat()
        artifact_path = self.run_dir / f"{run_id}.json"
        request_payload = serialize(request)
        result_payload = serialize(result)
        if self.settings.governance.redact_artifacts:
            request_payload = redact_payload(request_payload)
            result_payload = redact_payload(result_payload)
        payload = {
            "run_id": run_id,
            "task": task,
            "status": status,
            "created_at": created_at,
            "duration_ms": duration_ms,
            "request": request_payload,
            "result": result_payload,
            "summary": summary or {},
            "disclaimer": self.settings.governance.financial_disclaimer,
        }
        artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        record = RunRecord(
            run_id=run_id,
            task=task,
            status=status,
            created_at=created_at,
            duration_ms=duration_ms,
            artifact_path=str(artifact_path),
            summary=summary or {},
        )
        self.repository.save(record)
        self.repository.trim(self.max_history)
        self.audit_logger.log(
            action="run_created",
            principal=str((summary or {}).get("principal")) if summary else None,
            details={"run_id": run_id, "task": task, "status": status},
        )
        return record

    def list_runs(self, limit: int = 20, offset: int = 0) -> list[RunRecord]:
        return self.repository.list(limit=limit, offset=offset)

    def read_artifact(self, run_id: str) -> dict[str, Any]:
        path = self.run_dir / f"{run_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"No artifact exists for run {run_id}.")
        return json.loads(path.read_text(encoding="utf-8"))


def serialize(value: BaseModel | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value
