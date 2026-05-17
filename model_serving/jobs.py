from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from ber_tokenscope.schemas import JobRecord


class JobManager:
    def __init__(self, max_workers: int = 2) -> None:
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.jobs: dict[str, JobRecord] = {}

    def submit(self, task: str, function) -> JobRecord:
        now = datetime.now(UTC).isoformat()
        job = JobRecord(
            job_id=uuid.uuid4().hex,
            task=task,
            status="queued",
            created_at=now,
            updated_at=now,
        )
        self.jobs[job.job_id] = job
        self.executor.submit(self._run, job.job_id, function)
        return job

    def get(self, job_id: str) -> JobRecord | None:
        return self.jobs.get(job_id)

    def _run(self, job_id: str, function) -> None:
        self._update(job_id, status="running")
        try:
            result = function()
            self._update(job_id, status="completed", result=result)
        except Exception as exc:
            self._update(job_id, status="failed", error=str(exc))

    def _update(
        self,
        job_id: str,
        *,
        status: str,
        result: dict | None = None,
        error: str | None = None,
    ) -> None:
        current = self.jobs[job_id]
        self.jobs[job_id] = current.model_copy(
            update={
                "status": status,
                "updated_at": datetime.now(UTC).isoformat(),
                "result": result,
                "error": error,
            }
        )
