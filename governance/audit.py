from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ber_tokenscope.settings import get_settings


class AuditLogger:
    def __init__(self, path: str | Path | None = None) -> None:
        settings = get_settings()
        self.path = Path(path or settings.governance.audit_log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self, *, action: str, principal: str | None, details: dict[str, Any]
    ) -> None:
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "action": action,
            "principal": principal,
            "details": details,
        }
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, default=str) + "\n")

    def tail(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines[-limit:] if line.strip()]
