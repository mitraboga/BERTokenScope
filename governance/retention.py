from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path


def cleanup_old_artifacts(run_dir: str | Path, retention_days: int) -> int:
    path = Path(run_dir)
    if not path.exists():
        return 0
    cutoff = datetime.now(UTC) - timedelta(days=retention_days)
    deleted = 0
    for artifact in path.glob("*.json"):
        modified = datetime.fromtimestamp(artifact.stat().st_mtime, tz=UTC)
        if modified < cutoff:
            artifact.unlink()
            deleted += 1
    return deleted
