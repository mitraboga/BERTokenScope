from ber_tokenscope.schemas import RunRecord
from persistence.database import sqlite_path_from_url
from persistence.run_repository import RunRepository


def test_sqlite_path_from_url_parses_local_path():
    assert str(sqlite_path_from_url("sqlite:///artifacts/test.db")).endswith(
        "artifacts\\test.db"
    ) or str(sqlite_path_from_url("sqlite:///artifacts/test.db")).endswith(
        "artifacts/test.db"
    )


def test_run_repository_saves_lists_and_gets_records(tmp_path):
    repository = RunRepository(database_url=f"sqlite:///{tmp_path / 'runs.db'}")
    record = RunRecord(
        run_id="abc",
        task="test",
        status="completed",
        created_at="2026-05-16T00:00:00+00:00",
        duration_ms=1.2,
        artifact_path="artifact.json",
        summary={"ok": True},
    )

    repository.save(record)

    assert repository.get("abc") == record
    assert repository.list(limit=5)[0] == record


def test_run_repository_trims_records(tmp_path):
    repository = RunRepository(database_url=f"sqlite:///{tmp_path / 'runs.db'}")
    for index in range(3):
        repository.save(
            RunRecord(
                run_id=f"run-{index}",
                task="test",
                status="completed",
                created_at=f"2026-05-16T00:00:0{index}+00:00",
                duration_ms=1.0,
                artifact_path=f"{index}.json",
                summary={},
            )
        )

    repository.trim(2)

    assert len(repository.list(limit=10)) == 2


def test_run_repository_supports_offset(tmp_path):
    repository = RunRepository(database_url=f"sqlite:///{tmp_path / 'runs.db'}")
    for index in range(3):
        repository.save(
            RunRecord(
                run_id=f"run-{index}",
                task="test",
                status="completed",
                created_at=f"2026-05-16T00:00:0{index}+00:00",
                duration_ms=1.0,
                artifact_path=f"{index}.json",
                summary={},
            )
        )

    assert repository.list(limit=1, offset=1)[0].run_id == "run-1"
