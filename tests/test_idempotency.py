from persistence.idempotency_repository import IdempotencyRepository


def test_idempotency_repository_saves_and_reads_response(tmp_path):
    repository = IdempotencyRepository(database_url=f"sqlite:///{tmp_path / 'idem.db'}")

    repository.save(
        key="abc",
        task="test",
        request_hash="hash",
        response={"ok": True},
    )
    record = repository.get("abc")

    assert record is not None
    assert record["request_hash"] == "hash"
    assert record["response"] == {"ok": True}
