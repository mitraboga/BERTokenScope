import os
import time

from governance.audit import AuditLogger
from governance.redaction import redact_payload, redact_text
from governance.retention import cleanup_old_artifacts


def test_redact_text_masks_sensitive_patterns():
    text = "Contact jane@example.com at 555-123-4567 with SSN 123-45-6789."

    redacted = redact_text(text)

    assert "jane@example.com" not in redacted
    assert "555-123-4567" not in redacted
    assert "123-45-6789" not in redacted
    assert "[REDACTED_EMAIL]" in redacted


def test_redact_payload_recurses_nested_values():
    payload = {"text": "Email jane@example.com", "items": ["Call 555-123-4567"]}

    redacted = redact_payload(payload)

    assert redacted["text"] == "Email [REDACTED_EMAIL]"
    assert redacted["items"][0] == "Call [REDACTED_PHONE]"


def test_audit_logger_writes_jsonl_events(tmp_path):
    logger = AuditLogger(path=tmp_path / "audit.jsonl")

    logger.log(action="test", principal="user", details={"ok": True})

    events = logger.tail()
    assert events[0]["action"] == "test"
    assert events[0]["principal"] == "user"


def test_cleanup_old_artifacts_deletes_expired_files(tmp_path):
    old_file = tmp_path / "old.json"
    old_file.write_text("{}", encoding="utf-8")
    old_timestamp = time.time() - (3 * 24 * 60 * 60)
    os.utime(old_file, (old_timestamp, old_timestamp))

    deleted = cleanup_old_artifacts(tmp_path, retention_days=1)

    assert deleted == 1
    assert not old_file.exists()
