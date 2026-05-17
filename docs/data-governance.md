# Data Governance

BERTokenScope includes baseline data governance controls for financial and sensitive text workflows.

## Sensitive Data Redaction

Stored run artifacts redact common sensitive patterns:

- email addresses
- phone numbers
- SSNs
- credit card-like numbers

Configure:

```text
BERTSCOPE_REDACT_ARTIFACTS=true
```

## Retention

Artifacts can be cleaned up by retention policy:

```text
BERTSCOPE_RETENTION_DAYS=30
POST /governance/retention/cleanup
```

## Audit Log

Run creation and retention cleanup events are written to:

```text
BERTSCOPE_AUDIT_LOG_PATH=artifacts/audit/audit.jsonl
```

Inspect audit events:

```text
GET /governance/audit
```

## Financial Disclaimer

BERTokenScope outputs are analytical model signals, not financial advice. The system is intended for explainability, research, and analyst-support workflows. Human review is required before decisions are made from model outputs.
