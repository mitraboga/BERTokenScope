# Security

BERTokenScope includes baseline production security controls.

## Authentication

Protected API endpoints require:

```text
X-API-Key: <key>
```

Roles:

- `viewer`
- `analyst`
- `admin`

## HTTP Hardening

The API applies:

- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Referrer-Policy: no-referrer`
- restrictive permissions policy
- baseline content security policy

## CORS

Configure allowed origins:

```text
BERTSCOPE_ALLOWED_ORIGINS=http://localhost:8501
```

## Request Limits

Configure max request size:

```text
BERTSCOPE_MAX_REQUEST_BYTES=1000000
```

## Rate Limiting

Configure per-client request limits:

```text
BERTSCOPE_RATE_LIMIT_REQUESTS=120
BERTSCOPE_RATE_LIMIT_WINDOW_SECONDS=60
```

## Error Handling

Unhandled exceptions return a generic 500 response while details are logged server-side.
