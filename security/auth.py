from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, status

from ber_tokenscope.settings import AuthSettings, get_settings

ROLE_ORDER = {
    "viewer": 1,
    "analyst": 2,
    "admin": 3,
}


@dataclass(frozen=True)
class Principal:
    key_id: str
    role: str


def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def role_allows(actual: str, required: str) -> bool:
    return ROLE_ORDER.get(actual, 0) >= ROLE_ORDER.get(required, 999)


def authenticate_api_key(
    x_api_key: str | None,
    auth_settings: AuthSettings | None = None,
) -> Principal:
    settings = auth_settings or get_settings().auth
    if not settings.enabled:
        return Principal(key_id="auth-disabled", role="admin")
    if settings.allow_test_bypass and os.getenv("PYTEST_CURRENT_TEST"):
        return Principal(key_id="pytest-bypass", role="admin")
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key.",
        )

    candidate_hash = hash_api_key(x_api_key)
    for key in settings.api_keys:
        if hmac.compare_digest(candidate_hash, key.key_hash):
            return Principal(key_id=key.key_id, role=key.role)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key.",
    )


def require_role(required_role: str):
    def dependency(x_api_key: str | None = Header(default=None)) -> Principal:
        principal = authenticate_api_key(x_api_key)
        if not role_allows(principal.role, required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient role for this operation.",
            )
        return principal

    return dependency


RequireViewer = Depends(require_role("viewer"))
RequireAnalyst = Depends(require_role("analyst"))
RequireAdmin = Depends(require_role("admin"))
