import pytest
from fastapi import HTTPException

from ber_tokenscope.settings import ApiKeySettings, AuthSettings
from security.auth import authenticate_api_key, hash_api_key, role_allows


def test_hash_api_key_is_stable():
    assert hash_api_key("secret") == hash_api_key("secret")
    assert hash_api_key("secret") != "secret"


def test_authenticate_api_key_accepts_valid_key():
    settings = AuthSettings(
        enabled=True,
        api_keys=[
            ApiKeySettings(
                key_id="test-key",
                key_hash=hash_api_key("secret"),
                role="analyst",
            )
        ],
    )

    principal = authenticate_api_key("secret", settings)

    assert principal.key_id == "test-key"
    assert principal.role == "analyst"


def test_authenticate_api_key_rejects_invalid_key():
    settings = AuthSettings(
        enabled=True,
        api_keys=[
            ApiKeySettings(
                key_id="test-key",
                key_hash=hash_api_key("secret"),
                role="analyst",
            )
        ],
    )

    with pytest.raises(HTTPException):
        authenticate_api_key("wrong", settings)


def test_role_hierarchy():
    assert role_allows("admin", "viewer")
    assert role_allows("analyst", "viewer")
    assert not role_allows("viewer", "analyst")
