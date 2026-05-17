from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ModelSettings(BaseModel):
    masked_lm: str = "bert-base-uncased"
    sentence_embedding: str = "sentence-transformers/all-MiniLM-L6-v2"
    financial_sentiment: str = "ProsusAI/finbert"


class RuntimeSettings(BaseModel):
    top_k: int = Field(default=5, ge=1, le=20)
    max_sequence_length: int = Field(default=128, ge=8, le=512)
    attention_rollout_discard_ratio: float = Field(default=0.0, ge=0.0, le=0.9)


class ArtifactSettings(BaseModel):
    run_dir: str = "artifacts/runs"
    max_history: int = Field(default=100, ge=1, le=10000)


class DatabaseSettings(BaseModel):
    url: str = "sqlite:///artifacts/bertscope.db"


class ModelServingSettings(BaseModel):
    cache_dir: str = "artifacts/model-cache"
    allow_downloads: bool = False
    inference_timeout_seconds: int = Field(default=30, ge=1, le=3600)
    warmup_on_start: bool = False


class ServiceSettings(BaseModel):
    environment: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8788, ge=1, le=65535)
    dashboard_port: int = Field(default=8501, ge=1, le=65535)
    public_base_url: str = "http://localhost:8501"
    worker_concurrency: int = Field(default=1, ge=1, le=32)


class ApiKeySettings(BaseModel):
    key_id: str
    key_hash: str
    role: str = "analyst"


class AuthSettings(BaseModel):
    enabled: bool = True
    allow_test_bypass: bool = False
    api_keys: list[ApiKeySettings] = Field(default_factory=list)


class SecuritySettings(BaseModel):
    allowed_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:8501"]
    )
    max_request_bytes: int = Field(default=1_000_000, ge=1024)
    rate_limit_requests: int = Field(default=120, ge=1)
    rate_limit_window_seconds: int = Field(default=60, ge=1)
    enable_security_headers: bool = True


class GovernanceSettings(BaseModel):
    redact_artifacts: bool = True
    retention_days: int = Field(default=30, ge=1, le=3650)
    audit_log_path: str = "artifacts/audit/audit.jsonl"
    financial_disclaimer: str = (
        "BERTScope outputs are analytical model signals, not financial advice."
    )


class Settings(BaseModel):
    models: ModelSettings = Field(default_factory=ModelSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    artifacts: ArtifactSettings = Field(default_factory=ArtifactSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    model_serving: ModelServingSettings = Field(default_factory=ModelServingSettings)
    service: ServiceSettings = Field(default_factory=ServiceSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    governance: GovernanceSettings = Field(default_factory=GovernanceSettings)


@lru_cache(maxsize=1)
def get_settings(config_path: str | Path = "configs/settings.yaml") -> Settings:
    path = Path(config_path)
    if not path.exists():
        return Settings()

    with path.open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file) or {}
    settings = Settings.model_validate(raw)
    return apply_env_overrides(settings)


def apply_env_overrides(settings: Settings) -> Settings:
    if os.getenv("BERTSCOPE_ENVIRONMENT"):
        settings.service.environment = os.getenv("BERTSCOPE_ENVIRONMENT", "development")
    if os.getenv("BERTSCOPE_API_PORT"):
        settings.service.api_port = int(os.getenv("BERTSCOPE_API_PORT", "8788"))
    if os.getenv("BERTSCOPE_DASHBOARD_PORT"):
        settings.service.dashboard_port = int(
            os.getenv("BERTSCOPE_DASHBOARD_PORT", "8501")
        )
    if os.getenv("BERTSCOPE_PUBLIC_BASE_URL"):
        settings.service.public_base_url = os.getenv(
            "BERTSCOPE_PUBLIC_BASE_URL",
            settings.service.public_base_url,
        )
    if os.getenv("BERTSCOPE_DATABASE_URL"):
        settings.database.url = os.getenv(
            "BERTSCOPE_DATABASE_URL", settings.database.url
        )
    if os.getenv("BERTSCOPE_MODEL_CACHE_DIR"):
        settings.model_serving.cache_dir = os.getenv(
            "BERTSCOPE_MODEL_CACHE_DIR",
            settings.model_serving.cache_dir,
        )
    if os.getenv("BERTSCOPE_ALLOW_MODEL_DOWNLOADS") is not None:
        settings.model_serving.allow_downloads = (
            os.getenv("BERTSCOPE_ALLOW_MODEL_DOWNLOADS", "").lower() == "true"
        )
    if os.getenv("BERTSCOPE_INFERENCE_TIMEOUT_SECONDS"):
        settings.model_serving.inference_timeout_seconds = int(
            os.getenv("BERTSCOPE_INFERENCE_TIMEOUT_SECONDS", "30")
        )
    if os.getenv("BERTSCOPE_AUTH_ENABLED") is not None:
        settings.auth.enabled = (
            os.getenv("BERTSCOPE_AUTH_ENABLED", "").lower() == "true"
        )
    if os.getenv("BERTSCOPE_ALLOW_TEST_BYPASS") is not None:
        settings.auth.allow_test_bypass = (
            os.getenv("BERTSCOPE_ALLOW_TEST_BYPASS", "").lower() == "true"
        )
    api_key = os.getenv("BERTSCOPE_API_KEY")
    if api_key:
        from security.auth import hash_api_key

        settings.auth.api_keys = [
            ApiKeySettings(
                key_id=os.getenv("BERTSCOPE_API_KEY_ID", "env-admin"),
                key_hash=hash_api_key(api_key),
                role=os.getenv("BERTSCOPE_API_KEY_ROLE", "admin"),
            )
        ]
    if os.getenv("BERTSCOPE_ALLOWED_ORIGINS"):
        settings.security.allowed_origins = [
            origin.strip()
            for origin in os.getenv("BERTSCOPE_ALLOWED_ORIGINS", "").split(",")
            if origin.strip()
        ]
    if os.getenv("BERTSCOPE_MAX_REQUEST_BYTES"):
        settings.security.max_request_bytes = int(
            os.getenv("BERTSCOPE_MAX_REQUEST_BYTES", "1000000")
        )
    if os.getenv("BERTSCOPE_RATE_LIMIT_REQUESTS"):
        settings.security.rate_limit_requests = int(
            os.getenv("BERTSCOPE_RATE_LIMIT_REQUESTS", "120")
        )
    if os.getenv("BERTSCOPE_RATE_LIMIT_WINDOW_SECONDS"):
        settings.security.rate_limit_window_seconds = int(
            os.getenv("BERTSCOPE_RATE_LIMIT_WINDOW_SECONDS", "60")
        )
    if os.getenv("BERTSCOPE_REDACT_ARTIFACTS") is not None:
        settings.governance.redact_artifacts = (
            os.getenv("BERTSCOPE_REDACT_ARTIFACTS", "").lower() == "true"
        )
    if os.getenv("BERTSCOPE_RETENTION_DAYS"):
        settings.governance.retention_days = int(
            os.getenv("BERTSCOPE_RETENTION_DAYS", "30")
        )
    if os.getenv("BERTSCOPE_AUDIT_LOG_PATH"):
        settings.governance.audit_log_path = os.getenv(
            "BERTSCOPE_AUDIT_LOG_PATH",
            settings.governance.audit_log_path,
        )
    return settings
