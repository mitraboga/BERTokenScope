from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from fastapi import APIRouter, FastAPI, Header, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from attention.engine import AttentionEngine
from ber_tokenscope import __version__
from ber_tokenscope.schemas import (
    AttentionExplorerResult,
    EmbeddingDocument,
    EmbeddingReport,
    ExplainabilityReport,
    FinancialAnalysis,
    FinancialTranscriptReport,
    GovernanceStatus,
    JobRecord,
    MaskedLMResult,
    ModelComparisonReport,
    ModelRegistryEntry,
    ModelStatus,
    ModelWarmupResult,
    ObservabilitySummary,
    PaginatedRuns,
    RetentionCleanupResult,
    RunRecord,
)
from ber_tokenscope.settings import get_settings
from embeddings.engine import EmbeddingEngine
from explainability.engine import ExplainabilityEngine
from financial_nlp.sentiment import FinancialTextAnalyzer
from financial_nlp.transcript_analysis import PeriodDocument, TranscriptAnalyzer
from governance.audit import AuditLogger
from governance.retention import cleanup_old_artifacts
from model_comparison.engine import ModelComparisonEngine
from model_serving.jobs import JobManager
from model_serving.registry import MODEL_REGISTRY
from model_serving.runtime import ModelServingRuntime
from observability.logging import configure_logging
from observability.metrics import MetricsRegistry
from observability.middleware import ObservabilityMiddleware
from observability.tracker import RunTracker
from persistence.idempotency_repository import IdempotencyRepository
from security.auth import Principal, RequireAdmin, RequireAnalyst, RequireViewer
from security.errors import (
    http_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from security.middleware import (
    RateLimitMiddleware,
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
)

configure_logging()
metrics_registry = MetricsRegistry()
settings = get_settings()
app = FastAPI(
    title="BERTScope API",
    version=__version__,
    description="Transformer explainability and financial NLP intelligence service.",
)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "X-Request-ID", "Content-Type"],
)
app.add_middleware(
    SecurityHeadersMiddleware, enabled=settings.security.enable_security_headers
)
app.add_middleware(
    RequestSizeLimitMiddleware, max_request_bytes=settings.security.max_request_bytes
)
app.add_middleware(RateLimitMiddleware, settings=settings.security)
app.add_middleware(ObservabilityMiddleware, metrics=metrics_registry)

attention_engine = AttentionEngine()
financial_analyzer = FinancialTextAnalyzer()
transcript_analyzer = TranscriptAnalyzer()
embedding_engine = EmbeddingEngine()
comparison_engine = ModelComparisonEngine()
explainability_engine = ExplainabilityEngine()
run_tracker = RunTracker()
idempotency_repository = IdempotencyRepository()
model_runtime = ModelServingRuntime()
job_manager = JobManager()
audit_logger = AuditLogger()


class MaskRequest(BaseModel):
    text: str = Field(examples=["The company reported record [MASK] this quarter."])
    top_k: int = Field(default=5, ge=1, le=20)


class FinancialRequest(BaseModel):
    text: str


class PeriodDocumentRequest(BaseModel):
    period: str = Field(examples=["Q1 2026"])
    text: str


class TranscriptReportRequest(BaseModel):
    company: str | None = Field(default=None, examples=["Example Corp"])
    documents: list[PeriodDocumentRequest]
    max_words: int = Field(default=180, ge=40, le=400)


class AttentionRequest(BaseModel):
    text: str = Field(
        examples=["The bank was muddy because the river overflowed near the [MASK]."]
    )
    layer: int = Field(default=1, ge=1)
    head: int = Field(default=1, ge=1)
    export_diagram: bool = False


class EmbeddingRequest(BaseModel):
    documents: list[EmbeddingDocument]
    cluster_count: int = Field(default=3, ge=1, le=12)
    similarity_limit: int = Field(default=10, ge=1, le=50)


class ModelComparisonRequest(BaseModel):
    masked_text: str = Field(default="The company reported record [MASK] this quarter.")
    financial_text: str = Field(
        default=(
            "Management reported strong revenue growth, but noted inflation risk "
            "and uncertain guidance."
        )
    )
    documents: list[EmbeddingDocument]
    top_k: int = Field(default=5, ge=1, le=10)


class ExplainabilityRequest(BaseModel):
    text: str
    task: str = Field(
        default="financial-sentiment", pattern="^(financial-sentiment|masked-language)$"
    )
    layer: int = Field(default=1, ge=1)
    head: int = Field(default=1, ge=1)


class ModelWarmupRequest(BaseModel):
    model_id: str
    async_job: bool = False


def request_fingerprint(task: str, request: BaseModel) -> str:
    payload = {
        "task": task,
        "request": request.model_dump(mode="json"),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def idempotent_execute(
    *,
    task: str,
    request: BaseModel,
    idempotency_key: str | None,
    operation,
):
    if not idempotency_key:
        return operation()
    fingerprint = request_fingerprint(task, request)
    existing = idempotency_repository.get(idempotency_key)
    if existing:
        if existing["request_hash"] != fingerprint:
            raise HTTPException(
                status_code=409,
                detail="Idempotency key was already used with a different request.",
            )
        return existing["response"]
    result = operation()
    if isinstance(result, BaseModel):
        response = result.model_dump(mode="json")
    else:
        response = result
    idempotency_repository.save(
        key=idempotency_key,
        task=task,
        request_hash=fingerprint,
        response=response,
    )
    return result


@app.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    return {
        "status": "ok",
        "version": __version__,
        "environment": settings.service.environment,
    }


@app.get("/ready")
def ready() -> dict[str, str]:
    return {"status": "ready"}


@app.get("/observability/summary", response_model=ObservabilitySummary)
def observability_summary(principal: Principal = RequireViewer) -> ObservabilitySummary:
    return metrics_registry.summary()


@app.get("/metrics")
def metrics() -> str:
    from fastapi.responses import PlainTextResponse

    return PlainTextResponse(metrics_registry.prometheus_text())


@app.get("/governance/status", response_model=GovernanceStatus)
def governance_status(principal: Principal = RequireViewer) -> GovernanceStatus:
    settings = get_settings()
    return GovernanceStatus(
        redact_artifacts=settings.governance.redact_artifacts,
        retention_days=settings.governance.retention_days,
        audit_log_path=settings.governance.audit_log_path,
        financial_disclaimer=settings.governance.financial_disclaimer,
    )


@app.get("/governance/audit")
def governance_audit(
    limit: int = 50, principal: Principal = RequireAdmin
) -> list[dict]:
    return audit_logger.tail(limit=limit)


@app.post("/governance/retention/cleanup", response_model=RetentionCleanupResult)
def governance_retention_cleanup(
    principal: Principal = RequireAdmin,
) -> RetentionCleanupResult:
    settings = get_settings()
    deleted = cleanup_old_artifacts(
        settings.artifacts.run_dir,
        settings.governance.retention_days,
    )
    audit_logger.log(
        action="retention_cleanup",
        principal=principal.key_id,
        details={"deleted_artifacts": deleted},
    )
    return RetentionCleanupResult(
        deleted_artifacts=deleted,
        retention_days=settings.governance.retention_days,
    )


@app.get("/models/registry", response_model=list[ModelRegistryEntry])
def list_model_registry(
    principal: Principal = RequireViewer,
) -> list[ModelRegistryEntry]:
    return MODEL_REGISTRY


@app.get("/models/status", response_model=list[ModelStatus])
def list_model_status(principal: Principal = RequireViewer) -> list[ModelStatus]:
    return model_runtime.statuses()


@app.post("/models/warmup", response_model=ModelWarmupResult | JobRecord)
def warmup_model(
    request: ModelWarmupRequest,
    principal: Principal = RequireAnalyst,
) -> ModelWarmupResult | JobRecord:
    if request.async_job:
        return job_manager.submit(
            "model-warmup",
            lambda: model_runtime.warmup(request.model_id).model_dump(mode="json"),
        )
    return model_runtime.warmup(request.model_id)


@app.get("/jobs/{job_id}", response_model=JobRecord)
def get_job(job_id: str, principal: Principal = RequireViewer) -> JobRecord:
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.post("/masked-token", response_model=MaskedLMResult)
def masked_token(
    request: MaskRequest, principal: Principal = RequireAnalyst
) -> MaskedLMResult:
    started_at = time.perf_counter()
    result = attention_engine.analyze_mask(request.text, top_k=request.top_k)
    run_tracker.track(
        task="masked-token",
        request=request,
        result=result,
        started_at=started_at,
        summary={
            "principal": principal.key_id,
            "model_name": result.model_name,
            "prediction_count": len(result.predictions),
        },
    )
    return result


@app.post("/financial-analysis", response_model=FinancialAnalysis)
def financial_analysis(
    request: FinancialRequest,
    principal: Principal = RequireAnalyst,
) -> FinancialAnalysis:
    started_at = time.perf_counter()
    result = financial_analyzer.analyze(request.text)
    run_tracker.track(
        task="financial-analysis",
        request=request,
        result=result,
        started_at=started_at,
        summary={
            "principal": principal.key_id,
            "model_name": result.model_name,
            "sentiment": result.sentiment.label,
            "confidence": result.sentiment.score,
        },
    )
    return result


@app.post("/financial-analysis/transcript", response_model=FinancialTranscriptReport)
def financial_transcript_report(
    request: TranscriptReportRequest,
    principal: Principal = RequireAnalyst,
) -> FinancialTranscriptReport:
    started_at = time.perf_counter()
    documents = [
        PeriodDocument(period=document.period, text=document.text)
        for document in request.documents
    ]
    result = transcript_analyzer.analyze_report(
        documents,
        company=request.company,
        max_words=request.max_words,
    )
    run_tracker.track(
        task="financial-transcript",
        request=request,
        result=result,
        started_at=started_at,
        summary={
            "principal": principal.key_id,
            "company": result.company,
            "period_count": len(result.periods),
        },
    )
    return result


@app.post("/attention/explore", response_model=AttentionExplorerResult)
def explore_attention(
    request: AttentionRequest,
    principal: Principal = RequireAnalyst,
) -> AttentionExplorerResult:
    started_at = time.perf_counter()
    result = attention_engine.explore_attention(
        request.text,
        layer=request.layer,
        head=request.head,
        export_diagram=request.export_diagram,
    )
    run_tracker.track(
        task="attention-explore",
        request=request,
        result=result,
        started_at=started_at,
        summary={
            "principal": principal.key_id,
            "model_name": result.model_name,
            "layer": result.selected_layer,
            "head": result.selected_head,
        },
    )
    return result


@app.post("/embeddings/analyze", response_model=EmbeddingReport)
def analyze_embeddings(
    request: EmbeddingRequest,
    principal: Principal = RequireAnalyst,
) -> EmbeddingReport:
    started_at = time.perf_counter()
    result = embedding_engine.analyze(
        request.documents,
        cluster_count=request.cluster_count,
        similarity_limit=request.similarity_limit,
    )
    run_tracker.track(
        task="embeddings-analyze",
        request=request,
        result=result,
        started_at=started_at,
        summary={
            "principal": principal.key_id,
            "model_name": result.model_name,
            "document_count": len(result.points),
        },
    )
    return result


@app.post("/models/compare", response_model=ModelComparisonReport)
def compare_models(
    request: ModelComparisonRequest,
    principal: Principal = RequireAnalyst,
) -> ModelComparisonReport:
    started_at = time.perf_counter()
    result = comparison_engine.compare(
        masked_text=request.masked_text,
        financial_text=request.financial_text,
        documents=request.documents,
        top_k=request.top_k,
    )
    run_tracker.track(
        task="models-compare",
        request=request,
        result=result,
        started_at=started_at,
        summary={
            "principal": principal.key_id,
            "masked_runs": len(result.masked_language.runs),
            "financial_runs": len(result.financial_sentiment.runs),
            "embedding_runs": len(result.embeddings.runs),
        },
    )
    return result


@app.post("/explainability/analyze", response_model=ExplainabilityReport)
def explain_prediction(
    request: ExplainabilityRequest,
    principal: Principal = RequireAnalyst,
) -> ExplainabilityReport:
    started_at = time.perf_counter()
    if request.task == "masked-language":
        result = explainability_engine.explain_masked_language(
            request.text,
            layer=request.layer,
            head=request.head,
        )
    else:
        result = explainability_engine.explain_financial_text(request.text)
    run_tracker.track(
        task="explainability-analyze",
        request=request,
        result=result,
        started_at=started_at,
        summary={
            "principal": principal.key_id,
            "task": result.task,
            "prediction": result.prediction,
            "confidence": result.confidence,
        },
    )
    return result


@app.get("/runs", response_model=list[RunRecord])
def list_runs(
    limit: int = 20,
    offset: int = 0,
    principal: Principal = RequireViewer,
) -> list[RunRecord]:
    return run_tracker.list_runs(limit=limit, offset=offset)


@app.get("/runs/{run_id}")
def read_run_artifact(
    run_id: str,
    principal: Principal = RequireViewer,
) -> dict[str, Any]:
    return run_tracker.read_artifact(run_id)


v1 = APIRouter(prefix="/api/v1", tags=["v1"])


@v1.get("/health")
def v1_health() -> dict[str, str]:
    return health()


@v1.get("/ready")
def v1_ready() -> dict[str, str]:
    return ready()


@v1.post("/masked-token", response_model=MaskedLMResult)
def v1_masked_token(
    request: MaskRequest,
    idempotency_key: str | None = Header(default=None, alias="X-Idempotency-Key"),
    principal: Principal = RequireAnalyst,
):
    return idempotent_execute(
        task="v1.masked-token",
        request=request,
        idempotency_key=idempotency_key,
        operation=lambda: masked_token(request, principal),
    )


@v1.post("/financial-analysis", response_model=FinancialAnalysis)
def v1_financial_analysis(
    request: FinancialRequest,
    idempotency_key: str | None = Header(default=None, alias="X-Idempotency-Key"),
    principal: Principal = RequireAnalyst,
):
    return idempotent_execute(
        task="v1.financial-analysis",
        request=request,
        idempotency_key=idempotency_key,
        operation=lambda: financial_analysis(request, principal),
    )


@v1.post("/financial-analysis/transcript", response_model=FinancialTranscriptReport)
def v1_financial_transcript_report(
    request: TranscriptReportRequest,
    idempotency_key: str | None = Header(default=None, alias="X-Idempotency-Key"),
    principal: Principal = RequireAnalyst,
):
    return idempotent_execute(
        task="v1.financial-transcript",
        request=request,
        idempotency_key=idempotency_key,
        operation=lambda: financial_transcript_report(request, principal),
    )


@v1.post("/attention/explore", response_model=AttentionExplorerResult)
def v1_explore_attention(
    request: AttentionRequest,
    idempotency_key: str | None = Header(default=None, alias="X-Idempotency-Key"),
    principal: Principal = RequireAnalyst,
):
    return idempotent_execute(
        task="v1.attention-explore",
        request=request,
        idempotency_key=idempotency_key,
        operation=lambda: explore_attention(request, principal),
    )


@v1.post("/embeddings/analyze", response_model=EmbeddingReport)
def v1_analyze_embeddings(
    request: EmbeddingRequest,
    idempotency_key: str | None = Header(default=None, alias="X-Idempotency-Key"),
    principal: Principal = RequireAnalyst,
):
    return idempotent_execute(
        task="v1.embeddings-analyze",
        request=request,
        idempotency_key=idempotency_key,
        operation=lambda: analyze_embeddings(request, principal),
    )


@v1.post("/models/compare", response_model=ModelComparisonReport)
def v1_compare_models(
    request: ModelComparisonRequest,
    idempotency_key: str | None = Header(default=None, alias="X-Idempotency-Key"),
    principal: Principal = RequireAnalyst,
):
    return idempotent_execute(
        task="v1.models-compare",
        request=request,
        idempotency_key=idempotency_key,
        operation=lambda: compare_models(request, principal),
    )


@v1.post("/explainability/analyze", response_model=ExplainabilityReport)
def v1_explain_prediction(
    request: ExplainabilityRequest,
    idempotency_key: str | None = Header(default=None, alias="X-Idempotency-Key"),
    principal: Principal = RequireAnalyst,
):
    return idempotent_execute(
        task="v1.explainability-analyze",
        request=request,
        idempotency_key=idempotency_key,
        operation=lambda: explain_prediction(request, principal),
    )


@v1.get("/models/registry", response_model=list[ModelRegistryEntry])
def v1_model_registry(principal: Principal = RequireViewer) -> list[ModelRegistryEntry]:
    return list_model_registry(principal)


@v1.get("/models/status", response_model=list[ModelStatus])
def v1_model_status(principal: Principal = RequireViewer) -> list[ModelStatus]:
    return list_model_status(principal)


@v1.post("/models/warmup", response_model=ModelWarmupResult | JobRecord)
def v1_warmup_model(
    request: ModelWarmupRequest,
    principal: Principal = RequireAnalyst,
) -> ModelWarmupResult | JobRecord:
    return warmup_model(request, principal)


@v1.get("/jobs/{job_id}", response_model=JobRecord)
def v1_get_job(job_id: str, principal: Principal = RequireViewer) -> JobRecord:
    return get_job(job_id, principal)


@v1.get("/runs", response_model=PaginatedRuns)
def v1_list_runs(
    limit: int = 20,
    offset: int = 0,
    principal: Principal = RequireViewer,
) -> PaginatedRuns:
    items = run_tracker.list_runs(limit=limit, offset=offset)
    return PaginatedRuns(items=items, limit=limit, offset=offset, count=len(items))


@v1.get("/runs/{run_id}")
def v1_read_run_artifact(
    run_id: str,
    principal: Principal = RequireViewer,
) -> dict[str, Any]:
    return read_run_artifact(run_id, principal)


@v1.get("/observability/summary", response_model=ObservabilitySummary)
def v1_observability_summary(
    principal: Principal = RequireViewer,
) -> ObservabilitySummary:
    return observability_summary(principal)


@v1.get("/metrics")
def v1_metrics() -> str:
    return metrics()


@v1.get("/governance/status", response_model=GovernanceStatus)
def v1_governance_status(principal: Principal = RequireViewer) -> GovernanceStatus:
    return governance_status(principal)


@v1.get("/governance/audit")
def v1_governance_audit(
    limit: int = 50,
    principal: Principal = RequireAdmin,
) -> list[dict]:
    return governance_audit(limit, principal)


@v1.post("/governance/retention/cleanup", response_model=RetentionCleanupResult)
def v1_governance_retention_cleanup(
    principal: Principal = RequireAdmin,
) -> RetentionCleanupResult:
    return governance_retention_cleanup(principal)


app.include_router(v1)
