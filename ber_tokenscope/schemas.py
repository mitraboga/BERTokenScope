from __future__ import annotations

from pydantic import BaseModel, Field


class MaskPrediction(BaseModel):
    token: str
    sequence: str
    probability: float = Field(ge=0.0, le=1.0)


class AttentionLink(BaseModel):
    source: str
    target: str
    source_index: int
    target_index: int
    score: float = Field(ge=0.0, le=1.0)


class MaskedLMResult(BaseModel):
    text: str
    model_name: str
    predictions: list[MaskPrediction]
    tokens: list[str]
    strongest_links: list[AttentionLink] = Field(default_factory=list)


class AttentionHeadSummary(BaseModel):
    layer: int = Field(ge=1)
    head: int = Field(ge=1)
    entropy: float = Field(ge=0.0)
    focus_score: float = Field(ge=0.0, le=1.0)


class AttentionExplorerResult(BaseModel):
    text: str
    model_name: str
    tokens: list[str]
    layer_count: int
    head_count: int
    selected_layer: int
    selected_head: int
    attention_matrix: list[list[float]]
    rollout_matrix: list[list[float]]
    strongest_links: list[AttentionLink]
    head_summaries: list[AttentionHeadSummary]
    diagram_path: str | None = None


class FinancialSignal(BaseModel):
    label: str
    score: float
    explanation: str


class FinancialAnalysis(BaseModel):
    text: str
    sentiment: FinancialSignal
    risk_score: float
    uncertainty_score: float
    optimism_score: float
    matched_terms: dict[str, list[str]]
    model_name: str = "lexical-financial-baseline"


class TranscriptChunkAnalysis(BaseModel):
    chunk_id: int
    text: str
    analysis: FinancialAnalysis


class FinancialPeriodAnalysis(BaseModel):
    period: str
    text: str
    aggregate: FinancialAnalysis
    chunks: list[TranscriptChunkAnalysis]


class ToneDriftPoint(BaseModel):
    period: str
    sentiment_score: float
    risk_score: float
    uncertainty_score: float
    optimism_score: float
    delta_sentiment: float | None = None
    delta_risk: float | None = None


class FinancialTranscriptReport(BaseModel):
    company: str | None = None
    periods: list[FinancialPeriodAnalysis]
    tone_drift: list[ToneDriftPoint]
    executive_summary: list[str]


class EmbeddingDocument(BaseModel):
    label: str
    text: str
    metadata: dict[str, str] = Field(default_factory=dict)


class EmbeddingPoint(BaseModel):
    label: str
    text: str
    metadata: dict[str, str] = Field(default_factory=dict)
    x: float
    y: float
    cluster: int


class EmbeddingSimilarityPair(BaseModel):
    left: str
    right: str
    similarity: float = Field(ge=-1.0, le=1.0)


class EmbeddingReport(BaseModel):
    model_name: str
    dimensions: int
    points: list[EmbeddingPoint]
    similarities: list[EmbeddingSimilarityPair]


class ModelRunMetric(BaseModel):
    model_name: str
    task: str
    latency_ms: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    output: str
    memory_mb: float | None = Field(default=None, ge=0.0)
    notes: list[str] = Field(default_factory=list)


class MaskedModelComparison(BaseModel):
    text: str
    runs: list[ModelRunMetric]
    prediction_overlap: dict[str, list[str]]


class FinancialModelComparison(BaseModel):
    text: str
    runs: list[ModelRunMetric]


class EmbeddingModelComparison(BaseModel):
    documents: list[EmbeddingDocument]
    runs: list[ModelRunMetric]


class ModelComparisonReport(BaseModel):
    masked_language: MaskedModelComparison
    financial_sentiment: FinancialModelComparison
    embeddings: EmbeddingModelComparison
    recommendation: list[str]


class TokenAttribution(BaseModel):
    token: str
    score: float = Field(ge=0.0)
    source: str
    rationale: str


class CounterfactualTokenImpact(BaseModel):
    token: str
    original_score: float
    counterfactual_score: float
    delta: float
    counterfactual_text: str


class ExplainabilityReport(BaseModel):
    text: str
    task: str
    prediction: str
    confidence: float = Field(ge=0.0, le=1.0)
    token_attributions: list[TokenAttribution]
    counterfactuals: list[CounterfactualTokenImpact]
    rationale: list[str]
    model_name: str


class RunRecord(BaseModel):
    run_id: str
    task: str
    status: str
    created_at: str
    duration_ms: float = Field(ge=0.0)
    artifact_path: str
    summary: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class ModelRegistryEntry(BaseModel):
    model_id: str
    display_name: str
    task: str
    provider: str = "huggingface"
    revision: str | None = None
    required: bool = False
    fallback: str | None = None


class ModelStatus(BaseModel):
    model_id: str
    display_name: str
    task: str
    cached: bool
    cache_path: str
    serving_mode: str
    notes: list[str] = Field(default_factory=list)


class ModelWarmupResult(BaseModel):
    model_id: str
    status: str
    duration_ms: float
    message: str


class JobRecord(BaseModel):
    job_id: str
    task: str
    status: str
    created_at: str
    updated_at: str
    result: dict | None = None
    error: str | None = None


class ObservabilitySummary(BaseModel):
    request_count: int
    error_count: int
    average_latency_ms: float
    routes: dict[str, int]
    statuses: dict[str, int]


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str | None = None


class ErrorEnvelope(BaseModel):
    error: ErrorDetail


class PaginatedRuns(BaseModel):
    items: list[RunRecord]
    limit: int
    offset: int
    count: int


class GovernanceStatus(BaseModel):
    redact_artifacts: bool
    retention_days: int
    audit_log_path: str
    financial_disclaimer: str


class RetentionCleanupResult(BaseModel):
    deleted_artifacts: int
    retention_days: int
