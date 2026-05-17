from __future__ import annotations

import time

import numpy as np

from attention.engine import AttentionEngine
from ber_tokenscope.schemas import (
    EmbeddingDocument,
    EmbeddingModelComparison,
    FinancialModelComparison,
    MaskedModelComparison,
    ModelComparisonReport,
    ModelRunMetric,
)
from embeddings.engine import EmbeddingEngine
from financial_nlp.finbert_pipeline import HybridFinancialAnalyzer
from financial_nlp.sentiment import FinancialTextAnalyzer
from model_comparison.profiles import (
    EMBEDDING_PROFILES,
    FINANCIAL_PROFILES,
    MASKED_LM_PROFILES,
    ModelProfile,
)


def timed_call(function):
    start = time.perf_counter()
    value = function()
    latency_ms = (time.perf_counter() - start) * 1000
    return value, round(latency_ms, 3)


class ModelComparisonEngine:
    """Compare task-specific model behavior, confidence, and latency."""

    def compare(
        self,
        *,
        masked_text: str,
        financial_text: str,
        documents: list[EmbeddingDocument],
        top_k: int = 5,
    ) -> ModelComparisonReport:
        masked = self.compare_masked_language(masked_text, top_k=top_k)
        financial = self.compare_financial_sentiment(financial_text)
        embeddings = self.compare_embeddings(documents)
        return ModelComparisonReport(
            masked_language=masked,
            financial_sentiment=financial,
            embeddings=embeddings,
            recommendation=build_recommendations(masked, financial, embeddings),
        )

    def compare_masked_language(
        self, text: str, *, top_k: int = 5
    ) -> MaskedModelComparison:
        runs: list[ModelRunMetric] = []
        model_predictions: dict[str, list[str]] = {}
        for profile in MASKED_LM_PROFILES:
            result, latency_ms = timed_call(
                lambda profile=profile: AttentionEngine(
                    model_name=profile.model_name,
                    offline=profile.offline,
                ).analyze_mask(text, top_k=top_k)
            )
            tokens = [prediction.token for prediction in result.predictions]
            model_predictions[profile.display_name] = tokens
            confidence = (
                result.predictions[0].probability if result.predictions else 0.0
            )
            runs.append(
                ModelRunMetric(
                    model_name=profile.display_name,
                    task=profile.task,
                    latency_ms=latency_ms,
                    confidence=round(confidence, 4),
                    output=", ".join(tokens),
                    notes=list(profile.notes),
                )
            )
        return MaskedModelComparison(
            text=text,
            runs=runs,
            prediction_overlap=prediction_overlap(model_predictions),
        )

    def compare_financial_sentiment(self, text: str) -> FinancialModelComparison:
        runs = [
            self._financial_run(FINANCIAL_PROFILES[0], FinancialTextAnalyzer(), text),
            self._financial_run(
                FINANCIAL_PROFILES[1],
                HybridFinancialAnalyzer(prefer_finbert=True),
                text,
            ),
        ]
        return FinancialModelComparison(text=text, runs=runs)

    def compare_embeddings(
        self, documents: list[EmbeddingDocument]
    ) -> EmbeddingModelComparison:
        runs: list[ModelRunMetric] = []
        for profile in EMBEDDING_PROFILES:
            prefer_transformer = profile.model_name != "deterministic-feature-hashing"
            report, latency_ms = timed_call(
                lambda prefer_transformer=prefer_transformer: EmbeddingEngine(
                    prefer_transformer=prefer_transformer
                ).analyze(documents, cluster_count=min(3, len(documents)))
            )
            mean_similarity = 0.0
            if report.similarities:
                mean_similarity = float(
                    np.mean([pair.similarity for pair in report.similarities])
                )
            runs.append(
                ModelRunMetric(
                    model_name=profile.display_name,
                    task=profile.task,
                    latency_ms=latency_ms,
                    confidence=round((mean_similarity + 1) / 2, 4),
                    output=f"{len(report.points)} points, {len(report.similarities)} similarity pairs",
                    notes=[*profile.notes, f"runtime model: {report.model_name}"],
                )
            )
        return EmbeddingModelComparison(documents=documents, runs=runs)

    @staticmethod
    def _financial_run(
        profile: ModelProfile,
        analyzer,
        text: str,
    ) -> ModelRunMetric:
        result, latency_ms = timed_call(lambda: analyzer.analyze(text))
        return ModelRunMetric(
            model_name=profile.display_name,
            task=profile.task,
            latency_ms=latency_ms,
            confidence=result.sentiment.score,
            output=(
                f"{result.sentiment.label} | risk={result.risk_score} | "
                f"uncertainty={result.uncertainty_score} | optimism={result.optimism_score}"
            ),
            notes=[*profile.notes, f"runtime model: {result.model_name}"],
        )


def prediction_overlap(model_predictions: dict[str, list[str]]) -> dict[str, list[str]]:
    overlap: dict[str, list[str]] = {}
    names = list(model_predictions)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            shared = sorted(
                set(model_predictions[left]) & set(model_predictions[right])
            )
            overlap[f"{left} vs {right}"] = shared
    return overlap


def build_recommendations(
    masked: MaskedModelComparison,
    financial: FinancialModelComparison,
    embeddings: EmbeddingModelComparison,
) -> list[str]:
    fastest = min(
        [*masked.runs, *financial.runs, *embeddings.runs],
        key=lambda run: run.latency_ms,
    )
    highest_confidence = max(
        [*masked.runs, *financial.runs, *embeddings.runs],
        key=lambda run: run.confidence,
    )
    return [
        f"Fastest observed run: {fastest.model_name} on {fastest.task}.",
        f"Highest confidence run: {highest_confidence.model_name} on {highest_confidence.task}.",
        "Use domain-specific models for final finance decisions and compact baselines for fast exploratory workflows.",
    ]
