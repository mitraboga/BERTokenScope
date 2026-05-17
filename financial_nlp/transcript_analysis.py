from __future__ import annotations

import re
from dataclasses import dataclass

from ber_tokenscope.schemas import (
    FinancialAnalysis,
    FinancialPeriodAnalysis,
    FinancialSignal,
    FinancialTranscriptReport,
    ToneDriftPoint,
    TranscriptChunkAnalysis,
)
from financial_nlp.finbert_pipeline import FinancialAnalyzer, HybridFinancialAnalyzer

SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class PeriodDocument:
    period: str
    text: str


def chunk_text(text: str, *, max_words: int = 180) -> list[str]:
    """Chunk transcript text on sentence boundaries for model-safe analysis."""

    words = text.split()
    if len(words) <= max_words:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    current: list[str] = []
    for sentence in SENTENCE_PATTERN.split(text.strip()):
        sentence_words = sentence.split()
        if current and len(current) + len(sentence_words) > max_words:
            chunks.append(" ".join(current))
            current = []
        current.extend(sentence_words)
    if current:
        chunks.append(" ".join(current))
    return chunks


def aggregate_chunk_analyses(
    text: str, analyses: list[FinancialAnalysis]
) -> FinancialAnalysis:
    if not analyses:
        return HybridFinancialAnalyzer(prefer_finbert=False).analyze(text)

    risk_score = sum(item.risk_score for item in analyses) / len(analyses)
    uncertainty_score = sum(item.uncertainty_score for item in analyses) / len(analyses)
    optimism_score = sum(item.optimism_score for item in analyses) / len(analyses)
    sentiment_score = sum(item.sentiment.score for item in analyses) / len(analyses)

    labels = [item.sentiment.label for item in analyses]
    label = max(set(labels), key=labels.count)
    matched_terms: dict[str, list[str]] = {
        "risk": [],
        "uncertainty": [],
        "optimism": [],
    }
    for analysis in analyses:
        for category, terms in analysis.matched_terms.items():
            matched_terms.setdefault(category, [])
            matched_terms[category].extend(terms)
    matched_terms = {
        category: sorted(set(terms)) for category, terms in matched_terms.items()
    }

    return FinancialAnalysis(
        text=text,
        sentiment=FinancialSignal(
            label=label,
            score=round(sentiment_score, 4),
            explanation="Aggregated transcript chunk sentiment and financial language signals.",
        ),
        risk_score=round(risk_score, 4),
        uncertainty_score=round(uncertainty_score, 4),
        optimism_score=round(optimism_score, 4),
        matched_terms=matched_terms,
        model_name=analyses[0].model_name,
    )


class TranscriptAnalyzer:
    def __init__(self, analyzer: FinancialAnalyzer | None = None) -> None:
        self.analyzer = analyzer or HybridFinancialAnalyzer(prefer_finbert=False)

    def analyze_period(
        self,
        document: PeriodDocument,
        *,
        max_words: int = 180,
    ) -> FinancialPeriodAnalysis:
        chunks = chunk_text(document.text, max_words=max_words)
        chunk_results = [
            TranscriptChunkAnalysis(
                chunk_id=index,
                text=chunk,
                analysis=self.analyzer.analyze(chunk),
            )
            for index, chunk in enumerate(chunks, start=1)
        ]
        aggregate = aggregate_chunk_analyses(
            document.text,
            [chunk.analysis for chunk in chunk_results],
        )
        return FinancialPeriodAnalysis(
            period=document.period,
            text=document.text,
            aggregate=aggregate,
            chunks=chunk_results,
        )

    def analyze_report(
        self,
        documents: list[PeriodDocument],
        *,
        company: str | None = None,
        max_words: int = 180,
    ) -> FinancialTranscriptReport:
        periods = [
            self.analyze_period(document, max_words=max_words) for document in documents
        ]
        drift = compute_tone_drift(periods)
        return FinancialTranscriptReport(
            company=company,
            periods=periods,
            tone_drift=drift,
            executive_summary=build_executive_summary(company, drift),
        )


def compute_tone_drift(periods: list[FinancialPeriodAnalysis]) -> list[ToneDriftPoint]:
    points: list[ToneDriftPoint] = []
    previous: ToneDriftPoint | None = None
    for period in periods:
        aggregate = period.aggregate
        point = ToneDriftPoint(
            period=period.period,
            sentiment_score=aggregate.sentiment.score,
            risk_score=aggregate.risk_score,
            uncertainty_score=aggregate.uncertainty_score,
            optimism_score=aggregate.optimism_score,
            delta_sentiment=(
                None
                if previous is None
                else round(aggregate.sentiment.score - previous.sentiment_score, 4)
            ),
            delta_risk=(
                None
                if previous is None
                else round(aggregate.risk_score - previous.risk_score, 4)
            ),
        )
        points.append(point)
        previous = point
    return points


def build_executive_summary(
    company: str | None,
    drift: list[ToneDriftPoint],
) -> list[str]:
    subject = company or "The company"
    if not drift:
        return [f"{subject} has no transcript periods to summarize."]

    latest = drift[-1]
    summary = [
        (
            f"{subject} latest tone score is {latest.sentiment_score:.2f}, with "
            f"risk at {latest.risk_score:.2f} and uncertainty at {latest.uncertainty_score:.2f}."
        )
    ]
    if latest.delta_sentiment is not None:
        direction = "improved" if latest.delta_sentiment >= 0 else "weakened"
        summary.append(
            f"Sentiment {direction} by {abs(latest.delta_sentiment):.2f} versus the prior period."
        )
    if latest.delta_risk is not None:
        direction = "increased" if latest.delta_risk >= 0 else "decreased"
        summary.append(
            f"Risk language {direction} by {abs(latest.delta_risk):.2f} versus the prior period."
        )
    return summary
