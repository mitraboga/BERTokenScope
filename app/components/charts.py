from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import plotly.express as px

if TYPE_CHECKING:
    from ber_tokenscope.schemas import (
        EmbeddingReport,
        ExplainabilityReport,
        FinancialAnalysis,
        FinancialTranscriptReport,
        MaskPrediction,
        ModelComparisonReport,
    )


def prediction_bar_chart(predictions: list[MaskPrediction]):
    frame = pd.DataFrame([prediction.model_dump() for prediction in predictions])
    return px.bar(
        frame,
        x="token",
        y="probability",
        text="probability",
        labels={"token": "Token", "probability": "Probability"},
        title="Top Masked-Token Predictions",
    )


def financial_signal_chart(analysis: FinancialAnalysis):
    frame = pd.DataFrame(
        [
            {"signal": "Risk", "score": analysis.risk_score},
            {"signal": "Uncertainty", "score": analysis.uncertainty_score},
            {"signal": "Optimism", "score": analysis.optimism_score},
        ]
    )
    return px.bar(
        frame,
        x="signal",
        y="score",
        color="signal",
        range_y=[0, 1],
        title="Financial Language Signals",
    )


def tone_drift_chart(report: FinancialTranscriptReport):
    frame = pd.DataFrame([point.model_dump() for point in report.tone_drift])
    return px.line(
        frame,
        x="period",
        y=["sentiment_score", "risk_score", "uncertainty_score", "optimism_score"],
        markers=True,
        labels={"value": "Score", "period": "Period", "variable": "Signal"},
        title="Financial Tone Drift",
    )


def transcript_chunk_chart(report: FinancialTranscriptReport):
    rows = []
    for period in report.periods:
        for chunk in period.chunks:
            rows.append(
                {
                    "period": period.period,
                    "chunk": chunk.chunk_id,
                    "risk": chunk.analysis.risk_score,
                    "uncertainty": chunk.analysis.uncertainty_score,
                    "optimism": chunk.analysis.optimism_score,
                    "sentiment": chunk.analysis.sentiment.score,
                }
            )
    frame = pd.DataFrame(rows)
    return px.bar(
        frame,
        x="chunk",
        y=["risk", "uncertainty", "optimism"],
        color_discrete_sequence=px.colors.qualitative.Safe,
        facet_col="period",
        barmode="group",
        title="Transcript Chunk Diagnostics",
    )


def embedding_scatter_chart(report: EmbeddingReport):
    frame = pd.DataFrame([point.model_dump() for point in report.points])
    return px.scatter(
        frame,
        x="x",
        y="y",
        color="cluster",
        hover_name="label",
        hover_data={"text": True, "x": ":.3f", "y": ":.3f", "cluster": True},
        title="Semantic Embedding Map",
    )


def similarity_heatmap(report: EmbeddingReport):
    labels = [point.label for point in report.points]
    matrix = pd.DataFrame(0.0, index=labels, columns=labels)
    for label in labels:
        matrix.loc[label, label] = 1.0
    for pair in report.similarities:
        matrix.loc[pair.left, pair.right] = pair.similarity
        matrix.loc[pair.right, pair.left] = pair.similarity
    return px.imshow(
        matrix,
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        zmin=-1,
        zmax=1,
        title="Document Similarity Matrix",
    )


def model_latency_chart(report: ModelComparisonReport):
    rows = []
    for section in [
        report.masked_language.runs,
        report.financial_sentiment.runs,
        report.embeddings.runs,
    ]:
        for run in section:
            rows.append(run.model_dump())
    frame = pd.DataFrame(rows)
    return px.bar(
        frame,
        x="model_name",
        y="latency_ms",
        color="task",
        labels={"model_name": "Model", "latency_ms": "Latency (ms)", "task": "Task"},
        title="Model Runtime Comparison",
    )


def model_confidence_chart(report: ModelComparisonReport):
    rows = []
    for section in [
        report.masked_language.runs,
        report.financial_sentiment.runs,
        report.embeddings.runs,
    ]:
        for run in section:
            rows.append(run.model_dump())
    frame = pd.DataFrame(rows)
    return px.bar(
        frame,
        x="model_name",
        y="confidence",
        color="task",
        range_y=[0, 1],
        labels={"model_name": "Model", "confidence": "Confidence", "task": "Task"},
        title="Model Confidence Comparison",
    )


def token_attribution_chart(report: ExplainabilityReport):
    frame = pd.DataFrame([item.model_dump() for item in report.token_attributions])
    if frame.empty:
        frame = pd.DataFrame([{"token": "none", "score": 0.0, "source": "none"}])
    return px.bar(
        frame.head(15),
        x="token",
        y="score",
        color="source",
        labels={"token": "Token", "score": "Attribution Score"},
        title="Token Attribution",
    )


def counterfactual_impact_chart(report: ExplainabilityReport):
    frame = pd.DataFrame([item.model_dump() for item in report.counterfactuals])
    if frame.empty:
        frame = pd.DataFrame([{"token": "none", "delta": 0.0}])
    return px.bar(
        frame.head(12),
        x="token",
        y="delta",
        labels={"token": "Removed Token", "delta": "Prediction Score Delta"},
        title="Counterfactual Token Impact",
    )
