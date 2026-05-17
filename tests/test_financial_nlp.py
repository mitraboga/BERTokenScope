from financial_nlp.sentiment import FinancialTextAnalyzer
from financial_nlp.transcript_analysis import (
    PeriodDocument,
    TranscriptAnalyzer,
    chunk_text,
    compute_tone_drift,
)


def test_financial_analyzer_detects_risk_and_optimism():
    text = "Revenue growth was strong, but inflation risk created uncertain guidance."
    analysis = FinancialTextAnalyzer().analyze(text)

    assert "growth" in analysis.matched_terms["optimism"]
    assert "risk" in analysis.matched_terms["risk"]
    assert "uncertain" in analysis.matched_terms["uncertainty"]
    assert 0 <= analysis.risk_score <= 1


def test_chunk_text_splits_long_transcripts():
    text = "Revenue was strong. " * 80

    chunks = chunk_text(text, max_words=30)

    assert len(chunks) > 1
    assert all(len(chunk.split()) <= 30 for chunk in chunks)


def test_transcript_analyzer_generates_tone_drift():
    report = TranscriptAnalyzer().analyze_report(
        [
            PeriodDocument(
                period="Q1",
                text="Revenue growth was strong and demand improved.",
            ),
            PeriodDocument(
                period="Q2",
                text="Inflation risk and uncertain guidance created headwinds.",
            ),
        ],
        company="Example Corp",
        max_words=40,
    )

    assert len(report.periods) == 2
    assert len(report.tone_drift) == 2
    assert report.tone_drift[1].delta_sentiment is not None
    assert report.executive_summary


def test_compute_tone_drift_marks_initial_delta_as_none():
    report = TranscriptAnalyzer().analyze_report(
        [PeriodDocument(period="Q1", text="Revenue growth was strong.")],
        max_words=40,
    )

    drift = compute_tone_drift(report.periods)

    assert drift[0].delta_sentiment is None
