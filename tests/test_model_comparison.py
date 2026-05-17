from ber_tokenscope.schemas import EmbeddingDocument
from model_comparison.engine import ModelComparisonEngine, prediction_overlap


def test_prediction_overlap_returns_shared_tokens():
    overlap = prediction_overlap(
        {
            "A": ["earnings", "revenue"],
            "B": ["revenue", "growth"],
            "C": ["cash"],
        }
    )

    assert overlap["A vs B"] == ["revenue"]
    assert overlap["A vs C"] == []


def test_model_comparison_engine_generates_all_sections():
    report = ModelComparisonEngine().compare(
        masked_text="The company reported record [MASK] this quarter.",
        financial_text="Revenue growth was strong, but inflation risk remains.",
        documents=[
            EmbeddingDocument(label="A", text="revenue growth demand"),
            EmbeddingDocument(label="B", text="strong revenue growth"),
            EmbeddingDocument(label="C", text="litigation risk recession"),
        ],
    )

    assert len(report.masked_language.runs) == 3
    assert len(report.financial_sentiment.runs) == 2
    assert len(report.embeddings.runs) == 2
    assert report.recommendation
