import json
from pathlib import Path

import pytest

from attention.engine import AttentionEngine
from ber_tokenscope.schemas import EmbeddingDocument
from embeddings.engine import EmbeddingEngine
from financial_nlp.sentiment import FinancialTextAnalyzer

pytestmark = pytest.mark.regression


def load_cases():
    path = Path(__file__).parent / "fixtures" / "nlp_regression_cases.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_masked_language_regression_fixture():
    case = load_cases()["masked_language"][0]
    result = AttentionEngine(offline=True).analyze_mask(case["text"], top_k=3)

    assert [prediction.token for prediction in result.predictions] == case[
        "expected_top_tokens"
    ]


def test_financial_terms_regression_fixture():
    case = load_cases()["financial"][0]
    result = FinancialTextAnalyzer().analyze(case["text"])

    for category, expected_terms in case["expected_terms"].items():
        assert result.matched_terms[category] == expected_terms


def test_embedding_similarity_regression_fixture():
    case = load_cases()["embeddings"][0]
    documents = [EmbeddingDocument(**document) for document in case["documents"]]
    report = EmbeddingEngine().analyze(documents, cluster_count=2)

    closest = report.similarities[0]
    assert [closest.left, closest.right] == case["expected_closest_pair"]
