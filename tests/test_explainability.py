import numpy as np

from explainability.counterfactuals import (
    financial_counterfactual_impacts,
    remove_first_token_occurrence,
)
from explainability.engine import ExplainabilityEngine, financial_token_attributions
from explainability.token_importance import attention_token_importance


def test_attention_token_importance_sorts_by_inbound_attention():
    tokens = ["company", "growth"]
    matrix = np.array([[0.1, 0.9], [0.2, 0.8]])

    importance = attention_token_importance(tokens, matrix)

    assert importance[0]["token"] == "growth"


def test_remove_first_token_occurrence_removes_matching_word():
    text = "Inflation risk remains, but revenue improved."

    result = remove_first_token_occurrence(text, "risk")

    assert "risk" not in result.lower()
    assert "Inflation" in result


def test_financial_counterfactual_impacts_returns_deltas():
    impacts = financial_counterfactual_impacts(
        "Revenue growth was strong, but inflation risk remains.",
        ["growth", "risk"],
    )

    assert impacts
    assert {"token", "delta", "counterfactual_text"} <= set(impacts[0])


def test_financial_token_attributions_use_signal_sources():
    attributions = financial_token_attributions(
        "Revenue growth had risk.",
        {"risk": ["risk"], "uncertainty": [], "optimism": ["growth"]},
    )

    assert attributions[0].source == "financial-risk-lexicon"


def test_explainability_engine_financial_report():
    report = ExplainabilityEngine().explain_financial_text(
        "Revenue growth was strong, but inflation risk remains."
    )

    assert report.task == "financial-sentiment"
    assert report.token_attributions
    assert report.rationale


def test_explainability_engine_masked_report():
    report = ExplainabilityEngine().explain_masked_language(
        "The company reported record [MASK] this quarter.",
        layer=1,
        head=1,
    )

    assert report.task == "masked-language"
    assert report.prediction
    assert report.token_attributions
