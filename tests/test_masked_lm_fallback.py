from ber_tokenscope.model_adapters import DeterministicMaskedLM


def test_deterministic_masked_lm_returns_finance_predictions():
    output = DeterministicMaskedLM().predict(
        "The company reported record [MASK] this quarter.",
        top_k=3,
    )

    assert [prediction.token for prediction in output.predictions] == [
        "earnings",
        "revenue",
        "profits",
    ]
    assert output.tokens
