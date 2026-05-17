from pathlib import Path

from attention.engine import AttentionEngine
from explainability.engine import ExplainabilityEngine

ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_cloud_assets_exist() -> None:
    assert (ROOT / ".streamlit" / "config.toml").exists()
    assert (ROOT / "docs" / "streamlit-cloud.md").exists()
    assert (ROOT / "docs" / "portfolio-deployment.md").exists()
    assert (ROOT / "requirements-models.txt").exists()


def test_public_demo_attention_uses_offline_fallback() -> None:
    result = AttentionEngine(offline=True).analyze_mask(
        "The company reported record [MASK] this quarter.",
        top_k=3,
    )

    assert result.model_name == "deterministic-fallback"
    assert [prediction.token for prediction in result.predictions] == [
        "earnings",
        "revenue",
        "profits",
    ]


def test_public_demo_explainability_uses_offline_fallback() -> None:
    report = ExplainabilityEngine(offline=True).explain_masked_language(
        "The company reported record [MASK] this quarter."
    )

    assert report.model_name == "deterministic-fallback"
    assert report.prediction == "earnings"
