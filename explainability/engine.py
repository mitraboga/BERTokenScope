from __future__ import annotations

import numpy as np

from attention.engine import AttentionEngine
from ber_tokenscope.schemas import (
    CounterfactualTokenImpact,
    ExplainabilityReport,
    TokenAttribution,
)
from explainability.counterfactuals import financial_counterfactual_impacts
from explainability.token_importance import attention_token_importance
from financial_nlp.sentiment import FinancialTextAnalyzer


class ExplainabilityEngine:
    def __init__(self, *, offline: bool = False) -> None:
        self.attention_engine = AttentionEngine(offline=offline)
        self.financial_analyzer = FinancialTextAnalyzer()

    def explain_masked_language(
        self,
        text: str,
        *,
        layer: int = 1,
        head: int = 1,
    ) -> ExplainabilityReport:
        attention = self.attention_engine.explore_attention(
            text, layer=layer, head=head
        )
        matrix = np.asarray(attention.attention_matrix, dtype=float)
        importance = attention_token_importance(attention.tokens, matrix)
        mask_result = self.attention_engine.analyze_mask(text)
        prediction = mask_result.predictions[0] if mask_result.predictions else None

        attributions = [
            TokenAttribution(
                token=str(item["token"]),
                score=round(float(item["importance"]), 6),
                source=f"attention layer {layer}, head {head}",
                rationale="Token received high inbound attention from other sequence positions.",
            )
            for item in importance
        ]
        rationale = build_masked_rationale(
            attributions[:5], prediction.token if prediction else ""
        )
        return ExplainabilityReport(
            text=text,
            task="masked-language",
            prediction=prediction.token if prediction else "",
            confidence=prediction.probability if prediction else 0.0,
            token_attributions=attributions,
            counterfactuals=[],
            rationale=rationale,
            model_name=attention.model_name,
        )

    def explain_financial_text(self, text: str) -> ExplainabilityReport:
        analysis = self.financial_analyzer.analyze(text)
        attributions = financial_token_attributions(text, analysis.matched_terms)
        impacts = financial_counterfactual_impacts(
            text,
            [attribution.token for attribution in attributions],
            analyzer=self.financial_analyzer,
        )
        counterfactuals = [CounterfactualTokenImpact(**impact) for impact in impacts]
        return ExplainabilityReport(
            text=text,
            task="financial-sentiment",
            prediction=analysis.sentiment.label,
            confidence=analysis.sentiment.score,
            token_attributions=attributions,
            counterfactuals=counterfactuals,
            rationale=build_financial_rationale(
                analysis, attributions, counterfactuals
            ),
            model_name=analysis.model_name,
        )


def financial_token_attributions(
    text: str,
    matched_terms: dict[str, list[str]],
) -> list[TokenAttribution]:
    weights = {"risk": 1.0, "uncertainty": 0.85, "optimism": 0.75}
    attributions = []
    for category, terms in matched_terms.items():
        for term in terms:
            attributions.append(
                TokenAttribution(
                    token=term,
                    score=weights.get(category, 0.5),
                    source=f"financial-{category}-lexicon",
                    rationale=f"Token matched the {category} financial language signal.",
                )
            )
    return sorted(attributions, key=lambda item: item.score, reverse=True)


def build_masked_rationale(
    attributions: list[TokenAttribution],
    prediction: str,
) -> list[str]:
    if not attributions:
        return ["No strong token-level attention rationale was available."]
    tokens = ", ".join(attribution.token for attribution in attributions)
    return [
        f"The predicted token `{prediction}` is explained using attention-based attribution.",
        f"The highest-attribution context tokens were: {tokens}.",
        "These tokens received the most inbound attention in the selected attention head.",
    ]


def build_financial_rationale(
    analysis,
    attributions: list[TokenAttribution],
    counterfactuals: list[CounterfactualTokenImpact],
) -> list[str]:
    rationale = [analysis.sentiment.explanation]
    if attributions:
        tokens = ", ".join(attribution.token for attribution in attributions[:5])
        rationale.append(f"The strongest financial language signals were: {tokens}.")
    if counterfactuals:
        top = counterfactuals[0]
        direction = "lowered" if top.delta > 0 else "raised"
        rationale.append(
            f"Removing `{top.token}` {direction} the sentiment score impact by {abs(top.delta):.3f}."
        )
    return rationale
