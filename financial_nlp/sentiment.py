from __future__ import annotations

import re
from collections.abc import Iterable

from ber_tokenscope.schemas import FinancialAnalysis, FinancialSignal
from financial_nlp.lexicons import OPTIMISM_TERMS, RISK_TERMS, UNCERTAINTY_TERMS

TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z\-']*")


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def matches(tokens: Iterable[str], lexicon: set[str]) -> list[str]:
    return sorted({token for token in tokens if token in lexicon})


def ratio(count: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return min(1.0, count / denominator)


class FinancialTextAnalyzer:
    """Finance-specific lexical analyzer with a stable offline baseline."""

    model_name = "lexical-financial-baseline"

    def analyze(self, text: str) -> FinancialAnalysis:
        tokens = tokenize(text)
        token_count = max(len(tokens), 1)

        risk_terms = matches(tokens, RISK_TERMS)
        uncertainty_terms = matches(tokens, UNCERTAINTY_TERMS)
        optimism_terms = matches(tokens, OPTIMISM_TERMS)

        risk_score = ratio(len(risk_terms) * 8, token_count)
        uncertainty_score = ratio(len(uncertainty_terms) * 8, token_count)
        optimism_score = ratio(len(optimism_terms) * 8, token_count)
        net_score = optimism_score - ((risk_score + uncertainty_score) / 2)

        if net_score > 0.08:
            label = "positive"
            explanation = (
                "Optimistic financial language outweighs risk and uncertainty terms."
            )
        elif net_score < -0.08:
            label = "negative"
            explanation = (
                "Risk and uncertainty language outweighs optimistic financial terms."
            )
        else:
            label = "neutral"
            explanation = (
                "Optimism, risk, and uncertainty signals are broadly balanced."
            )

        return FinancialAnalysis(
            text=text,
            sentiment=FinancialSignal(
                label=label,
                score=round((net_score + 1) / 2, 4),
                explanation=explanation,
            ),
            risk_score=round(risk_score, 4),
            uncertainty_score=round(uncertainty_score, 4),
            optimism_score=round(optimism_score, 4),
            matched_terms={
                "risk": risk_terms,
                "uncertainty": uncertainty_terms,
                "optimism": optimism_terms,
            },
            model_name=self.model_name,
        )
