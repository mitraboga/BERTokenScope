from __future__ import annotations

from financial_nlp.sentiment import FinancialTextAnalyzer, tokenize


def remove_first_token_occurrence(text: str, token: str) -> str:
    words = text.split()
    lowered = token.lower()
    for index, word in enumerate(words):
        cleaned = word.strip(".,;:!?()[]{}\"'").lower()
        if cleaned == lowered:
            return " ".join([*words[:index], *words[index + 1 :]])
    return text


def financial_counterfactual_impacts(
    text: str,
    important_tokens: list[str],
    *,
    analyzer: FinancialTextAnalyzer | None = None,
    limit: int = 8,
):
    analyzer = analyzer or FinancialTextAnalyzer()
    original = analyzer.analyze(text)
    impacts = []
    seen = set()
    available = set(tokenize(text))

    for token in important_tokens:
        normalized = token.lower().replace("##", "")
        if normalized in seen or normalized not in available:
            continue
        seen.add(normalized)
        counterfactual_text = remove_first_token_occurrence(text, normalized)
        counterfactual = analyzer.analyze(counterfactual_text)
        delta = original.sentiment.score - counterfactual.sentiment.score
        impacts.append(
            {
                "token": normalized,
                "original_score": original.sentiment.score,
                "counterfactual_score": counterfactual.sentiment.score,
                "delta": round(delta, 6),
                "counterfactual_text": counterfactual_text,
            }
        )
        if len(impacts) >= limit:
            break

    return sorted(impacts, key=lambda item: abs(item["delta"]), reverse=True)
