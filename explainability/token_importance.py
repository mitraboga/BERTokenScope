from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def attention_token_importance(
    tokens: Sequence[str], attention_matrix: np.ndarray
) -> list[dict[str, float | str]]:
    """Estimate token importance from inbound attention mass."""

    if attention_matrix.shape != (len(tokens), len(tokens)):
        raise ValueError("attention_matrix must be square and match token count.")

    inbound = attention_matrix.sum(axis=0)
    total = float(inbound.sum()) or 1.0
    normalized = inbound / total
    return [
        {"token": token, "importance": float(score)}
        for token, score in sorted(
            zip(tokens, normalized, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
