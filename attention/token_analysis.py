from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ber_tokenscope.schemas import AttentionLink


def strongest_attention_links(
    tokens: Sequence[str],
    attention_matrix: np.ndarray,
    *,
    limit: int = 12,
    skip_special_tokens: bool = True,
) -> list[AttentionLink]:
    """Return the strongest directed token-to-token attention links."""

    if attention_matrix.shape != (len(tokens), len(tokens)):
        raise ValueError("attention_matrix must be square and match token count.")

    special = {"[CLS]", "[SEP]", "[PAD]"}
    links: list[AttentionLink] = []
    for source_index, source in enumerate(tokens):
        if skip_special_tokens and source in special:
            continue
        for target_index, target in enumerate(tokens):
            if source_index == target_index:
                continue
            if skip_special_tokens and target in special:
                continue
            links.append(
                AttentionLink(
                    source=source,
                    target=target,
                    source_index=source_index,
                    target_index=target_index,
                    score=float(attention_matrix[source_index, target_index]),
                )
            )

    return sorted(links, key=lambda link: link.score, reverse=True)[:limit]
