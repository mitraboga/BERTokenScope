from __future__ import annotations

import numpy as np


def attention_entropy(attention_matrix: np.ndarray) -> float:
    """Return normalized entropy, where lower values indicate sharper focus."""

    if (
        attention_matrix.ndim != 2
        or attention_matrix.shape[0] != attention_matrix.shape[1]
    ):
        raise ValueError("attention_matrix must be square.")

    clipped = np.clip(attention_matrix, 1e-12, 1.0)
    row_entropy = -(clipped * np.log(clipped)).sum(axis=1)
    max_entropy = np.log(attention_matrix.shape[1])
    if max_entropy == 0:
        return 0.0
    return float(np.mean(row_entropy / max_entropy))


def focus_score(attention_matrix: np.ndarray) -> float:
    """Return mean max attention per token; higher means more concentrated attention."""

    if (
        attention_matrix.ndim != 2
        or attention_matrix.shape[0] != attention_matrix.shape[1]
    ):
        raise ValueError("attention_matrix must be square.")
    return float(attention_matrix.max(axis=1).mean())


def normalize_attention_rows(attention_matrix: np.ndarray) -> np.ndarray:
    row_sums = attention_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return attention_matrix / row_sums
