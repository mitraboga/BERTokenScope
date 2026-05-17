from __future__ import annotations

import numpy as np


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D array.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = vectors / norms
    return normalized @ normalized.T


def top_similarity_pairs(
    labels: list[str],
    vectors: np.ndarray,
    *,
    limit: int = 10,
) -> list[tuple[str, str, float]]:
    if len(labels) != vectors.shape[0]:
        raise ValueError("labels must match vector count.")

    matrix = cosine_similarity_matrix(vectors)
    pairs = []
    for row in range(len(labels)):
        for column in range(row + 1, len(labels)):
            pairs.append((labels[row], labels[column], float(matrix[row, column])))
    return sorted(pairs, key=lambda item: item[2], reverse=True)[:limit]
