from __future__ import annotations

import numpy as np


def reduce_to_2d(vectors: np.ndarray) -> np.ndarray:
    """Project vectors to 2D with PCA for deterministic dashboard plots."""

    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D array.")
    if vectors.shape[0] < 2:
        return np.zeros((vectors.shape[0], 2))

    centered = vectors - vectors.mean(axis=0, keepdims=True)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ components[:2].T
    if projected.shape[1] == 1:
        projected = np.column_stack([projected[:, 0], np.zeros(projected.shape[0])])
    return projected[:, :2]
