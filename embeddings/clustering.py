from __future__ import annotations

import numpy as np


def assign_clusters(vectors: np.ndarray, cluster_count: int) -> list[int]:
    """Cluster embeddings with deterministic NumPy k-means."""

    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D array.")
    if vectors.shape[0] == 0:
        return []

    cluster_count = max(1, min(cluster_count, vectors.shape[0]))
    if cluster_count == 1:
        return [0 for _ in range(vectors.shape[0])]

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = vectors / norms

    seed_indices = np.linspace(0, normalized.shape[0] - 1, cluster_count, dtype=int)
    centroids = normalized[seed_indices].copy()
    labels = np.zeros(normalized.shape[0], dtype=int)

    for _ in range(25):
        distances = np.linalg.norm(
            normalized[:, None, :] - centroids[None, :, :], axis=2
        )
        next_labels = distances.argmin(axis=1)
        if np.array_equal(labels, next_labels):
            break
        labels = next_labels
        for cluster in range(cluster_count):
            members = normalized[labels == cluster]
            if len(members):
                centroids[cluster] = members.mean(axis=0)

    return [int(label) for label in labels]
