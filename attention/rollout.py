from __future__ import annotations

import numpy as np


def attention_rollout(layer_attentions: list[np.ndarray]) -> np.ndarray:
    """Aggregate attention across layers using residual-aware rollout."""

    if not layer_attentions:
        raise ValueError("layer_attentions must not be empty.")

    size = layer_attentions[0].shape[-1]
    result = np.eye(size)
    for layer in layer_attentions:
        if layer.shape != (size, size):
            raise ValueError("all attention matrices must have the same square shape.")
        residual = layer + np.eye(size)
        residual = residual / residual.sum(axis=-1, keepdims=True)
        result = residual @ result
    return result


def average_heads(layer_attention: np.ndarray) -> np.ndarray:
    """Average a layer's attention heads into a single token-token matrix."""

    if layer_attention.ndim != 3:
        raise ValueError("layer_attention must have shape heads x tokens x tokens.")
    return layer_attention.mean(axis=0)
