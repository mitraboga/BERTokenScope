from __future__ import annotations

from typing import Any

import numpy as np

from attention.metrics import attention_entropy, focus_score
from attention.rollout import attention_rollout, average_heads
from attention.static_diagram import export_attention_diagram
from attention.token_analysis import strongest_attention_links
from ber_tokenscope.model_adapters import (
    DeterministicMaskedLM,
    HuggingFaceMaskedLM,
    MaskedLMOutput,
    ModelDependencyError,
)
from ber_tokenscope.schemas import (
    AttentionExplorerResult,
    AttentionHeadSummary,
    MaskedLMResult,
)
from ber_tokenscope.settings import get_settings


def get_mask_token_index(mask_token_id: int, input_ids: list[int]) -> int | None:
    """CS50AI-compatible helper for locating the mask token."""

    try:
        return input_ids.index(mask_token_id)
    except ValueError:
        return None


def get_color_for_attention_score(score: float) -> tuple[int, int, int]:
    """Map an attention score in [0, 1] to a grayscale RGB tuple."""

    if not 0 <= score <= 1:
        raise ValueError("attention score must be between 0 and 1.")
    channel = round(score * 255)
    return channel, channel, channel


class AttentionEngine:
    def __init__(self, model_name: str | None = None, offline: bool = False) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.models.masked_lm
        self.top_k = settings.runtime.top_k
        self.max_length = settings.runtime.max_sequence_length
        self.offline = offline

    def _adapter(self) -> HuggingFaceMaskedLM | DeterministicMaskedLM:
        if self.offline:
            return DeterministicMaskedLM()
        return HuggingFaceMaskedLM(self.model_name, max_length=self.max_length)

    def analyze_mask(self, text: str, top_k: int | None = None) -> MaskedLMResult:
        adapter = self._adapter()
        try:
            output = adapter.predict(text, top_k=top_k or self.top_k)
            model_name = getattr(adapter, "model_name", self.model_name)
        except ModelDependencyError:
            fallback = DeterministicMaskedLM()
            output = fallback.predict(text, top_k=top_k or self.top_k)
            model_name = fallback.model_name

        links = []
        matrix = first_attention_matrix(output.attentions)
        if matrix is not None:
            links = strongest_attention_links(output.tokens, matrix)

        return MaskedLMResult(
            text=text,
            model_name=model_name,
            predictions=output.predictions,
            tokens=output.tokens,
            strongest_links=links,
        )

    def explore_attention(
        self,
        text: str,
        *,
        layer: int = 1,
        head: int = 1,
        export_diagram: bool = False,
    ) -> AttentionExplorerResult:
        output, model_name = self._run_prediction(text)
        attention_layers = attention_layers_to_numpy(output.attentions)
        if not attention_layers:
            raise ValueError("Attention tensors are unavailable for this model run.")

        layer_count = len(attention_layers)
        head_count = attention_layers[0].shape[0]
        if not 1 <= layer <= layer_count:
            raise ValueError(f"layer must be between 1 and {layer_count}.")
        if not 1 <= head <= head_count:
            raise ValueError(f"head must be between 1 and {head_count}.")

        selected_matrix = attention_layers[layer - 1][head - 1]
        rollout_layers = [
            average_heads(layer_attention) for layer_attention in attention_layers
        ]
        rollout_matrix = attention_rollout(rollout_layers)
        links = strongest_attention_links(output.tokens, selected_matrix)
        summaries = summarize_attention_heads(attention_layers)

        diagram_path = None
        if export_diagram:
            diagram_path = str(
                export_attention_diagram(
                    output.tokens,
                    selected_matrix,
                    layer=layer,
                    head=head,
                )
            )

        return AttentionExplorerResult(
            text=text,
            model_name=model_name,
            tokens=output.tokens,
            layer_count=layer_count,
            head_count=head_count,
            selected_layer=layer,
            selected_head=head,
            attention_matrix=selected_matrix.round(6).tolist(),
            rollout_matrix=rollout_matrix.round(6).tolist(),
            strongest_links=links,
            head_summaries=summaries,
            diagram_path=diagram_path,
        )

    def _run_prediction(self, text: str) -> tuple[MaskedLMOutput, str]:
        adapter = self._adapter()
        try:
            output = adapter.predict(text, top_k=self.top_k)
            model_name = getattr(adapter, "model_name", self.model_name)
        except ModelDependencyError:
            fallback = DeterministicMaskedLM()
            output = fallback.predict(text, top_k=self.top_k)
            model_name = fallback.model_name
        return output, model_name


def first_attention_matrix(attentions: tuple[Any, ...] | None) -> np.ndarray | None:
    if not attentions:
        return None
    matrix = attentions[0][0][0]
    if hasattr(matrix, "detach"):
        matrix = matrix.detach().cpu().numpy()
    return np.asarray(matrix, dtype=float)


def attention_layers_to_numpy(attentions: tuple[Any, ...] | None) -> list[np.ndarray]:
    """Convert model attentions to layer arrays with shape heads x tokens x tokens."""

    if not attentions:
        return []

    layers = []
    for layer in attentions:
        layer_array = layer
        if hasattr(layer_array, "detach"):
            layer_array = layer_array.detach().cpu().numpy()
        layer_array = np.asarray(layer_array, dtype=float)
        if layer_array.ndim != 4 or layer_array.shape[0] != 1:
            raise ValueError(
                "attention tensors must have shape batch x heads x tokens x tokens."
            )
        layers.append(layer_array[0])
    return layers


def summarize_attention_heads(
    attention_layers: list[np.ndarray],
) -> list[AttentionHeadSummary]:
    summaries: list[AttentionHeadSummary] = []
    for layer_index, layer_attention in enumerate(attention_layers, start=1):
        for head_index, matrix in enumerate(layer_attention, start=1):
            summaries.append(
                AttentionHeadSummary(
                    layer=layer_index,
                    head=head_index,
                    entropy=round(attention_entropy(matrix), 6),
                    focus_score=round(focus_score(matrix), 6),
                )
            )
    return summaries
