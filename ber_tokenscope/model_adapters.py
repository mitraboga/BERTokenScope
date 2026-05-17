from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ber_tokenscope.schemas import MaskPrediction
from ber_tokenscope.settings import get_settings


class ModelDependencyError(RuntimeError):
    """Raised when optional ML dependencies are unavailable."""


@dataclass(frozen=True)
class MaskedLMOutput:
    predictions: list[MaskPrediction]
    tokens: list[str]
    attentions: tuple[Any, ...] | None


class HuggingFaceMaskedLM:
    """Lazy Hugging Face masked-language-model adapter."""

    def __init__(self, model_name: str, max_length: int = 128) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - depends on local env
            raise ModelDependencyError(
                "Install torch and transformers to run live masked-token inference."
            ) from exc

        settings = get_settings()
        self._torch = torch
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=settings.model_serving.cache_dir,
                local_files_only=not settings.model_serving.allow_downloads,
            )
            self._model = AutoModelForMaskedLM.from_pretrained(
                self.model_name,
                output_attentions=True,
                cache_dir=settings.model_serving.cache_dir,
                local_files_only=not settings.model_serving.allow_downloads,
            )
        except Exception as exc:  # pragma: no cover - depends on model cache
            raise ModelDependencyError(
                "Live masked-token model weights are unavailable. "
                "Enable downloads or use the deterministic fallback."
            ) from exc
        self._model.eval()

    def predict(self, text: str, top_k: int = 5) -> MaskedLMOutput:
        self._load()
        torch = self._torch
        tokenizer = self._tokenizer
        model = self._model
        assert tokenizer is not None and model is not None

        if tokenizer.mask_token not in text:
            raise ValueError(f"Input must include {tokenizer.mask_token}.")

        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        mask_positions = (encoded["input_ids"] == tokenizer.mask_token_id).nonzero(
            as_tuple=False
        )
        if len(mask_positions) != 1:
            raise ValueError("Input must contain exactly one mask token.")

        with torch.no_grad():
            outputs = model(**encoded)

        mask_index = mask_positions[0, 1]
        logits = outputs.logits[0, mask_index]
        probabilities = torch.softmax(logits, dim=0)
        top = torch.topk(probabilities, top_k)
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])

        predictions: list[MaskPrediction] = []
        for token_id, probability in zip(
            top.indices.tolist(),
            top.values.tolist(),
            strict=True,
        ):
            token = tokenizer.decode([token_id]).strip()
            sequence = text.replace(tokenizer.mask_token, token, 1)
            predictions.append(
                MaskPrediction(
                    token=token,
                    sequence=sequence,
                    probability=float(probability),
                )
            )

        return MaskedLMOutput(
            predictions=predictions,
            tokens=list(tokens),
            attentions=outputs.attentions,
        )


class DeterministicMaskedLM:
    """Small offline fallback used for demos and tests."""

    finance_predictions = ["earnings", "revenue", "profits", "growth", "cash"]
    general_predictions = ["field", "area", "place", "room", "road"]

    def __init__(self, model_name: str = "deterministic-fallback") -> None:
        self.model_name = model_name

    def predict(self, text: str, top_k: int = 5) -> MaskedLMOutput:
        if "[MASK]" not in text:
            raise ValueError("Input must include [MASK].")

        lower = text.lower()
        vocabulary = (
            self.finance_predictions
            if any(term in lower for term in ["company", "quarter", "revenue"])
            else self.general_predictions
        )
        selected = vocabulary[:top_k]
        weights = np.linspace(1.0, 0.35, num=len(selected))
        weights = weights / weights.sum()
        predictions = [
            MaskPrediction(
                token=token,
                sequence=text.replace("[MASK]", token, 1),
                probability=float(weight),
            )
            for token, weight in zip(selected, weights, strict=True)
        ]
        tokens = text.replace(".", " .").replace(",", " ,").split()
        attentions = self._synthetic_attentions(tokens)
        return MaskedLMOutput(
            predictions=predictions, tokens=tokens, attentions=attentions
        )

    @staticmethod
    def _synthetic_attentions(tokens: list[str]) -> tuple[np.ndarray, ...]:
        """Create deterministic attention tensors for offline exploration."""

        token_count = len(tokens)
        layers = []
        lower_tokens = [token.lower() for token in tokens]
        finance_terms = {"bank", "river", "company", "revenue", "earnings", "quarter"}

        for _layer_index in range(2):
            heads = []
            for head_index in range(4):
                matrix = np.full((token_count, token_count), 1 / token_count)
                for source_index in range(token_count):
                    if head_index == 0:
                        target_index = min(source_index + 1, token_count - 1)
                    elif head_index == 1:
                        target_index = max(source_index - 1, 0)
                    elif (
                        head_index == 2 and lower_tokens[source_index] in finance_terms
                    ):
                        candidates = [
                            index
                            for index, token in enumerate(lower_tokens)
                            if token in finance_terms and index != source_index
                        ]
                        target_index = candidates[0] if candidates else source_index
                    else:
                        target_index = source_index

                    matrix[source_index] *= 0.25
                    matrix[source_index, target_index] += 0.75
                    matrix[source_index] /= matrix[source_index].sum()
                heads.append(matrix)
            layers.append(np.expand_dims(np.stack(heads), axis=0))

        return tuple(layers)
