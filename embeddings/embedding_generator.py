from __future__ import annotations

import hashlib
import re

import numpy as np

from ber_tokenscope.settings import get_settings

TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z\-']*")


class DeterministicEmbeddingGenerator:
    """Feature-hashed token embeddings for offline demos and tests."""

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            vector = np.zeros(self.dimensions, dtype=float)
            tokens = [token.lower() for token in TOKEN_PATTERN.findall(text)]
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], byteorder="big") % self.dimensions
                sign = 1 if digest[4] % 2 == 0 else -1
                vector[index] += sign
            norm = np.linalg.norm(vector)
            if norm:
                vector = vector / norm
            vectors.append(vector)
        return np.vstack(vectors)


class TransformerEmbeddingGenerator:
    """Lazy transformer embedding adapter using mean pooling."""

    def __init__(self, model_name: str, max_length: int = 256) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Install torch and transformers for live embeddings."
            ) from exc

        settings = get_settings()
        self._torch = torch
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=settings.model_serving.cache_dir,
                local_files_only=not settings.model_serving.allow_downloads,
            )
            self._model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=settings.model_serving.cache_dir,
                local_files_only=not settings.model_serving.allow_downloads,
            )
        except Exception as exc:  # pragma: no cover - depends on model cache
            raise RuntimeError(
                "Live embedding model weights are unavailable. "
                "Enable downloads or use deterministic feature hashing."
            ) from exc
        self._model.eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        self._load()
        tokenizer = self._tokenizer
        model = self._model
        torch = self._torch
        assert tokenizer is not None and model is not None

        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = model(**encoded)

        hidden = output.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
        summed = torch.sum(hidden * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        embeddings = summed / counts
        return embeddings.detach().cpu().numpy()


class HybridEmbeddingGenerator:
    """Use transformer embeddings when requested; otherwise deterministic fallback."""

    def __init__(
        self,
        model_name: str,
        *,
        prefer_transformer: bool = False,
        dimensions: int = 128,
    ) -> None:
        self.model_name = (
            model_name if prefer_transformer else "deterministic-feature-hashing"
        )
        self.prefer_transformer = prefer_transformer
        self._transformer = TransformerEmbeddingGenerator(model_name)
        self._fallback = DeterministicEmbeddingGenerator(dimensions=dimensions)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self.prefer_transformer:
            return self._fallback.encode(texts)
        try:
            return self._transformer.encode(texts)
        except RuntimeError:
            self.model_name = "deterministic-feature-hashing"
            return self._fallback.encode(texts)
