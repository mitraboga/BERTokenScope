from __future__ import annotations

from ber_tokenscope.schemas import ModelRegistryEntry

MODEL_REGISTRY = [
    ModelRegistryEntry(
        model_id="bert-base-uncased",
        display_name="BERT Base",
        task="masked-language",
        fallback="deterministic-fallback",
    ),
    ModelRegistryEntry(
        model_id="distilbert-base-uncased",
        display_name="DistilBERT",
        task="masked-language",
        fallback="deterministic-fallback",
    ),
    ModelRegistryEntry(
        model_id="roberta-base",
        display_name="RoBERTa",
        task="masked-language",
        fallback="deterministic-fallback",
    ),
    ModelRegistryEntry(
        model_id="ProsusAI/finbert",
        display_name="FinBERT",
        task="financial-sentiment",
        fallback="lexical-financial-baseline",
    ),
    ModelRegistryEntry(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        display_name="MiniLM Sentence Transformer",
        task="embedding",
        fallback="deterministic-feature-hashing",
    ),
]


def registry_by_id() -> dict[str, ModelRegistryEntry]:
    return {entry.model_id: entry for entry in MODEL_REGISTRY}
