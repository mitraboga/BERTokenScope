from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    display_name: str
    model_name: str
    task: str
    offline: bool = True
    notes: tuple[str, ...] = ()


MASKED_LM_PROFILES = [
    ModelProfile(
        display_name="BERT Base",
        model_name="bert-base-uncased",
        task="masked-language",
        notes=("general-purpose BERT baseline",),
    ),
    ModelProfile(
        display_name="DistilBERT",
        model_name="distilbert-base-uncased",
        task="masked-language",
        notes=("smaller and faster BERT-family model",),
    ),
    ModelProfile(
        display_name="RoBERTa",
        model_name="roberta-base",
        task="masked-language",
        notes=("robustly optimized transformer baseline",),
    ),
]


FINANCIAL_PROFILES = [
    ModelProfile(
        display_name="Financial Lexical Baseline",
        model_name="lexical-financial-baseline",
        task="financial-sentiment",
        notes=("deterministic risk, uncertainty, and optimism scoring",),
    ),
    ModelProfile(
        display_name="FinBERT",
        model_name="ProsusAI/finbert",
        task="financial-sentiment",
        notes=(
            "finance-domain transformer sentiment model when weights are available",
        ),
    ),
]


EMBEDDING_PROFILES = [
    ModelProfile(
        display_name="Feature Hashing",
        model_name="deterministic-feature-hashing",
        task="embedding",
        notes=("offline deterministic embedding baseline",),
    ),
    ModelProfile(
        display_name="MiniLM Sentence Transformer",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        task="embedding",
        notes=("compact transformer embedding model when weights are available",),
    ),
]
