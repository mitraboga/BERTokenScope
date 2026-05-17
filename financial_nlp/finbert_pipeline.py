from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ber_tokenscope.schemas import FinancialAnalysis, FinancialSignal
from ber_tokenscope.settings import get_settings
from financial_nlp.sentiment import FinancialTextAnalyzer


class FinancialAnalyzer(Protocol):
    model_name: str

    def analyze(self, text: str) -> FinancialAnalysis: ...


class FinBERTUnavailableError(RuntimeError):
    """Raised when a live FinBERT pipeline cannot be initialized."""


@dataclass
class FinBERTSentimentAnalyzer:
    """Lazy FinBERT sentiment adapter with lexical risk overlays."""

    model_name: str = "ProsusAI/finbert"

    def __post_init__(self) -> None:
        self._pipeline = None
        self._baseline = FinancialTextAnalyzer()

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline
        except Exception as exc:  # pragma: no cover - depends on local env
            raise FinBERTUnavailableError(
                "Install transformers and model weights to run FinBERT analysis."
            ) from exc

        try:
            settings = get_settings()
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=settings.model_serving.cache_dir,
                local_files_only=not settings.model_serving.allow_downloads,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                cache_dir=settings.model_serving.cache_dir,
                local_files_only=not settings.model_serving.allow_downloads,
            )
            self._pipeline = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                truncation=True,
            )
        except Exception as exc:  # pragma: no cover - downloads/env dependent
            raise FinBERTUnavailableError(
                f"Unable to initialize FinBERT model {self.model_name}."
            ) from exc

    def analyze(self, text: str) -> FinancialAnalysis:
        self._load()
        assert self._pipeline is not None

        baseline = self._baseline.analyze(text)
        prediction = self._pipeline(text[:4000])[0]
        label = str(prediction["label"]).lower()
        confidence = float(prediction["score"])

        return FinancialAnalysis(
            text=text,
            sentiment=FinancialSignal(
                label=label,
                score=round(confidence, 4),
                explanation=(
                    "FinBERT classified financial tone; lexical overlays provide "
                    "risk, uncertainty, and optimism diagnostics."
                ),
            ),
            risk_score=baseline.risk_score,
            uncertainty_score=baseline.uncertainty_score,
            optimism_score=baseline.optimism_score,
            matched_terms=baseline.matched_terms,
            model_name=self.model_name,
        )


class HybridFinancialAnalyzer:
    """Use FinBERT when available; otherwise fall back to deterministic finance lexicons."""

    def __init__(
        self, prefer_finbert: bool = False, model_name: str = "ProsusAI/finbert"
    ) -> None:
        self.prefer_finbert = prefer_finbert
        self._baseline = FinancialTextAnalyzer()
        self._finbert = FinBERTSentimentAnalyzer(model_name=model_name)
        self.model_name = model_name if prefer_finbert else self._baseline.model_name

    def analyze(self, text: str) -> FinancialAnalysis:
        if not self.prefer_finbert:
            return self._baseline.analyze(text)
        try:
            return self._finbert.analyze(text)
        except FinBERTUnavailableError:
            return self._baseline.analyze(text)
