from __future__ import annotations

from dataclasses import dataclass

from ber_tokenscope.schemas import (
    EmbeddingDocument,
    EmbeddingPoint,
    EmbeddingReport,
    EmbeddingSimilarityPair,
)
from ber_tokenscope.settings import get_settings
from embeddings.clustering import assign_clusters
from embeddings.dimensionality_reduction import reduce_to_2d
from embeddings.embedding_generator import HybridEmbeddingGenerator
from embeddings.similarity import top_similarity_pairs


@dataclass
class EmbeddingEngine:
    prefer_transformer: bool = False

    def __post_init__(self) -> None:
        settings = get_settings()
        self.generator = HybridEmbeddingGenerator(
            settings.models.sentence_embedding,
            prefer_transformer=self.prefer_transformer,
        )

    def analyze(
        self,
        documents: list[EmbeddingDocument],
        *,
        cluster_count: int = 3,
        similarity_limit: int = 10,
    ) -> EmbeddingReport:
        if not documents:
            raise ValueError("documents must not be empty.")

        texts = [document.text for document in documents]
        labels = [document.label for document in documents]
        vectors = self.generator.encode(texts)
        coordinates = reduce_to_2d(vectors)
        clusters = assign_clusters(vectors, cluster_count)
        pairs = top_similarity_pairs(labels, vectors, limit=similarity_limit)

        points = [
            EmbeddingPoint(
                label=document.label,
                text=document.text,
                metadata=document.metadata,
                x=round(float(coordinates[index, 0]), 6),
                y=round(float(coordinates[index, 1]), 6),
                cluster=clusters[index],
            )
            for index, document in enumerate(documents)
        ]
        similarities = [
            EmbeddingSimilarityPair(
                left=left,
                right=right,
                similarity=round(score, 6),
            )
            for left, right, score in pairs
        ]
        return EmbeddingReport(
            model_name=self.generator.model_name,
            dimensions=int(vectors.shape[1]),
            points=points,
            similarities=similarities,
        )
