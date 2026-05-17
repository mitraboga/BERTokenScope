import numpy as np

from ber_tokenscope.schemas import EmbeddingDocument
from embeddings.clustering import assign_clusters
from embeddings.dimensionality_reduction import reduce_to_2d
from embeddings.embedding_generator import DeterministicEmbeddingGenerator
from embeddings.engine import EmbeddingEngine
from embeddings.similarity import cosine_similarity_matrix, top_similarity_pairs


def test_deterministic_embeddings_make_related_texts_more_similar():
    generator = DeterministicEmbeddingGenerator(dimensions=64)
    vectors = generator.encode(
        [
            "revenue growth demand",
            "strong revenue growth",
            "litigation risk recession",
        ]
    )
    matrix = cosine_similarity_matrix(vectors)

    assert matrix[0, 1] > matrix[0, 2]


def test_reduce_to_2d_returns_two_coordinates():
    vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)

    coordinates = reduce_to_2d(vectors)

    assert coordinates.shape == (3, 2)


def test_assign_clusters_matches_document_count():
    labels = assign_clusters(np.array([[1, 0], [0.9, 0.1], [-1, 0]]), cluster_count=2)

    assert len(labels) == 3
    assert set(labels) <= {0, 1}


def test_top_similarity_pairs_returns_ranked_pairs():
    vectors = np.array([[1, 0], [0.9, 0.1], [-1, 0]], dtype=float)

    pairs = top_similarity_pairs(["A", "B", "C"], vectors, limit=1)

    assert pairs[0][0:2] == ("A", "B")


def test_embedding_engine_generates_report():
    report = EmbeddingEngine().analyze(
        [
            EmbeddingDocument(label="A", text="revenue growth demand"),
            EmbeddingDocument(label="B", text="strong revenue growth"),
            EmbeddingDocument(label="C", text="litigation risk recession"),
        ],
        cluster_count=2,
    )

    assert report.model_name == "deterministic-feature-hashing"
    assert len(report.points) == 3
    assert report.similarities
