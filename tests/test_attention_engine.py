import numpy as np
import pytest

from attention.engine import (
    AttentionEngine,
    attention_layers_to_numpy,
    get_color_for_attention_score,
    get_mask_token_index,
)
from attention.metrics import attention_entropy, focus_score
from attention.rollout import attention_rollout
from attention.static_diagram import export_attention_diagram
from attention.token_analysis import strongest_attention_links


def test_get_mask_token_index_returns_zero_based_index():
    assert get_mask_token_index(103, [101, 2023, 103, 102]) == 2


def test_get_mask_token_index_returns_none_when_missing():
    assert get_mask_token_index(103, [101, 2023, 2003, 102]) is None


def test_get_color_for_attention_score_scales_grayscale():
    assert get_color_for_attention_score(0) == (0, 0, 0)
    assert get_color_for_attention_score(1) == (255, 255, 255)
    assert get_color_for_attention_score(0.5) == (128, 128, 128)


def test_get_color_for_attention_score_rejects_invalid_values():
    with pytest.raises(ValueError):
        get_color_for_attention_score(1.2)


def test_strongest_attention_links_returns_sorted_links():
    tokens = ["[CLS]", "bank", "river", "[SEP]"]
    matrix = np.array(
        [
            [0.1, 0.2, 0.6, 0.1],
            [0.0, 0.1, 0.8, 0.1],
            [0.0, 0.7, 0.2, 0.1],
            [0.1, 0.1, 0.1, 0.7],
        ]
    )
    links = strongest_attention_links(tokens, matrix, limit=2)
    assert [(link.source, link.target) for link in links] == [
        ("bank", "river"),
        ("river", "bank"),
    ]


def test_attention_rollout_preserves_shape():
    layers = [
        np.array([[0.8, 0.2], [0.4, 0.6]]),
        np.array([[0.5, 0.5], [0.1, 0.9]]),
    ]
    assert attention_rollout(layers).shape == (2, 2)


def test_attention_metrics_are_normalized():
    matrix = np.array([[0.9, 0.1], [0.5, 0.5]])

    assert 0 <= attention_entropy(matrix) <= 1
    assert 0 <= focus_score(matrix) <= 1


def test_attention_layers_to_numpy_removes_batch_dimension():
    attentions = (np.zeros((1, 2, 3, 3)),)

    layers = attention_layers_to_numpy(attentions)

    assert len(layers) == 1
    assert layers[0].shape == (2, 3, 3)


def test_attention_engine_explores_offline_attention():
    result = AttentionEngine(offline=True).explore_attention(
        "The bank was muddy because the river overflowed near the [MASK].",
        layer=1,
        head=1,
    )

    assert result.layer_count == 2
    assert result.head_count == 4
    assert result.strongest_links
    assert len(result.attention_matrix) == len(result.tokens)


def test_export_attention_diagram_writes_png(tmp_path):
    file_path = export_attention_diagram(
        ["bank", "river"],
        np.array([[0.1, 0.9], [0.8, 0.2]]),
        layer=1,
        head=2,
        output_dir=tmp_path,
    )

    assert file_path.exists()
    assert file_path.suffix == ".png"
