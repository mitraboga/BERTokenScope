from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go


def attention_heatmap(
    tokens: Sequence[str],
    attention_matrix: np.ndarray,
    *,
    title: str = "Attention Heatmap",
) -> go.Figure:
    matrix = np.asarray(attention_matrix, dtype=float)
    if matrix.shape != (len(tokens), len(tokens)):
        raise ValueError("attention_matrix must be square and match token count.")

    return go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=list(tokens),
            y=list(tokens),
            colorscale="Greys",
            reversescale=False,
            zmin=0,
            zmax=1,
            hovertemplate="Source=%{y}<br>Target=%{x}<br>Score=%{z:.4f}<extra></extra>",
        ),
        layout=go.Layout(
            title=title,
            xaxis_title="Attended token",
            yaxis_title="Source token",
            height=560,
            margin=dict(l=80, r=30, t=60, b=80),
        ),
    )
