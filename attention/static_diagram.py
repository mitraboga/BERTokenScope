from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np


def export_attention_diagram(
    tokens: Sequence[str],
    attention_matrix: np.ndarray,
    *,
    layer: int,
    head: int,
    output_dir: str | Path = "artifacts/attention",
) -> Path:
    """Export a CS50AI-style grayscale attention diagram as a PNG."""

    if attention_matrix.shape != (len(tokens), len(tokens)):
        raise ValueError("attention_matrix must be square and match token count.")

    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("Install pillow to export attention diagrams.") from exc

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cell_size = 42
    label_width = 140
    label_height = 90
    width = label_width + (cell_size * len(tokens))
    height = label_height + (cell_size * len(tokens))

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.text((12, 12), f"Layer {layer}, Head {head}", fill="black", font=font)
    for index, token in enumerate(tokens):
        x = label_width + (index * cell_size)
        y = label_height + (index * cell_size)
        draw.text((x + 4, label_height - 24), token[:14], fill="black", font=font)
        draw.text((8, y + 14), token[:18], fill="black", font=font)

    for row in range(len(tokens)):
        for column in range(len(tokens)):
            value = int(round(float(attention_matrix[row, column]) * 255))
            color = (value, value, value)
            x0 = label_width + (column * cell_size)
            y0 = label_height + (row * cell_size)
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            draw.rectangle((x0, y0, x1, y1), fill=color, outline=(210, 210, 210))

    file_path = output_path / f"layer_{layer:02d}_head_{head:02d}.png"
    image.save(file_path)
    return file_path
