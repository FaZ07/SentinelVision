"""Turn per-tile anomaly scores into an explainable heat-map overlay.

This is the "explainability" pillar: an operator doesn't just see PASS/FAIL,
they see *exactly where* on the part the model thinks the defect is.
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageFilter

from .config import settings
from .tiling import Tile


def _jet(x: np.ndarray) -> np.ndarray:
    """Minimal JET colormap (keeps matplotlib out of the dependency tree)."""
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def _gaussian(img: Image.Image, radius: float) -> Image.Image:
    try:
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    except Exception:
        return img


def build_overlay(
    canvas: Image.Image,
    tiles: list[Tile],
    tile_scores: np.ndarray,
    threshold: float,
) -> str:
    """Return a base64 PNG: original part with a translucent anomaly heat-map.

    `tile_scores` aligns with `tiles`; index 0 is the global tile and is
    excluded from spatial localisation (it only feeds the image-level score).
    """
    c = settings.canvas
    accum = np.zeros((c, c), dtype="float32")
    weight = np.zeros((c, c), dtype="float32")

    for tile, score in zip(tiles[1:], tile_scores[1:]):
        left, t, r, b = tile.box
        accum[t:b, left:r] += float(score)
        weight[t:b, left:r] += 1.0
    weight[weight == 0] = 1.0
    field = accum / weight

    # Centre the colour scale on the decision threshold so it is meaningful.
    lo, hi = threshold * 0.5, threshold * 1.8
    norm = (field - lo) / (hi - lo + 1e-8)

    heat_rgb = (_jet(norm) * 255).astype("uint8")
    heat_img = _gaussian(Image.fromarray(heat_rgb), radius=c / 64)

    blended = Image.blend(canvas.convert("RGB"), heat_img, alpha=0.45)
    return _to_b64(blended)


def _to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
