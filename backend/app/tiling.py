"""Deterministic image tiling for spatial anomaly localisation.

The image is resized to a square canvas, then split into an overlapping
`grid x grid` lattice of tiles. Tile centres are tracked so per-tile anomaly
scores can be splatted back onto a full-resolution heat-map.
"""
from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from .config import settings


@dataclass(frozen=True)
class Tile:
    image: Image.Image
    row: int
    col: int
    box: tuple[int, int, int, int]  # (left, top, right, bottom) on the canvas


def make_canvas(image: Image.Image) -> Image.Image:
    return image.convert("RGB").resize((settings.canvas, settings.canvas), Image.BILINEAR)


def tile_image(image: Image.Image) -> tuple[Image.Image, list[Tile]]:
    """Return (canvas, tiles). tiles[0] is the *global* view; the rest form
    the local grid used for localisation."""
    canvas = make_canvas(image)
    g = settings.grid
    c = settings.canvas
    step = c / g
    size = int(step * (1 + settings.tile_overlap))
    size = min(size, c)

    tiles: list[Tile] = [Tile(canvas, -1, -1, (0, 0, c, c))]  # global context
    for r in range(g):
        for col in range(g):
            cx = (col + 0.5) * step
            cy = (r + 0.5) * step
            left = int(max(0, cx - size / 2))
            top = int(max(0, cy - size / 2))
            right = int(min(c, left + size))
            bottom = int(min(c, top + size))
            crop = canvas.crop((left, top, right, bottom))
            tiles.append(Tile(crop, r, col, (left, top, right, bottom)))
    return canvas, tiles
