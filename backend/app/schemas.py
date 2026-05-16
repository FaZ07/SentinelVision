"""Pydantic response models — typed API contract."""
from __future__ import annotations

from pydantic import BaseModel


class Health(BaseModel):
    status: str
    model: str
    device: str
    gallery_ready: bool
    gallery_tiles: int


class GalleryStatus(BaseModel):
    images: int
    tiles: int
    threshold: float
    calibration: dict | None = None


class Inspection(BaseModel):
    score: float
    threshold: float
    verdict: str
    confidence: float
    overlay: str
    tile_grid: list[list[float]]


class Benchmark(BaseModel):
    n: int
    auroc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: dict
    threshold: float


class OnnxExport(BaseModel):
    path: str
    size_mb: float
    max_abs_diff: float
    parity_ok: bool
