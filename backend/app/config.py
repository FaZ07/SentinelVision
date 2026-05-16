"""Central configuration. All knobs in one place — no magic numbers scattered around."""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)


class Settings(BaseModel):
    # --- CLIP backbone -----------------------------------------------------
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"  # strong open weights, no API
    device: str = "cpu"  # auto-upgraded to "cuda" if available at runtime

    # --- Tiling / localization --------------------------------------------
    # Image is resized to `canvas`, then split into a grid of `grid x grid`
    # overlapping tiles. Each tile is embedded by CLIP and compared to the
    # normal memory bank — this gives PatchCore-style spatial localization
    # without any training.
    canvas: int = 448
    grid: int = 4              # 4x4 = 16 local tiles + 1 global view
    tile_overlap: float = 0.25

    # --- Memory bank -------------------------------------------------------
    # k nearest normal tiles averaged → robust anomaly distance.
    knn: int = 3
    # Coreset subsampling keeps the bank small & fast (PatchCore trick).
    coreset_ratio: float = 1.0  # 1.0 = keep all (small demo galleries)

    # --- Decision threshold ------------------------------------------------
    # Calibrated as mean + k*std of normal-image scores. User-tunable in UI.
    threshold_sigma: float = 3.0

    artifact_dir: Path = ARTIFACT_DIR
    data_dir: Path = DATA_DIR


settings = Settings()
