"""Detector orchestration — ties CLIP + memory bank + heat-map together."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .clip_engine import get_engine
from .heatmap import build_overlay
from .memory_bank import MemoryBank
from .tiling import tile_image

_IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass
class InspectionResult:
    score: float
    threshold: float
    verdict: str          # "PASS" | "DEFECT"
    confidence: float     # 0..1, distance from threshold
    overlay: str          # base64 PNG
    tile_grid: list[list[float]] = field(default_factory=list)


class Detector:
    def __init__(self) -> None:
        self.engine = get_engine()
        self.bank: MemoryBank | None = MemoryBank.load()

    # --- gallery -----------------------------------------------------------
    def build_gallery(self, image_paths: list[Path]) -> dict:
        """Embed every tile of every known-good image into the memory bank."""
        engine = self.engine
        bank = MemoryBank(engine.embed_dim)
        per_image_scores: list[float] = []

        all_emb: list[np.ndarray] = []
        cached: list[tuple[list, np.ndarray]] = []
        for p in image_paths:
            img = Image.open(p)
            _, tiles = tile_image(img)
            emb = engine.encode([t.image for t in tiles])
            all_emb.append(emb)
            cached.append((tiles, emb))
        bank.add(np.vstack(all_emb))

        # Leave-in calibration: score each normal image against the bank.
        for _, emb in cached:
            per_image_scores.append(float(bank.anomaly_scores(emb).max()))
        bank.calibrate(per_image_scores)
        bank.save()
        self.bank = bank
        return {
            "images": len(image_paths),
            "tiles": int(bank.size),
            "threshold": bank.threshold,
            "calibration": bank._calib,
        }

    # --- inspection --------------------------------------------------------
    def inspect(self, image: Image.Image) -> InspectionResult:
        if self.bank is None or self.bank.size == 0:
            raise RuntimeError("No gallery built. POST known-good images to /gallery/build first.")
        canvas, tiles = tile_image(image)
        emb = self.engine.encode([t.image for t in tiles])
        tile_scores = self.bank.anomaly_scores(emb)

        # Image-level score: blend global context with worst local region.
        global_s = float(tile_scores[0])
        local_max = float(tile_scores[1:].max())
        score = 0.35 * global_s + 0.65 * local_max

        thr = self.bank.threshold
        verdict = "DEFECT" if score > thr else "PASS"
        confidence = float(min(1.0, abs(score - thr) / (thr + 1e-6)))
        overlay = build_overlay(canvas, tiles, tile_scores, thr)

        g = int(np.sqrt(len(tiles) - 1))
        grid = tile_scores[1:].reshape(g, g).round(4).tolist()
        return InspectionResult(
            score=round(score, 5),
            threshold=round(thr, 5),
            verdict=verdict,
            confidence=round(confidence, 4),
            overlay=overlay,
            tile_grid=grid,
        )

    # --- benchmark ---------------------------------------------------------
    def benchmark(self, good_dir: Path, defect_dir: Path) -> dict:
        """Evaluate against a held-out test set → AUROC + confusion matrix."""
        scores, labels = [], []
        for d, lbl in ((good_dir, 0), (defect_dir, 1)):
            if not d.exists():
                continue
            for p in sorted(d.iterdir()):
                if p.suffix.lower() not in _IMG_EXT:
                    continue
                res = self.inspect(Image.open(p))
                scores.append(res.score)
                labels.append(lbl)
        if not scores:
            return {"error": "no test images found"}

        s = np.asarray(scores)
        y = np.asarray(labels)
        thr = self.bank.threshold  # type: ignore[union-attr]
        pred = (s > thr).astype(int)

        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        acc = (tp + tn) / len(y)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

        return {
            "n": len(y),
            "auroc": round(_auroc(y, s), 4),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "threshold": round(thr, 5),
        }


def _auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUROC (Mann–Whitney U) — no sklearn dependency needed."""
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype="float64")
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = y_true == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


_detector: Detector | None = None


def get_detector() -> Detector:
    global _detector
    if _detector is None:
        _detector = Detector()
    return _detector
