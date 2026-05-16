"""FAISS memory bank — the heart of the zero-shot detector.

We store CLIP embeddings of *normal* tiles only. At inspection time a tile is
anomalous if its nearest normal neighbours are far away in cosine space.
This is the PatchCore idea, but with a frozen CLIP backbone instead of a
WideResNet, so it needs zero training and zero defect labels.
"""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from .config import settings


class MemoryBank:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # inner product == cosine (L2-norm'd)
        self.size = 0
        self._calib: dict | None = None  # threshold calibration stats

    # --- build -------------------------------------------------------------
    def add(self, embeddings: np.ndarray) -> None:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(f"expected (N,{self.dim}), got {embeddings.shape}")
        bank = self._coreset(embeddings)
        self.index.add(bank)
        self.size += bank.shape[0]

    def _coreset(self, x: np.ndarray) -> np.ndarray:
        """Greedy-ish coreset subsample (keeps the bank fast for big galleries)."""
        ratio = settings.coreset_ratio
        if ratio >= 1.0 or x.shape[0] <= 16:
            return x
        keep = max(16, int(x.shape[0] * ratio))
        rng = np.random.default_rng(42)
        idx = rng.choice(x.shape[0], size=keep, replace=False)
        return x[idx]

    # --- query -------------------------------------------------------------
    def anomaly_scores(self, embeddings: np.ndarray) -> np.ndarray:
        """Return per-row anomaly score in [0, 2].

        score = 1 - mean(cosine similarity to k nearest normal tiles).
        0 → identical to known-good, higher → more anomalous.
        """
        if self.size == 0:
            raise RuntimeError("memory bank is empty — build the gallery first")
        k = min(settings.knn, self.size)
        sims, _ = self.index.search(embeddings, k)  # (N, k) cosine sims
        return 1.0 - sims.mean(axis=1)

    # --- calibration -------------------------------------------------------
    def calibrate(self, normal_image_scores: list[float]) -> dict:
        arr = np.asarray(normal_image_scores, dtype="float64")
        self._calib = {
            "mean": float(arr.mean()),
            "std": float(arr.std() + 1e-8),
            "max": float(arr.max()),
            "n": int(arr.size),
        }
        return self._calib

    @property
    def threshold(self) -> float:
        if not self._calib:
            return 0.35  # sensible cosine default before calibration
        return self._calib["mean"] + settings.threshold_sigma * self._calib["std"]

    # --- persistence -------------------------------------------------------
    def save(self, path: Path | None = None) -> None:
        path = path or settings.artifact_dir
        faiss.write_index(self.index, str(path / "bank.faiss"))
        (path / "bank.json").write_text(
            json.dumps({"dim": self.dim, "size": self.size, "calib": self._calib})
        )

    @classmethod
    def load(cls, path: Path | None = None) -> "MemoryBank | None":
        path = path or settings.artifact_dir
        idx_p, meta_p = path / "bank.faiss", path / "bank.json"
        if not idx_p.exists() or not meta_p.exists():
            return None
        meta = json.loads(meta_p.read_text())
        bank = cls(meta["dim"])
        bank.index = faiss.read_index(str(idx_p))
        bank.size = meta["size"]
        bank._calib = meta.get("calib")
        return bank
