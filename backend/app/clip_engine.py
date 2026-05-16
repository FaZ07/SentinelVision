"""CLIP feature extractor.

We use OpenCLIP (LAION weights) purely as a *frozen* feature extractor.
No fine-tuning, no labels, no API — this is what makes the whole system
zero-shot: industrial defects are detected by how *far* an image patch sits
from the manifold of known-good patches in CLIP's embedding space.
"""
from __future__ import annotations

import threading
from functools import lru_cache

import numpy as np
import torch
from PIL import Image

from .config import settings


class ClipEngine:
    """Thread-safe singleton wrapper around an OpenCLIP visual encoder."""

    _lock = threading.Lock()

    def __init__(self) -> None:
        import open_clip

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        settings.device = self.device

        model, _, preprocess = open_clip.create_model_and_transforms(
            settings.clip_model,
            pretrained=settings.clip_pretrained,
            device=self.device,
        )
        model.eval()
        self.model = model
        self.preprocess = preprocess
        self.embed_dim: int = model.visual.output_dim

    @torch.inference_mode()
    def encode(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        """Embed a list of PIL images → L2-normalised float32 matrix (N, D).

        L2 normalisation means inner-product == cosine similarity, so the
        FAISS memory bank can use a fast IndexFlatIP.
        """
        feats: list[torch.Tensor] = []
        for start in range(0, len(images), batch_size):
            chunk = images[start : start + batch_size]
            batch = torch.stack([self.preprocess(im) for im in chunk]).to(self.device)
            emb = self.model.encode_image(batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            feats.append(emb.float().cpu())
        return torch.cat(feats).numpy().astype("float32")

    @torch.inference_mode()
    def encode_one(self, image: Image.Image) -> np.ndarray:
        return self.encode([image])[0]


@lru_cache(maxsize=1)
def get_engine() -> ClipEngine:
    """Lazily construct the engine once (model load is ~2s on first call)."""
    with ClipEngine._lock:
        return ClipEngine()
