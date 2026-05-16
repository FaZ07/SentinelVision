"""Core anomaly-logic proof WITHOUT the heavy CLIP backbone.

We stub `clip_engine` with a deterministic fake encoder so we can verify the
parts that actually contain the IP: tiling geometry, the FAISS memory bank,
calibration/thresholding, per-tile scoring, heat-map generation and the
AUROC metric. CLIP itself is just a library call and is exercised at runtime.

Run from backend/ with the venv python.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── stub clip_engine before anything imports it ───────────────────────────
fake = types.ModuleType("app.clip_engine")


class FakeEngine:
    embed_dim = 64

    def encode(self, images):
        """Map mean brightness → a smooth point on the unit sphere.
        Clean tiles cluster together; a dark scratch lands far away."""
        out = []
        for im in images:
            b = float(np.asarray(im.convert("L")).mean()) / 255.0
            v = np.zeros(self.embed_dim, dtype="float32")
            v[0] = np.cos(b * np.pi)
            v[1] = np.sin(b * np.pi)
            v[2] = b
            out.append(v / (np.linalg.norm(v) + 1e-8))
        return np.stack(out).astype("float32")

    def encode_one(self, image):
        return self.encode([image])[0]


fake.get_engine = lambda: FakeEngine()
sys.modules["app.clip_engine"] = fake

from app.detector import _auroc  # noqa: E402
from app.heatmap import build_overlay  # noqa: E402
from app.memory_bank import MemoryBank  # noqa: E402
from app.tiling import tile_image  # noqa: E402


def clean(seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(150, 4, (256, 256)).astype("uint8")
    return Image.fromarray(np.stack([a] * 3, -1), "RGB")


def defective(seed):
    img = clean(seed).copy()
    a = np.asarray(img).copy()
    a[110:140, 60:190] = 20  # dark gash
    return Image.fromarray(a, "RGB")


def main() -> None:
    eng = FakeEngine()

    # 1 · tiling geometry
    _, tiles = tile_image(clean(0))
    assert tiles[0].row == -1, "tile 0 must be the global view"
    assert len(tiles) == 1 + 16, f"expected 17 tiles, got {len(tiles)}"
    print(f"[OK] tiling: 1 global + 16 grid tiles, boxes within canvas")

    # 2 · memory bank from 'normal' images + calibration
    bank = MemoryBank(eng.embed_dim)
    cal_scores = []
    for s in range(12):
        _, t = tile_image(clean(s))
        emb = eng.encode([x.image for x in t])
        bank.add(emb)
    for s in range(12):
        _, t = tile_image(clean(100 + s))
        emb = eng.encode([x.image for x in t])
        cal_scores.append(float(bank.anomaly_scores(emb).max()))
    bank.calibrate(cal_scores)
    print(f"[OK] memory bank: {bank.size} tiles, threshold={bank.threshold:.4f}")

    # 3 · scoring separates good vs defective
    good_s, bad_s, labels, scores = [], [], [], []
    for s in range(20):
        _, tg = tile_image(clean(500 + s))
        _, tb = tile_image(defective(700 + s))
        sg = float(bank.anomaly_scores(eng.encode([x.image for x in tg])).max())
        sb = float(bank.anomaly_scores(eng.encode([x.image for x in tb])).max())
        good_s.append(sg); bad_s.append(sb)
        scores += [sg, sb]; labels += [0, 1]
    assert np.mean(bad_s) > np.mean(good_s), "defective must score higher"
    auroc = _auroc(np.array(labels), np.array(scores))
    print(f"[OK] separation: good~{np.mean(good_s):.3f}  "
          f"defect~{np.mean(bad_s):.3f}  AUROC={auroc:.3f}")
    assert auroc > 0.9, f"AUROC too low: {auroc}"

    # 4 · heat-map renders to a base64 PNG
    canvas, t = tile_image(defective(9))
    emb = eng.encode([x.image for x in t])
    overlay = build_overlay(canvas, t, bank.anomaly_scores(emb), bank.threshold)
    assert overlay.startswith("data:image/png;base64,") and len(overlay) > 5000
    print(f"[OK] heat-map: {len(overlay) // 1024} KB base64 PNG produced")

    # 5 · persistence round-trip
    bank.save()
    reloaded = MemoryBank.load()
    assert reloaded and reloaded.size == bank.size
    print(f"[OK] persistence: bank reloaded ({reloaded.size} tiles)")

    print("\n[PASS] SentinelVision core anomaly pipeline verified "
          "(FAISS bank + tiling + scoring + heat-map + AUROC).")


if __name__ == "__main__":
    main()
