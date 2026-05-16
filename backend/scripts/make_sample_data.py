"""Generate a synthetic industrial-texture dataset so the demo works offline
with zero downloads.

Creates brushed-metal-like textures:
  data/gallery/         32 known-good (clean) tiles
  data/test/good/       12 clean (held-out)
  data/test/defect/     12 with injected scratches / pits / blobs

For a real benchmark, swap this for the MVTec AD dataset (free for research):
  https://www.mvtec.com/company/research/datasets/mvtec-ad
Just drop class images into the same folder layout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(7)
SIZE = 512


def brushed_metal(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = rng.normal(150, 6, (SIZE, SIZE)).astype("float32")
    # Horizontal brushing: smooth along x, sharp along y.
    kernel = np.ones(31, dtype="float32") / 31
    for r in range(SIZE):
        base[r] = np.convolve(base[r], kernel, mode="same")
    base += rng.normal(0, 3, (SIZE, SIZE))
    grad = np.linspace(-12, 12, SIZE, dtype="float32")
    base += grad[None, :]
    arr = np.clip(base, 0, 255).astype("uint8")
    return Image.fromarray(np.stack([arr] * 3, axis=-1), "RGB")


def inject_defect(img: Image.Image, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = img.copy()
    d = ImageDraw.Draw(img)
    kind = rng.integers(0, 3)
    if kind == 0:  # scratch
        x0, y0 = rng.integers(40, SIZE - 40, 2)
        x1, y1 = x0 + rng.integers(-120, 120), y0 + rng.integers(-120, 120)
        d.line([(x0, y0), (x1, y1)], fill=(40, 40, 40), width=int(rng.integers(2, 5)))
    elif kind == 1:  # pit cluster
        cx, cy = rng.integers(60, SIZE - 60, 2)
        for _ in range(rng.integers(6, 14)):
            ox, oy = rng.integers(-25, 25, 2)
            rad = int(rng.integers(3, 8))
            d.ellipse([cx + ox - rad, cy + oy - rad, cx + ox + rad, cy + oy + rad],
                      fill=(30, 30, 30))
    else:  # contamination blob
        cx, cy = rng.integers(60, SIZE - 60, 2)
        rad = int(rng.integers(18, 38))
        d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(200, 180, 120))
    return img


def dump(folder: Path, n: int, start: int, defect: bool) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = brushed_metal(start + i)
        if defect:
            img = inject_defect(img, start + i + 999)
        img.save(folder / f"{'defect' if defect else 'good'}_{i:03d}.png")


def main() -> None:
    data = ROOT / "data"
    dump(data / "gallery", 32, 0, defect=False)
    dump(data / "test" / "good", 12, 1000, defect=False)
    dump(data / "test" / "defect", 12, 2000, defect=True)
    print(f"✅ Synthetic dataset written under {data}")
    print("   gallery/ = 32 good · test/good = 12 · test/defect = 12")


if __name__ == "__main__":
    sys.exit(main())
