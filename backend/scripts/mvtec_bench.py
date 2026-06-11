"""MVTec AD benchmark for SentinelVision — real numbers, no synthetic data.

Usage:  python scripts/mvtec_bench.py <mvtec_root> <category> [gallery_n]

Builds the memory bank from train/good (first N images), then scores the
full official test set (good vs all defect types pooled) and prints
AUROC / precision / recall / F1 from the detector's own benchmark path.
"""
from __future__ import annotations

import sys, time, tempfile, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.detector import Detector  # noqa: E402


def main() -> None:
    mvtec_root, category = Path(sys.argv[1]), sys.argv[2]
    gallery_n = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    cat = mvtec_root / category
    train_good = sorted((cat / "train" / "good").glob("*.png"))[:gallery_n]
    test_good_dir = cat / "test" / "good"
    defect_dirs = [d for d in (cat / "test").iterdir() if d.is_dir() and d.name != "good"]
    n_defects = sum(len(list(d.glob("*.png"))) for d in defect_dirs)
    print(f"[mvtec_bench] {category}: gallery={len(train_good)} good train imgs · "
          f"test={len(list(test_good_dir.glob('*.png')))} good / {n_defects} defect "
          f"({', '.join(d.name for d in defect_dirs)})")

    # pool all defect types into one dir (detector.benchmark takes a single dir)
    pooled = Path(tempfile.mkdtemp(prefix="mvtec_defect_"))
    for d in defect_dirs:
        for p in d.glob("*.png"):
            shutil.copy(p, pooled / f"{d.name}_{p.name}")

    det = Detector()
    t0 = time.time()
    det.build_gallery(train_good)
    print(f"[mvtec_bench] gallery built in {time.time()-t0:.1f}s")

    t0 = time.time()
    result = det.benchmark(test_good_dir, pooled)
    print(f"[mvtec_bench] benchmark in {time.time()-t0:.1f}s")
    print("=" * 60)
    for k, v in result.items():
        print(f"  {k}: {v}")
    shutil.rmtree(pooled, ignore_errors=True)


if __name__ == "__main__":
    main()
