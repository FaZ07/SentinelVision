"""SentinelVision FastAPI service.

Endpoints
---------
GET  /api/health               liveness + model/device info
POST /api/gallery/build        upload known-good images → build memory bank
GET  /api/gallery/status       current bank size + calibration
POST /api/inspect              upload one image → score + verdict + heat-map
GET  /api/benchmark            evaluate on data/test/{good,defect}
POST /api/onnx/export          export CLIP visual encoder to ONNX
"""
from __future__ import annotations

import io
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from . import __version__
from .config import settings
from .detector import get_detector
from .schemas import Benchmark, GalleryStatus, Health, Inspection, OnnxExport

app = FastAPI(title="SentinelVision", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@app.get("/api/health", response_model=Health)
def health() -> Health:
    det = get_detector()
    ready = det.bank is not None and det.bank.size > 0
    return Health(
        status="ok",
        model=f"{settings.clip_model} / {settings.clip_pretrained}",
        device=settings.device,
        gallery_ready=ready,
        gallery_tiles=det.bank.size if ready else 0,
    )


@app.post("/api/gallery/build", response_model=GalleryStatus)
async def gallery_build(files: list[UploadFile] = File(...)) -> GalleryStatus:
    if len(files) < 3:
        raise HTTPException(400, "Provide at least 3 known-good images for calibration.")
    tmp = Path(tempfile.mkdtemp())
    try:
        paths: list[Path] = []
        for f in files:
            ext = Path(f.filename or "x.png").suffix.lower()
            if ext not in _IMG_EXT:
                continue
            dst = tmp / (f.filename or "img.png")
            dst.write_bytes(await f.read())
            paths.append(dst)
        if not paths:
            raise HTTPException(400, "No valid images uploaded.")
        stats = get_detector().build_gallery(paths)
        return GalleryStatus(**stats)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.get("/api/gallery/status", response_model=GalleryStatus)
def gallery_status() -> GalleryStatus:
    det = get_detector()
    if det.bank is None or det.bank.size == 0:
        raise HTTPException(404, "No gallery built yet.")
    return GalleryStatus(
        images=det.bank._calib["n"] if det.bank._calib else 0,
        tiles=det.bank.size,
        threshold=det.bank.threshold,
        calibration=det.bank._calib,
    )


@app.post("/api/inspect", response_model=Inspection)
async def inspect(file: UploadFile = File(...)) -> Inspection:
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw))
    except Exception:
        raise HTTPException(400, "Unreadable image file.")
    try:
        res = get_detector().inspect(img)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return Inspection(**res.__dict__)


@app.get("/api/benchmark", response_model=Benchmark)
def benchmark() -> Benchmark:
    det = get_detector()
    if det.bank is None or det.bank.size == 0:
        raise HTTPException(409, "Build a gallery before benchmarking.")
    out = det.benchmark(settings.data_dir / "test" / "good", settings.data_dir / "test" / "defect")
    if "error" in out:
        raise HTTPException(404, out["error"])
    return Benchmark(**out)


@app.post("/api/onnx/export", response_model=OnnxExport)
def onnx_export() -> OnnxExport:
    from .onnx_tools import export_visual_encoder

    return OnnxExport(**export_visual_encoder())
