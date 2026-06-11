# SentinelVision

![CI](https://github.com/FaZ07/SentinelVision/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

### Zero-Shot Industrial Anomaly Detection — no labels, no training, no API, fully offline

SentinelVision finds manufacturing defects (scratches, pits, contamination, missing
parts) from a camera image **without ever being shown a defect**. You teach it what
*good* looks like; anything that strays from that manifold is flagged — and the
system shows you **exactly where** on the part the anomaly is.

> Built on the **PatchCore** paradigm but with a **frozen CLIP backbone** instead of
> a supervised CNN — so it needs zero training, zero defect labels, and runs entirely
> on-device. The same encoder exports to **ONNX** for Jetson / Raspberry Pi edge units.

---

## Why this is hard (and why it matters)

| Naïve approach | Problem | SentinelVision |
|---|---|---|
| Train a classifier on defects | Defects are rare, infinite in variety, expensive to label | Models only *normality* — needs ~10 good samples |
| Autoencoder reconstruction error | Blurry, unstable, retrain per product line | Frozen CLIP features generalise across products zero-shot |
| Cloud vision API | Data leaves the factory; network dependency on the line | 100% offline, air-gapped, no API key |

This is the exact framing used in real factory QA research — *normality modelling +
patch-level memory bank + nearest-neighbour distance.*

---

## Architecture

```
                 ┌──────────────────────────────────────────────┐
  known-good ───▶ │ tiling (4×4 overlap grid + global view)      │
   images         └───────────────────┬──────────────────────────┘
                                       ▼
                          ┌────────────────────────┐
                          │ OpenCLIP ViT-B/32      │  frozen, no fine-tune
                          │ (LAION weights)        │
                          └───────────┬────────────┘
                                      ▼  L2-normalised embeddings
                          ┌────────────────────────┐
                          │ FAISS IndexFlatIP      │  ← "memory bank" of
                          │ (cosine memory bank)   │     normal tiles
                          └───────────┬────────────┘
                                      ▼
   test image ─tiling▶ embed ─▶ kNN distance ─▶ score = 0.35·global + 0.65·max-region
                                      │
                                      ├──▶ threshold = μ + 3σ  (auto-calibrated)
                                      ├──▶ PASS / DEFECT verdict + confidence
                                      └──▶ heat-map overlay (where the defect is)
```

**Decision score**: `1 − mean cosine similarity to k nearest known-good tiles`.
**Threshold**: auto-calibrated as `mean + 3·std` of normal-image scores (UI-tunable).
**Localisation**: every tile keeps its canvas coordinates → scores are splatted back
into a Gaussian-blended JET heat-map.

---

## Quick start

### 1 · Backend (Python 3.10+)

```bash
cd backend
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt

# generate a synthetic brushed-metal dataset (works offline, no downloads)
python scripts/make_sample_data.py

uvicorn app.main:app --reload --port 8000
```

First request downloads CLIP weights (~350 MB) **once** from HuggingFace, then it
runs forever with WiFi off.

### 2 · Frontend

```bash
cd frontend
npm install
npm run dev          # http://localhost:5180
```

### 3 · Use it

1. **Build memory bank** → upload `backend/data/gallery/*.png` (the 32 good samples)
2. **Inspect** → upload anything from `backend/data/test/defect/` — watch the heat-map
3. **Benchmark** → scores the held-out `test/{good,defect}` set → AUROC + confusion
4. **Export ONNX** → verified parity vs PyTorch for edge deployment

> **Real data:** swap the synthetic set for [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
> (free for research). Drop class images into the same `gallery/` and `test/` layout — nothing else changes.

---

## API

| Method | Route | Purpose |
|---|---|---|
| `GET`  | `/api/health` | model, device, gallery state |
| `POST` | `/api/gallery/build` | multipart known-good images → build + calibrate |
| `GET`  | `/api/gallery/status` | bank size + calibration stats |
| `POST` | `/api/inspect` | one image → score, verdict, heat-map, tile grid |
| `GET`  | `/api/benchmark` | AUROC / precision / recall / F1 / confusion |
| `POST` | `/api/onnx/export` | export CLIP visual encoder + parity check |

Interactive docs at `http://localhost:8000/docs`.

---

## Tech stack

**ML**: OpenCLIP (ViT-B/32, LAION) · FAISS · PyTorch · ONNX Runtime
**Backend**: FastAPI · Pydantic v2 · Uvicorn
**Frontend**: React 18 · Vite · Tailwind CSS
**Zero external services. Zero API keys. Runs air-gapped.**

---

## Engineering notes

- **Coreset subsampling** (`config.coreset_ratio`) keeps the memory bank fast for
  large galleries — the PatchCore trick.
- **Leave-in calibration**: each normal image is scored against the bank to derive a
  data-driven threshold instead of a hand-picked constant.
- **AUROC** is computed via the Mann–Whitney U rank statistic — no scikit-learn
  dependency, threshold-independent quality metric.
- **ONNX parity** is asserted to `< 1e-3` max abs diff so the edge model is provably
  the same as the served model.

---

## Screenshots

| Gallery Builder | Anomaly Inspector | Benchmark |
|---|---|---|
| Upload good samples → bank builds in seconds | Heat-map pinpoints the defect on the part | AUROC + confusion matrix on held-out set |

> Run the app locally to see the live UI — `make_sample_data.py` generates a ready-to-use dataset.

---

## License

MIT

---

*Built by Mohamed Fazil — AI/ML & Full-Stack Engineer.*

---

## 📊 Measured results — MVTec AD (industry-standard benchmark)

Real numbers from `backend/scripts/mvtec_bench.py`, reproducible end-to-end.
Setup: **20 known-good samples per category** (no defect ever shown, no training,
frozen CLIP ViT-B/32, CPU-only), scored on the full official test sets:

| Category | Test set | Image AUROC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| bottle    | 20 good / 63 defect | **0.994** | 0.969 | 0.984 | 0.976 |
| hazelnut  | 40 good / 70 defect | 0.904 | 0.765 | 0.929 | 0.839 |
| metal_nut | 22 good / 93 defect | 0.881 | 0.972 | 0.742 | 0.842 |
| **mean**  | 235 images | **0.926** | | | |

Reproduce: download any [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
category, then `python scripts/mvtec_bench.py <mvtec_root> bottle 20`.
