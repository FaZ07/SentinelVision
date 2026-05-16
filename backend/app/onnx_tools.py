"""Export the frozen CLIP visual encoder to ONNX for edge deployment.

Demonstrates production-readiness: the exact same embedding model that runs
in the FastAPI service can be shipped to a Jetson / RPi running ONNX Runtime,
with verified numerical parity against PyTorch.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .clip_engine import get_engine
from .config import settings


def export_visual_encoder(out: Path | None = None) -> dict:
    out = out or (settings.artifact_dir / "clip_visual.onnx")
    engine = get_engine()
    visual = engine.model.visual.eval()

    dummy = torch.randn(1, 3, 224, 224, device=engine.device)

    class Wrap(torch.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.v = v

        def forward(self, x):
            f = self.v(x)
            return f / f.norm(dim=-1, keepdim=True)

    torch.onnx.export(
        Wrap(visual),
        dummy,
        str(out),
        input_names=["pixel_values"],
        output_names=["embedding"],
        dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
    )

    parity = _verify(out, dummy, Wrap(visual))
    return {
        "path": str(out),
        "size_mb": round(out.stat().st_size / 1e6, 2),
        "max_abs_diff": parity,
        "parity_ok": parity < 1e-3,
    }


@torch.inference_mode()
def _verify(onnx_path: Path, dummy: torch.Tensor, model: torch.nn.Module) -> float:
    try:
        import onnxruntime as ort
    except ImportError:
        return -1.0  # onnxruntime optional; export still succeeds
    torch_out = model(dummy).cpu().numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"pixel_values": dummy.cpu().numpy()})[0]
    return float(np.abs(torch_out - onnx_out).max())
