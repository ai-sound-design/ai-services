"""Which torch device the search models run on.

CUDA where there is an NVIDIA GPU (the Docker stack), the Apple GPU (MPS) on a Mac
running the services natively, the CPU otherwise. FORCE_DEVICE (cuda, mps, cpu)
overrides the choice, as it does for the generation service.
"""
from __future__ import annotations

import os


def torch_device() -> str:
    import torch

    forced = os.getenv("FORCE_DEVICE", "auto").strip().lower()
    if forced in ("cuda", "mps", "cpu"):
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
