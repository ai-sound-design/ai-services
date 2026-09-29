"""
Audio embeddings with CLAP, shared by the indexer (embed_audio.py) and the
search API (/search/by_audio).

CLAP maps audio and text into one space, so a generated sound, a library
recording and a sentence can all be compared with cosine similarity. The
model listens to 10 s at a time (shorter input is repeat-padded), which is why
a library file is indexed as overlapping 10 s windows rather than as a whole:
a two-second footstep inside a three-minute kitchen ambience would otherwise
vanish in the average.

Environment:
    CLAP_MODEL      Hugging Face id, default laion/clap-htsat-fused (512 dims)
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import numpy as np

CLAP_MODEL = os.getenv("CLAP_MODEL", "laion/clap-htsat-fused")
SAMPLE_RATE = 48000              # what CLAP expects
WINDOW_SECONDS = 10.0            # CLAP's native input length
HOP_SECONDS = 5.0                # overlap, so a short event is never split between two windows
MAX_WINDOWS_PER_FILE = 120       # beyond 10 minutes a file is sampled more coarsely
EMBEDDING_DIM = 512

log = logging.getLogger("audio")


def decode(path: Path | str, start: float | None = None, length: float | None = None) -> np.ndarray:
    """Mono float32 samples at SAMPLE_RATE, via ffmpeg (any format, any rate)."""
    command = ["ffmpeg", "-v", "error", "-nostdin"]
    if start:
        command += ["-ss", f"{start:.3f}"]
    command += ["-i", str(path)]
    if length:
        command += ["-t", f"{length:.3f}"]
    command += ["-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"]
    done = subprocess.run(command, check=True, capture_output=True)
    return np.frombuffer(done.stdout, dtype=np.float32)


def window_offsets(duration: float, length: float = WINDOW_SECONDS, hop: float = HOP_SECONDS,
                   cap: int = MAX_WINDOWS_PER_FILE) -> list[float]:
    """Start times of the windows that cover `duration` seconds."""
    if duration <= length:
        return [0.0]
    count = int((duration - length) // hop) + 1
    offsets = [i * hop for i in range(count)]
    last = duration - length
    if last - offsets[-1] > hop / 2:           # a tail worth its own window
        offsets.append(round(last, 3))
    if len(offsets) > cap:                     # long file: spread `cap` windows evenly
        offsets = [offsets[round(i * (len(offsets) - 1) / (cap - 1))] for i in range(cap)]
    return offsets


def slice_window(samples: np.ndarray, offset: float, length: float = WINDOW_SECONDS) -> np.ndarray:
    start = int(offset * SAMPLE_RATE)
    return samples[start:start + int(length * SAMPLE_RATE)]


class ClapEncoder:
    """Audio and text into CLAP's joint space, unit length, as float32 arrays."""

    def __init__(self, model_name: str = CLAP_MODEL, device: str | None = None):
        import torch
        from transformers import ClapModel, ClapProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("loading %s on %s", model_name, self.device)
        self.model_name = model_name
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name).to(self.device).eval()
        self.embedding_dim = int(self.model.config.projection_dim)

    def encode_audio(self, clips: list[np.ndarray]) -> np.ndarray:
        """One embedding per clip (mono float32 at SAMPLE_RATE, any length up to 10 s)."""
        import torch

        clips = [np.asarray(c, dtype=np.float32) if len(c) >= SAMPLE_RATE // 10
                 else np.zeros(SAMPLE_RATE // 10, dtype=np.float32) for c in clips]
        # The feature extractor directly: the processor's audio keyword was renamed between versions.
        inputs = self.processor.feature_extractor(clips, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self.model.get_audio_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.float().cpu().numpy()

    def encode_text(self, texts: list[str]) -> np.ndarray:
        import torch

        inputs = self.processor.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.float().cpu().numpy()

    def encode_long_audio(self, samples: np.ndarray) -> np.ndarray:
        """One embedding for audio of any length: the mean of its 10 s windows."""
        duration = len(samples) / SAMPLE_RATE
        pieces = [slice_window(samples, o) for o in window_offsets(duration, cap=24)]
        vectors = self.encode_audio(pieces)
        mean = vectors.mean(axis=0)
        return mean / (np.linalg.norm(mean) or 1.0)
