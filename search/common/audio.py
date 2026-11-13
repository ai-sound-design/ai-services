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


# ── The adaptive index (embed_segments.py) ───────────────────────────────────
# Short windows find short events; merging the neighbours that sound alike keeps
# the index about as small as the 10 s windows. Measured on 343 BBC sounds: a 1 s
# excerpt found its recording first 77 % of the time (10 s windows: 57 %) with
# 18 % more rows; merging at 0.80 lost nothing against keeping every 2 s window.

FINE_SECONDS = 2.0               # window length of the adaptive index
FINE_HOP_SECONDS = 1.0           # a window starts every second, so they overlap by half
SEGMENT_THRESHOLD = float(os.getenv("SEGMENT_THRESHOLD", "0.80"))   # cosine to the segment's mean
SEGMENT_MAX_SECONDS = 30.0       # a long room tone still becomes several segments
QUIET_DB = 35.0                  # a window this far below the file's loud level is a pause


def fine_offsets(duration: float) -> list[float]:
    """Start times of the 2 s windows of a file; the tail gets a window of its own."""
    if duration <= FINE_SECONDS:
        return [0.0]
    offsets = [i * FINE_HOP_SECONDS for i in range(int((duration - FINE_SECONDS) // FINE_HOP_SECONDS) + 1)]
    last = round(duration - FINE_SECONDS, 3)
    if last - offsets[-1] > 0.25:
        offsets.append(last)
    return offsets


def rms_db(samples: np.ndarray) -> float:
    """Level in dBFS (-120 for silence)."""
    if len(samples) == 0:
        return -120.0
    return float(10.0 * np.log10(np.mean(np.asarray(samples, dtype=np.float64) ** 2) + 1e-12))


def segments(offsets: list[float], vectors: np.ndarray, levels: list[float], duration: float,
             threshold: float = SEGMENT_THRESHOLD, max_seconds: float = SEGMENT_MAX_SECONDS,
             quiet_db: float = QUIET_DB) -> list[tuple[float, float, np.ndarray]]:
    """[(start, end, unit vector)]: consecutive windows merged while each new one stays
    within `threshold` of the running mean; pauses are left out and end a segment."""
    if not offsets:
        return []
    loud = float(np.percentile(levels, 95))
    out: list[tuple[float, float, np.ndarray]] = []
    members: list[int] = []
    total = None

    def close() -> None:
        if members:
            mean = vectors[members].mean(axis=0)
            end = min(offsets[members[-1]] + FINE_SECONDS, duration)
            out.append((round(offsets[members[0]], 3), round(end, 3), mean / (np.linalg.norm(mean) or 1.0)))

    for i, vector in enumerate(vectors):
        if levels[i] < loud - quiet_db:
            close()
            members, total = [], None
            continue
        too_long = members and offsets[i] + FINE_SECONDS - offsets[members[0]] > max_seconds
        if total is None or too_long or float(vector @ (total / (np.linalg.norm(total) or 1.0))) < threshold:
            close()
            members, total = [i], vector.astype(np.float64).copy()
        else:
            members.append(i)
            total += vector
    close()
    return out


class ClapEncoder:
    """Audio and text into CLAP's joint space, unit length, as float32 arrays."""

    def __init__(self, model_name: str = CLAP_MODEL, device: str | None = None):
        import torch
        from transformers import ClapModel, ClapProcessor

        from common.device import torch_device

        self.device = device or torch_device()
        log.info("loading %s on %s", model_name, self.device)
        self.model_name = model_name
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name).to(self.device).eval()
        self.embedding_dim = int(self.model.config.projection_dim)
        self.gpu_features = os.getenv("CLAP_CPU_FEATURES", "") != "1"
        self._mel = self._mel_setup()

    def _mel_setup(self) -> dict | None:
        """What the GPU mel spectrogram needs, read off the extractor so both agree: the
        extractor's numpy version costs three times the model (3 s per 64 windows on the
        CPU against 1 s on the GPU), so the indexer was CPU-bound with the GPU idle."""
        import torch

        extractor = self.processor.feature_extractor
        try:
            if extractor.truncation != "fusion" or extractor.padding != "repeatpad":
                return None
            n_fft, hop, longest = int(extractor.fft_window_size), int(extractor.hop_length), int(extractor.nb_max_samples)
            filters = torch.tensor(np.asarray(extractor.mel_filters, dtype=np.float32).T, device=self.device)
        except (AttributeError, TypeError):
            return None
        return {"n_fft": n_fft, "hop": hop, "longest": longest, "filters": filters,          # filters: (mels, bins)
                "window": torch.hann_window(n_fft, periodic=True, device=self.device)}

    def _features(self, clips: list[np.ndarray]) -> dict:
        """The model's input for clips of up to 10 s: log-mel spectrograms (dB) of each clip
        repeated to 10 s, four identical fusion channels, computed on the GPU exactly as
        the extractor does on the CPU. Longer clips, or an unknown extractor, go the
        extractor's way."""
        import torch

        mel = self._mel if self.gpu_features else None
        if mel is None or any(len(c) > mel["longest"] for c in clips):
            # The feature extractor directly: the processor's audio keyword was renamed between versions.
            inputs = self.processor.feature_extractor(clips, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            if "is_longer" in inputs:
                # The fused model's extractor marks one random clip of a batch as "longer" when
                # none is (and a single clip always): that clip takes the fusion path and lands
                # elsewhere in the space, so a window embedded alone did not match itself
                # embedded in a batch (cosine ~0.7). The flag follows the real length instead.
                longer = [len(c) > WINDOW_SECONDS * SAMPLE_RATE for c in clips]
                inputs["is_longer"] = torch.tensor(longer, dtype=inputs["is_longer"].dtype).reshape(inputs["is_longer"].shape)
            return {k: v.to(self.device) for k, v in inputs.items()}
        longest = mel["longest"]
        padded = np.zeros((len(clips), longest), dtype=np.float32)
        for i, clip in enumerate(clips):
            tiled = np.tile(clip, longest // len(clip))        # "repeatpad": repeated as often as it fits, zeros after
            padded[i, :len(tiled)] = tiled
        half = mel["n_fft"] // 2
        x = torch.from_numpy(padded).to(self.device)
        x = torch.nn.functional.pad(x[:, None, :], (half, half), mode="reflect")[:, 0, :]
        spectrum = torch.stft(x, n_fft=mel["n_fft"], hop_length=mel["hop"], win_length=mel["n_fft"],
                              window=mel["window"], center=False, return_complex=True)
        power = spectrum.real ** 2 + spectrum.imag ** 2                       # (clips, bins, frames)
        log_mel = 10.0 * torch.log10(torch.clamp(mel["filters"] @ power, min=1e-10))
        features = log_mel.transpose(1, 2)[:, None, :, :].expand(-1, 4, -1, -1).contiguous()
        return {"input_features": features,
                "is_longer": torch.zeros((len(clips), 1), dtype=torch.bool, device=self.device)}

    def encode_audio(self, clips: list[np.ndarray]) -> np.ndarray:
        """One embedding per clip (mono float32 at SAMPLE_RATE, any length up to 10 s)."""
        import torch

        clips = [np.asarray(c, dtype=np.float32) if len(c) >= SAMPLE_RATE // 10
                 else np.zeros(SAMPLE_RATE // 10, dtype=np.float32) for c in clips]
        inputs = self._features(clips)
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
