"""Generation gateway: the `/generate` contract without a length limit.

The model services (mmaudio-api, or whatever replaces it) stay thin: one call, one
window of audio, and a `capabilities` block in their `/health` saying what they can
do. Everything that does not depend on the model lives here:

- a request longer than the model's window is generated in overlapping windows, each
  from its own stretch of the video, with the same prompt and seed, and joined with an
  equal-power crossfade (or, for a model that takes the previous audio as context,
  continued window by window and butt-joined);
- a request shorter than the model's minimum gets the video extended (last frame
  held) or, text-only, the minimum length, and the audio trimmed back afterwards;
- progress per window under `/generate/progress/{job_id}`.

Environment:
    GENERATION_MODEL_URL     http://mmaudio-api:8000   the model service
    GATEWAY_MAX_SECONDS      600                        the longest sound the gateway makes
    GATEWAY_OVERLAP_SECONDS  1.0                        overlap between crossfaded windows
    GATEWAY_CONTEXT_SECONDS  2.0                        audio handed to a continuing model
    MODEL_TIMEOUT_S          1800                       one window's generation at most
    API_PORT                 8010

Contract (the same as the model's): POST multipart `video` (optional), `prompt`,
`negative_prompt`, `seed`, `duration` (seconds; for text-only required, with a video
optional and at most the video's length), `output_format` (wav|flac), `job_id`
(optional, for progress). Other fields are passed through to the model. The answer is
the audio file, with `X-Generation-Parts` (windows used), `X-Duration`, `X-Seed`.

A model's `/health` may carry:
    "capabilities": {"min_seconds": 4, "max_seconds": 12, "modes": ["v2a", "t2a"],
                     "continuation": false, "context_field": "context_audio",
                     "context_seconds": 2}
Without the block, 4-12 s, both modes, no continuation are assumed.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import httpx
import numpy as np
import soundfile as sf
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

MODEL_URL = os.environ.get("GENERATION_MODEL_URL", "http://mmaudio-api:8000").rstrip("/")
MAX_SECONDS = float(os.environ.get("GATEWAY_MAX_SECONDS", "600"))
OVERLAP = max(0.0, float(os.environ.get("GATEWAY_OVERLAP_SECONDS", "1.0")))
CONTEXT_SECONDS = max(0.0, float(os.environ.get("GATEWAY_CONTEXT_SECONDS", "2.0")))
MODEL_TIMEOUT_S = float(os.environ.get("MODEL_TIMEOUT_S", "1800"))
PORT = int(os.environ.get("API_PORT", "8010"))
WORK_ROOT = Path(os.environ.get("GATEWAY_WORK_DIR", tempfile.gettempdir())) / "generation-gateway"
CUT_MARGIN = 0.25           # seconds of extra video per window: ffmpeg cuts on whole frames

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "info").upper(),
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gateway")

app = FastAPI(title="Generation gateway", version="1.0.0")

DEFAULT_CAPS = {"min_seconds": 4.0, "max_seconds": 12.0, "modes": ["v2a", "t2a"], "continuation": False,
                "context_field": "context_audio", "context_seconds": CONTEXT_SECONDS}
_caps_cache: dict[str, Any] = {"at": 0.0, "caps": None, "health": None}


# ── the model's capabilities ────────────────────────────────────────────────

async def model_health(client: httpx.AsyncClient) -> Optional[dict]:
    try:
        answer = await client.get(f"{MODEL_URL}/health", timeout=8.0)
    except httpx.HTTPError:
        return None
    if answer.status_code != 200:
        return None
    try:
        return answer.json()
    except ValueError:
        return {}


async def capabilities(client: httpx.AsyncClient, force: bool = False) -> dict:
    """The model's capabilities, from its health, remembered for a minute."""
    if not force and _caps_cache["caps"] is not None and time.time() - _caps_cache["at"] < 60.0:
        return _caps_cache["caps"]
    health = await model_health(client)
    caps = dict(DEFAULT_CAPS)
    if health is not None:
        block = health.get("capabilities") if isinstance(health.get("capabilities"), dict) else {}
        for key in ("min_seconds", "max_seconds", "context_seconds"):
            if key in block:
                try:
                    caps[key] = float(block[key])
                except (TypeError, ValueError):
                    pass
        if isinstance(block.get("modes"), list):
            caps["modes"] = [str(m).lower() for m in block["modes"]]
        caps["continuation"] = bool(block.get("continuation", False))
        if block.get("context_field"):
            caps["context_field"] = str(block["context_field"])
        _caps_cache.update(at=time.time(), caps=caps, health=health)
    return caps


# ── progress ────────────────────────────────────────────────────────────────

PROGRESS: dict[str, dict] = {}


def set_progress(job_id: Optional[str], **fields) -> None:
    if not job_id:
        return
    now = time.time()
    for key in [k for k, v in PROGRESS.items() if now - v.get("updated", now) > 3600]:
        PROGRESS.pop(key, None)
    entry = PROGRESS.setdefault(job_id, {})
    entry.update(fields)
    entry["updated"] = now


@app.get("/generate/progress/{job_id}")
async def progress(job_id: str):
    """Where a `/generate` sent with this `job_id` is: `stage` (generating, done,
    failed), `part` and `parts` (windows), `fraction` (0..1) and `detail`."""
    entry = PROGRESS.get(job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return {k: v for k, v in entry.items() if k != "updated"}


# ── health ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    async with httpx.AsyncClient() as client:
        caps = await capabilities(client, force=True)
        model = _caps_cache["health"]
    if model is None:
        raise HTTPException(status_code=503, detail=f"model service {MODEL_URL} not reachable")
    return {"status": "ok", "service": "generation gateway", "model": MODEL_URL,
            "model_service": model.get("service"), "device": model.get("device"),
            "min_seconds": caps["min_seconds"], "max_seconds": MAX_SECONDS,
            "window_seconds": caps["max_seconds"], "overlap_seconds": OVERLAP,
            "continuation": caps["continuation"], "modes": caps["modes"],
            "capabilities": {"min_seconds": caps["min_seconds"], "max_seconds": MAX_SECONDS,
                             "modes": caps["modes"], "continuation": caps["continuation"]}}


# ── video helpers ───────────────────────────────────────────────────────────

def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def video_length(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         check=True, capture_output=True, text=True).stdout.strip()
    try:
        return max(0.0, float(out))
    except ValueError:
        raise HTTPException(status_code=400, detail="could not read the video's length")


def cut_video(src: Path, start: float, length: float, dst: Path) -> None:
    run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{length:.3f}",
         "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", str(dst)])


def extend_video(src: Path, length: float, dst: Path) -> None:
    """The video held on its last frame until it is `length` seconds long."""
    have = video_length(src)
    pad = max(0.0, length - have)
    run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-an",
         "-vf", f"tpad=stop_mode=clone:stop_duration={pad:.3f}", "-t", f"{length:.3f}",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", str(dst)])


# ── the window plan ─────────────────────────────────────────────────────────

def plan_windows(total: float, window: float, overlap: float, minimum: float) -> list[tuple[float, float]]:
    """(start, length) of the windows that cover `total` seconds. One window when the
    total fits (at least `minimum` long, trimmed afterwards); otherwise the fewest
    windows of equal length (at most `window`, at least `minimum`) that overlap by
    `overlap`, so 40 s with a 12 s model is four windows of 10.75 s, not three full
    ones and a stub."""
    if total <= window + 1e-6:
        return [(0.0, max(total, minimum))]
    n = max(2, int(math.ceil((total - overlap) / max(0.5, window - overlap) - 1e-9)))
    length = (total + (n - 1) * overlap) / n
    if length < minimum - 1e-6:
        # The total barely exceeds the window and the model's minimum is large: full
        # windows stepping along, the last one pulled back to end at `total`
        step = max(0.5, window - overlap)
        starts = [0.0]
        while starts[-1] + window < total - 1e-6:
            starts.append(starts[-1] + step)
        starts[-1] = max(0.0, total - window)
        cleaned: list[float] = []
        for s in starts:
            if not cleaned or s > cleaned[-1] + 1e-6:
                cleaned.append(s)
        return [(s, window) for s in cleaned]
    return [(round(k * (length - overlap), 3), round(length, 3)) for k in range(n)]


def stitch(parts: list[tuple[float, np.ndarray]], rate: int, total: float, continuation: bool) -> np.ndarray:
    """The windows laid at their starts; where they overlap, an equal-power crossfade
    (a short one for continued windows, which should already match)."""
    n_total = int(round(total * rate))
    channels = parts[0][1].shape[1]
    out = np.zeros((n_total, channels), dtype=np.float32)
    end_filled = 0
    for start, audio in parts:
        s0 = int(round(start * rate))
        n = min(len(audio), n_total - s0)
        if n <= 0:
            continue
        chunk = audio[:n].astype(np.float32)
        ov = max(0, min(end_filled - s0, n))
        if continuation:
            ov = min(ov, int(0.02 * rate))          # a continued window joins at its edge; 20 ms against clicks
            s_join = max(s0, end_filled - ov)
            ov = max(0, min(end_filled - s_join, n))
            s0_eff = s_join
        else:
            s0_eff = s0
        if ov > 0:
            ramp = np.linspace(0.0, math.pi / 2, ov, dtype=np.float32)[:, None]
            out[s0_eff:s0_eff + ov] = out[s0_eff:s0_eff + ov] * np.cos(ramp) + chunk[:ov] * np.sin(ramp)
        rest = n - ov
        if rest > 0:
            out[s0_eff + ov:s0_eff + ov + rest] = chunk[ov:ov + rest]
        end_filled = max(end_filled, s0_eff + n)
    return out


# ── the model call ──────────────────────────────────────────────────────────

async def call_model(client: httpx.AsyncClient, video: Optional[Path], fields: dict[str, str],
                     duration: float, context: Optional[Path], context_field: str, dst: Path) -> None:
    data = dict(fields)
    data["duration"] = f"{duration:.2f}"
    data["output_format"] = "wav"
    files: dict[str, tuple] = {}
    handles = []
    try:
        if video is not None:
            h = video.open("rb")
            handles.append(h)
            files["video"] = (video.name, h, "video/mp4")
        if context is not None:
            h = context.open("rb")
            handles.append(h)
            files[context_field] = (context.name, h, "audio/wav")
        response = await client.post(f"{MODEL_URL}/generate", data=data, files=files or None,
                                     timeout=MODEL_TIMEOUT_S)
    finally:
        for h in handles:
            h.close()
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"model service: {response.status_code} {response.text[:300]}")
    dst.write_bytes(response.content)


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    data, rate = sf.read(str(path), always_2d=True, dtype="float32")
    return data, int(rate)


# ── /generate ───────────────────────────────────────────────────────────────

PASS_THROUGH = {"model_name", "num_steps", "cfg_strength", "full_precision"}


@app.post("/generate")
async def generate(request: Request, background: BackgroundTasks,
                   video: Optional[UploadFile] = File(None),
                   prompt: str = Form(""), negative_prompt: str = Form(""), seed: int = Form(42),
                   duration: Optional[float] = Form(None), output_format: str = Form("wav"),
                   job_id: Optional[str] = Form(None)):
    form = await request.form()
    extra = {k: str(v) for k, v in form.items() if k in PASS_THROUGH}
    fields = {"prompt": prompt, "negative_prompt": negative_prompt, "seed": str(seed), **extra}
    output_format = output_format.lower() if output_format.lower() in ("wav", "flac") else "wav"

    work = WORK_ROOT / uuid.uuid4().hex
    work.mkdir(parents=True, exist_ok=True)
    background.add_task(shutil.rmtree, work, True)

    async with httpx.AsyncClient() as client:
        caps = await capabilities(client)
        minimum, window = float(caps["min_seconds"]), float(caps["max_seconds"])

        src: Optional[Path] = None
        if video is not None and video.filename:
            src = work / "input.mp4"
            with src.open("wb") as handle:
                shutil.copyfileobj(video.file, handle)
            have = await asyncio.to_thread(video_length, src)
            if have <= 0.0:
                raise HTTPException(status_code=400, detail="the video has no length")
            total = min(float(duration), have) if duration and duration > 0 else have
            if "v2a" not in caps["modes"]:
                raise HTTPException(status_code=400, detail="the model takes no video")
        else:
            if not duration or duration <= 0:
                raise HTTPException(status_code=400, detail="duration is required without a video")
            total = float(duration)
            if "t2a" not in caps["modes"]:
                raise HTTPException(status_code=400, detail="the model needs a video")
        if total > MAX_SECONDS + 1e-6:
            raise HTTPException(status_code=400,
                                detail=f"Duration too long: {total:.1f}s (this gateway makes at most {MAX_SECONDS:.0f}s)")

        continuation = bool(caps["continuation"])
        overlap = 0.0 if continuation else OVERLAP
        windows = plan_windows(total, window, overlap, minimum)
        parts_n = len(windows)
        log.info("generate: %.1fs in %d window(s) of %.0fs (overlap %.1fs%s), seed %d, prompt %r",
                 total, parts_n, window, overlap, ", continuation" if continuation else "", seed, prompt[:60])
        set_progress(job_id, stage="generating", part=0, parts=parts_n, fraction=0.0,
                     detail=f"{total:.0f} s in {parts_n} part(s)")

        parts: list[tuple[float, np.ndarray]] = []
        rate = 0
        previous: Optional[Path] = None
        started = time.time()
        for k, (start, length) in enumerate(windows, start=1):
            if await request.is_disconnected():
                set_progress(job_id, stage="cancelled", fraction=1.0, detail="caller disconnected")
                raise HTTPException(status_code=499, detail="caller disconnected")
            set_progress(job_id, stage="generating", part=k, parts=parts_n, fraction=(k - 1) / parts_n,
                         detail=f"part {k} of {parts_n}")
            piece: Optional[Path] = None
            if src is not None:
                piece = work / f"window_{k}.mp4"
                if start <= 1e-6 and have < length - 1e-6:        # shorter than the model's minimum
                    await asyncio.to_thread(extend_video, src, length, piece)
                else:
                    # A little longer than asked (ffmpeg cuts on whole frames, and 3.97 s would
                    # fall under a 4 s minimum), but never past the model's window
                    await asyncio.to_thread(cut_video, src, start, min(length + CUT_MARGIN, window, have - start), piece)
            context: Optional[Path] = None
            if continuation and previous is not None and parts:
                context = work / f"context_{k}.wav"
                tail = parts[-1][1][-int(float(caps["context_seconds"]) * rate):]
                await asyncio.to_thread(sf.write, str(context), tail, rate)
            raw = work / f"window_{k}.wav"
            await call_model(client, piece, fields, length, context, str(caps["context_field"]), raw)
            audio, got_rate = await asyncio.to_thread(read_audio, raw)
            if rate and got_rate != rate:
                raise HTTPException(status_code=502, detail="the model answered with different sample rates")
            rate = got_rate
            parts.append((start, audio))
            previous = raw
            log.info("  window %d/%d: %.1fs from %.1fs, %.1fs so far", k, parts_n, length, start, time.time() - started)

    final = await asyncio.to_thread(stitch, parts, rate, total, continuation) if parts_n > 1 \
        else parts[0][1][:int(round(total * rate))]
    out = work / f"generated.{output_format}"
    await asyncio.to_thread(sf.write, str(out), final, rate, "PCM_24" if output_format == "wav" else "PCM_24")
    set_progress(job_id, stage="done", part=parts_n, parts=parts_n, fraction=1.0,
                 detail=f"{total:.0f} s in {parts_n} part(s)")
    log.info("generate: done, %.1fs in %d part(s), %.1fs", total, parts_n, time.time() - started)
    name = f"generated_{seed}.{output_format}"
    return FileResponse(out, media_type="audio/wav" if output_format == "wav" else "audio/flac", filename=name,
                        headers={"Content-Disposition": f'attachment; filename="{name}"',
                                 "X-Generation-Parts": str(parts_n), "X-Duration": f"{total:.2f}",
                                 "X-Seed": str(seed), "X-Window-Seconds": f"{window:.0f}",
                                 "X-Generation-Time": f"{time.time() - started:.1f}",
                                 "X-Sample-Rate": str(rate)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
