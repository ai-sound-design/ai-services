"""Hybrid API: a video range in, one generated sound per sound event out.

The plugin's hybrid mode wants the individual sounds of a scene, each with its
own start and end, rather than one mix for the whole range. This service is
the reference implementation and composes the two services next to it:

  1. The events come with the request (`events`: the memory locations inside
     the range, as JSON) or, when absent, from the spotting service.
  2. For every event the matching part of the video is cut out and sent to the
     generation service with the event's description as the prompt.
  3. Each answer is trimmed to the event's length and offered for download.
  4. With `match` on, each generated sound is also replaced by pieces of
     library recordings that sound like it (matching.py, via the search
     service's /search/by_audio); the pieces come back next to the sound.

A future model that does all of this in one pass replaces this service without
a change to the plugin: it only has to fulfil the same contract (see README).

Environment:
    GENERATION_URL   http://mmaudio-api:8000     video (or prompt) -> audio
    SPOTTING_URL     http://spotting-api:8003    video -> events, used when no events are sent
    SEARCH_URL       http://sound-search-api:8002 sound -> library windows, used with `match`
    GEN_MIN_SECONDS  4     the generation model's shortest length: shorter events are padded, then trimmed
    GEN_MAX_SECONDS  12    its longest length: longer events get a sound of this length
    OUTPUT_DIR       /data/hybrid                where the sounds are kept for download
    KEEP_HOURS       24                          how long
"""
from __future__ import annotations

import json
import logging
import os
import random
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from matching import match_generated

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "info").upper(),
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("hybrid")

GENERATION_URL = os.environ.get("GENERATION_URL", "http://mmaudio-api:8000").rstrip("/")
SPOTTING_URL = os.environ.get("SPOTTING_URL", "http://spotting-api:8003").rstrip("/")
SEARCH_URL = os.environ.get("SEARCH_URL", "http://sound-search-api:8002").rstrip("/")
GEN_MIN_SECONDS = float(os.environ.get("GEN_MIN_SECONDS", "4"))
GEN_MAX_SECONDS = float(os.environ.get("GEN_MAX_SECONDS", "12"))
CUT_MARGIN = 0.25   # seconds of extra video per piece, see the generation loop
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/data/hybrid"))
KEEP_HOURS = float(os.environ.get("KEEP_HOURS", "24"))
GENERATION_TIMEOUT_S = float(os.environ.get("GENERATION_TIMEOUT_S", "900"))
SPOTTING_TIMEOUT_S = float(os.environ.get("SPOTTING_TIMEOUT_S", "900"))

app = FastAPI(title="Hybrid API", version="0.1.0",
              description="One generated sound per sound event in a video range.")


# ── helpers ──────────────────────────────────────────────────────────────────

def ffprobe_duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def cut_video(src: Path, start: float, length: float, dst: Path) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{length:.3f}",
                    "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", str(dst)],
                   check=True, capture_output=True, text=True)


def trim_audio(src: Path, length: float, dst: Path) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-t", f"{length:.3f}", "-c:a", "pcm_s24le", str(dst)],
                   check=True, capture_output=True, text=True)


def housekeeping() -> None:
    cutoff = time.time() - KEEP_HOURS * 3600
    for f in OUTPUT_DIR.glob("*.wav"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass


async def spot_events(video: Path, start_timecode: str, fps: float,
                      job_id: Optional[str] = None) -> tuple[list[dict], Optional[str]]:
    data = {"start_timecode": start_timecode, "fps": str(fps)}
    if job_id:
        data["job_id"] = job_id
    async with httpx.AsyncClient(timeout=SPOTTING_TIMEOUT_S) as client:
        with video.open("rb") as handle:
            response = await client.post(f"{SPOTTING_URL}/spot", files={"video": (video.name, handle, "video/mp4")},
                                         data=data)
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"spotting service: {response.status_code} {response.text[:300]}")
    body = response.json()
    return body.get("events", []), body.get("model")


async def generate(video: Path, prompt: str, negative_prompt: str, seed: int, duration: float, dst: Path) -> None:
    data = {"prompt": prompt, "negative_prompt": negative_prompt, "seed": str(seed),
            "duration": f"{duration:.2f}", "output_format": "wav"}
    async with httpx.AsyncClient(timeout=GENERATION_TIMEOUT_S) as client:
        with video.open("rb") as handle:
            response = await client.post(f"{GENERATION_URL}/generate",
                                         files={"video": (video.name, handle, "video/mp4")}, data=data)
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"generation service: {response.status_code} {response.text[:300]}")
    dst.write_bytes(response.content)


def parse_events(raw: Optional[str], video_length: float) -> list[dict]:
    if not raw:
        return []
    try:
        items = json.loads(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"events is not valid JSON: {exc}")
    events = []
    for item in items if isinstance(items, list) else []:
        try:
            start = max(0.0, float(item.get("start_seconds", 0.0)))
            end = float(item.get("end_seconds", video_length))
        except (TypeError, ValueError):
            continue
        end = min(max(end, start), video_length)
        if start >= video_length:
            continue
        events.append({"label": str(item.get("label") or "sound"), "description": str(item.get("description") or ""),
                       "category": str(item.get("category") or "sfx"), "start_seconds": start, "end_seconds": end})
    return events


async def match_sound(generated: Path, work: Path, sound: dict, job_id: Optional[str], n: int, total: int,
                      base: float, pieces_per_10s: int, min_piece_seconds: float, layers: int,
                      text_weight: float, library: Optional[str], category_filter: bool,
                      min_similarity: float = 0.0, ambience_handle_seconds: float = 0.0,
                      event_seconds: float = 0.0) -> dict:
    """Library pieces that sound like `generated`, kept for download next to it.
    An ambience gets `ambience_handle_seconds` of the recording before and after each
    piece (for fades), and a recording that matched it as a whole is cut in the length
    of the event (`event_seconds`) when that is longer than the generated sound; other
    categories are cut to the piece. A failure here is reported in the answer and does
    not lose the generated sound."""
    ambience = str(sound.get("category", "")).lower() == "ambience"
    handle = ambience_handle_seconds if ambience else 0.0
    settings = {"pieces_per_10s": pieces_per_10s, "min_piece_seconds": min_piece_seconds, "layers": layers,
                "text_weight": text_weight, "library": library, "min_similarity": min_similarity,
                "handle_seconds": handle, "extend_to_seconds": event_seconds if ambience else 0.0}

    def progress(call: int, calls: int) -> None:
        set_progress(job_id, stage="matching",
                     fraction=base + (1 - base) * (n - 0.5) / max(1, total),
                     detail=f"sound {n} of {total}: {sound['label']}, library search {call}")

    try:
        found = await match_generated(generated, work, search_url=SEARCH_URL, text=sound.get("description", ""),
                                      settings=settings, category=sound.get("category") if category_filter else None,
                                      on_progress=progress)
    except Exception as exc:  # noqa: BLE001
        log.warning("match for %s failed: %s", sound["label"], exc)
        return {"pieces": [], "error": str(exc)[:300]}
    pieces = []
    for piece in found:
        piece_id = uuid.uuid4().hex
        # shutil.move, not rename: the work folder and OUTPUT_DIR are different file systems
        shutil.move(str(piece.pop("path")), str(OUTPUT_DIR / f"{piece_id}.wav"))
        pieces.append({**piece, "id": piece_id, "audio_url": f"/hybrid/files/{piece_id}.wav"})
    log.info("    %d library piece(s) in %d layer(s) for %s", len(pieces), layers, sound["label"])
    return {"pieces": pieces, "pieces_per_10s": pieces_per_10s, "layers": layers, "handle_seconds": handle}


# ── endpoints ────────────────────────────────────────────────────────────────

# ── Progress per job ─────────────────────────────────────────────────────────
# A caller that sends a `job_id` with its request can poll the state of that
# request while it waits: one entry per job, in memory, pruned after an hour.
PROGRESS: dict[str, dict] = {}


def set_progress(job_id: str | None, **fields) -> None:
    if not job_id:
        return
    now = time.time()
    for key in [k for k, v in PROGRESS.items() if now - v.get("updated", now) > 3600]:
        PROGRESS.pop(key, None)
    entry = PROGRESS.setdefault(job_id, {})
    entry.update(fields, updated=now)


@app.get("/hybrid/progress/{job_id}")
async def hybrid_progress(job_id: str):
    """State of a running `/hybrid` request sent with this `job_id`: `stage`
    (spotting, generating, done), `fraction` (0..1 of the whole request; spotting
    counts as the first half when the backend has to find the events) and `detail`."""
    entry = PROGRESS.get(job_id)
    if entry is None:
        # A /hybrid/scenes request runs entirely in the spotting service under the same id.
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                got = await client.get(f"{SPOTTING_URL}/spot/progress/{job_id}")
            if got.status_code == 200:
                return got.json()
        except httpx.HTTPError:
            pass
        raise HTTPException(status_code=404, detail="unknown job")
    if entry.get("stage") == "spotting":
        # The spotting service got the same job_id; fold its progress into ours.
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                got = await client.get(f"{SPOTTING_URL}/spot/progress/{job_id}")
            if got.status_code == 200:
                spot = got.json()
                return {**entry, "fraction": 0.5 * float(spot.get("fraction", 0.0)),
                        "detail": ("spotting, " + spot["detail"]) if spot.get("detail") else "spotting"}
        except httpx.HTTPError:
            pass
    return entry


@app.get("/health")
async def health():
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            gen = (await client.get(f"{GENERATION_URL}/health")).status_code == 200
        except Exception:
            gen = False
        try:
            spot = (await client.get(f"{SPOTTING_URL}/health")).status_code == 200
        except Exception:
            spot = False
        # Library match: the search service must be up and hold an audio index.
        match = {"available": False, "reason": "search service not reachable", "windows": 0}
        try:
            answer = await client.get(f"{SEARCH_URL}/health")
            if answer.status_code == 200:
                info = answer.json().get("audio_search") or {}
                match = {"available": bool(info.get("available")),
                         "reason": None if info.get("available") else (info.get("reason") or "no audio index"),
                         "windows": int(info.get("windows") or 0), "sounds": int(info.get("sounds") or 0)}
            else:
                match["reason"] = f"search service answered {answer.status_code}"
        except Exception:
            pass
    if not gen:
        raise HTTPException(status_code=503, detail="generation service not reachable")
    return {"status": "ok", "generation": GENERATION_URL, "spotting": SPOTTING_URL if spot else None,
            "spotting_available": spot, "scenes_available": spot,
            "min_seconds": GEN_MIN_SECONDS, "max_seconds": GEN_MAX_SECONDS,
            "search": SEARCH_URL, "database_match": match}


@app.post("/hybrid/scenes")
async def hybrid_scenes(
    videos: list[UploadFile] = File(...),
    names: Optional[str] = Form(None),
    job_id: Optional[str] = Form(None),
):
    """Which consecutive clips form one scene: forwarded to the spotting service's
    `/scenes` (see spotting/README.md). `videos` are the clips of the range in
    timeline order, `names` an optional JSON list of their names."""
    files = []
    for n, upload in enumerate(videos, start=1):
        files.append(("videos", (Path(upload.filename or f"clip_{n}.mp4").name, await upload.read(), "video/mp4")))
    data = {}
    if names:
        data["names"] = names
    if job_id:
        data["job_id"] = job_id
    async with httpx.AsyncClient(timeout=SPOTTING_TIMEOUT_S) as client:
        response = await client.post(f"{SPOTTING_URL}/scenes", files=files, data=data)
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"spotting service: {response.status_code} {response.text[:300]}")
    return response.json()


@app.post("/hybrid")
async def hybrid(
    video: UploadFile = File(...),
    start_timecode: str = Form("00:00:00:00"),
    fps: float = Form(25.0),
    prompt: str = Form(""),
    negative_prompt: str = Form(""),
    seed: int = Form(42),
    events: Optional[str] = Form(None),
    job_id: Optional[str] = Form(None),
    match: bool = Form(False),
    pieces_per_10s: int = Form(3),
    min_piece_seconds: float = Form(2.0),
    layers: int = Form(1),
    text_weight: float = Form(0.0),
    library: Optional[str] = Form(None),
    category_filter: bool = Form(False),
    min_similarity: float = Form(0.0),
    ambience_handle_seconds: float = Form(0.0),
):
    """One sound per event; with `match`, library pieces that sound like each one alongside.
    `events` (JSON list, optional) are relative to the start of the sent video.
    `ambience_handle_seconds`: an ambience piece keeps that much of its recording before
    and after the matched stretch (`handle_before_seconds` per piece), for fades."""
    started = time.time()
    if seed < 0:  # -1: pick a base seed; event n uses seed + n
        seed = random.randint(0, 2**30)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    housekeeping()
    work = Path(tempfile.mkdtemp(prefix="hybrid_"))
    try:
        src = work / (Path(video.filename or "range.mp4").name)
        src.write_bytes(await video.read())
        length = ffprobe_duration(src)
        if length <= 0:
            raise HTTPException(status_code=400, detail="could not read the video")

        given = parse_events(events, length)
        model = None
        if given:
            found, source = given, "memory_locations"
        else:
            set_progress(job_id, stage="spotting", fraction=0.0, detail="spotting")
            found, model = await spot_events(src, start_timecode, fps, job_id)
            source = "spotting"
        base = 0.5 if source == "spotting" else 0.0   # spotting took the first half of the bar
        found = [e for e in found if float(e.get("end_seconds", 0)) > float(e.get("start_seconds", 0))]
        log.info("%s: %.1fs, %d events (%s)", src.name, length, len(found), source)

        sounds: list[dict[str, Any]] = []
        for n, event in enumerate(found, start=1):
            set_progress(job_id, stage="generating", fraction=base + (1 - base) * (n - 1) / max(1, len(found)),
                         detail=f"sound {n} of {len(found)}: {event.get('label') or 'sound'}")
            start = float(event["start_seconds"])
            end = min(float(event["end_seconds"]), length)
            wanted = end - start
            # The generation model has a length window; pad short events (then trim), cap long ones.
            # The piece is cut a little longer than asked: ffmpeg rounds to whole frames and a
            # 3.97 s piece would fall under a 4 s minimum.
            gen_len = min(GEN_MAX_SECONDS, max(GEN_MIN_SECONDS, wanted))
            cut_start = min(start, max(0.0, length - gen_len - CUT_MARGIN))
            cut_len = min(gen_len + CUT_MARGIN, length - cut_start)
            piece = work / f"event_{n}.mp4"
            cut_video(src, cut_start, cut_len, piece)

            text = (event.get("description") or event.get("label") or "").strip()
            full_prompt = ", ".join(p for p in (text, prompt.strip()) if p)
            raw = work / f"event_{n}_raw.wav"
            await generate(piece, full_prompt, negative_prompt, seed + n, gen_len, raw)

            sound_id = uuid.uuid4().hex
            final = OUTPUT_DIR / f"{sound_id}.wav"
            trim_audio(raw, min(wanted, gen_len), final)
            sounds.append({"id": sound_id, "label": event.get("label") or f"sound {n}",
                           "category": event.get("category", "sfx"), "description": text,
                           "start_seconds": round(start, 3), "end_seconds": round(start + min(wanted, gen_len), 3),
                           "audio_url": f"/hybrid/files/{sound_id}.wav"})
            log.info("  %d/%d %s: %.1fs from %.1fs", n, len(found), sounds[-1]["label"], wanted, start)

            if match:
                sounds[-1]["match"] = await match_sound(final, work, sounds[-1], job_id, n, len(found), base,
                                                        pieces_per_10s, min_piece_seconds, layers, text_weight,
                                                        library, category_filter, min_similarity,
                                                        max(0.0, ambience_handle_seconds), wanted)

        set_progress(job_id, stage="done", fraction=1.0, detail=f"{len(sounds)} sounds")
        return {"sounds": sounds, "events_source": source, "video_duration_seconds": round(length, 3),
                "model": model or "hybrid: spotting + generation", "seconds_taken": round(time.time() - started, 1)}
    finally:
        for f in work.glob("*"):
            f.unlink(missing_ok=True)
        if job_id and PROGRESS.get(job_id, {}).get("stage") != "done":
            set_progress(job_id, stage="failed")
        work.rmdir()


@app.get("/hybrid/files/{name}")
async def get_file(name: str):
    path = OUTPUT_DIR / Path(name).name
    if not path.exists() or path.suffix.lower() != ".wav":
        raise HTTPException(status_code=404, detail="no such sound")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("API_PORT", "8004")))
