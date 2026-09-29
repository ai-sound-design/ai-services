"""Spotting API: video in, sound events with timecodes out.

Replaces the Wizard-of-Oz spotting of the user study with a vision-language
model. The plugin sends the video range the user selected, plus the timecode
that range starts at; the service samples frames, shows them to the model with
their timestamps, and returns the sound events a sound editor would need to
cover, each with a start and end timecode.

The model is reached over HTTP and is interchangeable. Two API dialects are
supported, chosen with VLM_API:

  ollama   Ollama's /api/chat with a JSON schema in `format` (default)
  openai   any OpenAI-compatible /v1/chat/completions (LM Studio, vLLM, ...)

Nothing about the model is baked in; VLM_URL, VLM_MODEL and VLM_API are all
environment variables, so the same image serves a local Ollama, a container
next to it, or a remote endpoint.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "info").upper(),
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("spotting")

VLM_URL = os.environ.get("VLM_URL", "http://ollama:11434").rstrip("/")
VLM_MODEL = os.environ.get("VLM_MODEL", "gemma4:e4b-it-qat")
VLM_API = os.environ.get("VLM_API", "ollama")
VLM_TIMEOUT_S = float(os.environ.get("VLM_TIMEOUT_S", "600"))
SAMPLE_FPS = float(os.environ.get("SAMPLE_FPS", "2"))
FRAME_WIDTH = int(os.environ.get("FRAME_WIDTH", "512"))
FRAMES_PER_CALL = int(os.environ.get("FRAMES_PER_CALL", "8"))
# Ollama's default context (8192) is too small for eight frames: Gemma 4 then
# fails with "Failed to tokenize prompt". Its prompt_eval_count understates the
# real cost of an image by an order of magnitude, so size this generously.
VLM_NUM_CTX = int(os.environ.get("VLM_NUM_CTX", "32768"))
# 0 keeps repeated runs on the same range as alike as the model allows.
VLM_TEMPERATURE = float(os.environ.get("VLM_TEMPERATURE", "0"))
FRAME_STAMP = os.environ.get("FRAME_STAMP", "1") == "1"
STAMP_FONT = os.environ.get("STAMP_FONT", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

CATEGORIES = ("dialogue", "foley", "sfx", "ambience", "music")

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

app = FastAPI(title="Spotting API", version="0.1.0",
              description="Vision-language spotting of sound events in a video range.")


# ── Models ───────────────────────────────────────────────────────────────────

class Event(BaseModel):
    label: str = Field(description="Short name, usable as a marker name")
    category: Literal["dialogue", "foley", "sfx", "ambience", "music"]
    description: str = Field(description="What the sound is and how it behaves")
    start_seconds: float = Field(ge=0, description="Relative to the start of the sent video")
    end_seconds: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    start_timecode: str | None = None
    end_timecode: str | None = None


class SpotResponse(BaseModel):
    events: list[Event]
    video_duration_seconds: float
    frames_analysed: int
    model: str
    seconds_taken: float


# The schema the model must fill. Kept separate from the response model so the
# model never sees timecodes; it reasons in seconds, we convert afterwards.
EVENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                    "description": {"type": "string"},
                    "start_seconds": {"type": "number"},
                    "end_seconds": {"type": "number"},
                    "confidence": {"type": "number"},
                },
                "required": ["label", "category", "description",
                             "start_seconds", "end_seconds", "confidence"],
            },
        }
    },
    "required": ["events"],
}

# The model never hears anything (the audio is stripped before sampling), and
# the picture may not have a soundtrack yet. Saying so explicitly matters: asked
# to list "audible" events, gemma4:e4b-it-qat returned nothing for a cat walking
# past a robot vacuum and for a serval facing a dog; asked to predict what the
# visible sources would sound like, it named the vacuum motor, the paw steps and
# the room tone. Spelling out that dialogue is human speech keeps bird calls
# out of the dialogue category.
SYSTEM_PROMPT = """You are a film sound editor doing a spotting pass on a silent picture.
You see consecutive frames of one shot, each labelled with its time in seconds.
There is no audio: the soundtrack has yet to be designed. Predict, from what is
visible, which sounds this scene would produce, so that each one can be created
or found later. Go through the frames and name every visible sound source
(people, animals, machines, vehicles, weather, objects being handled, surfaces
being walked on, the room or landscape itself) and list the sound each would
make: actions (foley), effects (sfx), the ambience of the location, speech
(dialogue) and on-screen music. Dialogue means human speech only; animal
vocalisations, machine noises and impacts are sfx. Give each event the time span
in seconds during which it would be heard, judged from the frames in which its
source is active. Be concrete: "dog barks twice", not "animal sounds". Only list
sounds a visible source would make; do not invent off-screen sounds. If a sound
plausibly continues between two frames, span it. Use the label as a marker name:
short, specific, no punctuation."""


# ── Video handling ───────────────────────────────────────────────────────────

def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def extract_frames(video: Path, out_dir: Path, sample_fps: float) -> list[tuple[float, Path]]:
    """Sample frames at `sample_fps`, scaled to FRAME_WIDTH. Returns (seconds, file).

    Each frame gets its time burned into the corner. The model aligns events to
    what it can read far better than to a list of times in the prompt: in a
    side-by-side test the stamped frames yielded more events with correct spans.
    """
    pattern = out_dir / "f_%05d.jpg"
    filters = f"fps={sample_fps},scale={FRAME_WIDTH}:-2"
    if FRAME_STAMP and Path(STAMP_FONT).exists():
        filters += (f",drawtext=fontfile={STAMP_FONT}:text='t=%{{pts\\:flt}}s':x=8:y=8:fontsize=26:"
                    "fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=6")
    elif FRAME_STAMP:
        log.warning("FRAME_STAMP is on but %s is missing; frames go out unstamped", STAMP_FONT)
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", filters,
                    "-q:v", "4", str(pattern)], check=True)
    frames = sorted(out_dir.glob("f_*.jpg"))
    # ffmpeg's fps filter places frame k at k / fps seconds.
    return [(index / sample_fps, frame) for index, frame in enumerate(frames)]


def to_timecode(seconds: float, fps: float, start_tc: str) -> str:
    """Add `seconds` to a HH:MM:SS:FF timecode at integer `fps`."""
    hh, mm, ss, ff = (int(part) for part in start_tc.split(":"))
    rate = int(round(fps))
    base_frames = ((hh * 60 + mm) * 60 + ss) * rate + ff
    total = base_frames + int(round(seconds * rate))
    frames = total % rate
    total //= rate
    s = total % 60
    total //= 60
    m = total % 60
    h = total // 60
    return f"{h:02d}:{m:02d}:{s:02d}:{frames:02d}"


# ── Model access ─────────────────────────────────────────────────────────────

def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _frame_caption(frames: list[tuple[float, Path]], hints: str | None,
                   duration: float, sample_fps: float) -> str:
    # Spelling out the clip length, the sampling interval and the allowed range
    # is what makes the model place events in time. With a bare list of frame
    # times it put every event at the same instant.
    first, last = frames[0][0], frames[-1][0]
    times = ", ".join(f"{t:.1f}s" for t, _ in frames)
    text = (f"These {len(frames)} frames were sampled every {1 / sample_fps:.2f} s from a clip of "
            f"{duration:.2f} s; they cover {first:.1f}s to {last:.1f}s and each frame shows its "
            f"time in the top-left corner. Frame times: {times}.\n"
            f"List the sound events. start_seconds and end_seconds must lie between {first:.1f} "
            f"and {min(duration, last + 1 / sample_fps):.2f} and reflect when each sound would be "
            "heard, judged from the frames its source appears in.")
    if hints:
        text += f"\nContext from the editor: {hints}"
    text += "\nReturn JSON matching the schema."
    return text


async def call_ollama(client: httpx.AsyncClient, frames: list[tuple[float, Path]],
                      hints: str | None, model: str, duration: float, sample_fps: float) -> dict:
    body = {
        "model": model,
        "stream": False,
        "format": EVENT_SCHEMA,
        "options": {"temperature": VLM_TEMPERATURE, "num_ctx": VLM_NUM_CTX},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _frame_caption(frames, hints, duration, sample_fps),
             "images": [_b64(path) for _, path in frames]},
        ],
    }
    response = await client.post(f"{VLM_URL}/api/chat", json=body)
    response.raise_for_status()
    payload = response.json()
    # A model without a vision projector silently drops the images and answers
    # from the text alone (plain gemma4:e4b does this). Ollama gives no reliable
    # signal for it: for Gemma 4 the prompt token count barely moves with an
    # image (about 86 tokens each, seen or not). So this cannot be detected
    # here; the README names the variants that carry a projector.
    log.debug("%s: %s prompt tokens, %s completion tokens",
              model, payload.get("prompt_eval_count"), payload.get("eval_count"))
    return json.loads(payload["message"]["content"])


async def call_openai(client: httpx.AsyncClient, frames: list[tuple[float, Path]],
                      hints: str | None, model: str, duration: float, sample_fps: float) -> dict:
    content: list[dict] = [{"type": "text", "text": _frame_caption(frames, hints, duration, sample_fps)
                            + "\nSchema: " + json.dumps(EVENT_SCHEMA)}]
    for _, path in frames:
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64," + _b64(path)}})
    body = {
        "model": model,
        "temperature": VLM_TEMPERATURE,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": content}],
    }
    response = await client.post(f"{VLM_URL}/v1/chat/completions", json=body)
    response.raise_for_status()
    return json.loads(response.json()["choices"][0]["message"]["content"])


CALLERS = {"ollama": call_ollama, "openai": call_openai}


def merge_events(events: list[dict], duration: float) -> list[dict]:
    """Clamp to the video, drop junk, merge same-label events that touch."""
    cleaned = []
    for event in events:
        try:
            start = max(0.0, float(event["start_seconds"]))
            end = min(duration, float(event["end_seconds"]))
        except (KeyError, TypeError, ValueError):
            continue
        if end < start:
            start, end = end, start
        if event.get("category") not in CATEGORIES or not str(event.get("label", "")).strip():
            continue
        cleaned.append({**event, "start_seconds": round(start, 3), "end_seconds": round(end, 3),
                        "confidence": min(1.0, max(0.0, float(event.get("confidence", 0.5))))})
    cleaned.sort(key=lambda e: (e["label"].lower(), e["start_seconds"]))
    merged: list[dict] = []
    for event in cleaned:
        last = merged[-1] if merged else None
        if last and last["label"].lower() == event["label"].lower() \
                and event["start_seconds"] <= last["end_seconds"] + 0.5:
            last["end_seconds"] = max(last["end_seconds"], event["end_seconds"])
            last["confidence"] = max(last["confidence"], event["confidence"])
        else:
            merged.append(dict(event))
    merged = _fold_near_duplicates(merged)
    merged.sort(key=lambda e: e["start_seconds"])
    return merged


# Words that carry no identity: "room ambience" and "indoor room ambience" are one event.
_GENERIC_WORDS = frozenset("""a an the of on in at and or to its their general ambient ambience
    ambiance sound sounds noise noises presence tone background""".split())


def _label_words(label: str) -> list[str]:
    return [w for w in "".join(c if c.isalnum() else " " for c in label.lower()).split()
            if w not in _GENERIC_WORDS]


def _same_event(a: dict, b: dict) -> bool:
    """Two overlapping chunks describe one sound in slightly different words.

    "cheetah paws on floor" / "cheetah footsteps" and "robot operation" /
    "robot running" are the same marker; "cat walking" / "dog walking" are not.
    So the labels must name the same subject (same first content word) or one
    must be a wording of the other (its content words a subset), and the two
    spans must touch.
    """
    if a["category"] != b["category"]:
        return False
    if b["start_seconds"] > a["end_seconds"] + 0.5 or a["start_seconds"] > b["end_seconds"] + 0.5:
        return False
    if a["category"] == "ambience":
        # One scene has one atmosphere; "room ambience" in one chunk and "indoor
        # ambience" in the next are the same bed, whatever the model called it.
        return True
    wa, wb = _label_words(a["label"]), _label_words(b["label"])
    if not wa or not wb:
        return False
    return wa[0] == wb[0] or set(wa) <= set(wb) or set(wb) <= set(wa)


def _fold_near_duplicates(events: list[dict]) -> list[dict]:
    folded: list[dict] = []
    for event in sorted(events, key=lambda e: (e["start_seconds"], -e["confidence"])):
        for kept in folded:
            if _same_event(kept, event):
                kept["start_seconds"] = min(kept["start_seconds"], event["start_seconds"])
                kept["end_seconds"] = max(kept["end_seconds"], event["end_seconds"])
                kept["confidence"] = max(kept["confidence"], event["confidence"])
                if len(event["label"]) < len(kept["label"]):
                    kept["label"] = event["label"]          # the shorter wording makes the better marker name
                break
        else:
            folded.append(dict(event))
    return folded


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Reachability of the model endpoint, and whether the model is present."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            if VLM_API == "ollama":
                tags = (await client.get(f"{VLM_URL}/api/tags")).json()
                names = {m["name"] for m in tags.get("models", [])}
                present = VLM_MODEL in names or f"{VLM_MODEL}:latest" in names
            else:
                models = (await client.get(f"{VLM_URL}/v1/models")).json()
                present = any(m.get("id") == VLM_MODEL for m in models.get("data", []))
    except Exception as exc:  # any failure means "not healthy"
        raise HTTPException(503, f"model endpoint {VLM_URL} unreachable: {exc}")
    if not present:
        raise HTTPException(503, f"model {VLM_MODEL!r} not available at {VLM_URL}")
    return {"status": "ok", "vlm_url": VLM_URL, "model": VLM_MODEL, "api": VLM_API}


@app.get("/capabilities")
async def capabilities():
    """Self-description for the plugin's backend adapter."""
    return {
        "service": "spotting",
        "input": {"video": "multipart file", "start_timecode": "HH:MM:SS:FF",
                  "fps": "timecode frame rate", "hints": "optional free text"},
        "output": "events[] with start/end seconds and timecodes",
        "categories": list(CATEGORIES),
        "model": VLM_MODEL,
        "sample_fps": SAMPLE_FPS,
        "progress": "/spot/progress/{job_id}",
    }


@app.get("/spot/progress/{job_id}")
async def spot_progress(job_id: str):
    """State of a running `/spot` request that was sent with this `job_id`:
    `stage`, `fraction` (0..1 of the model calls done) and a `detail` text."""
    entry = PROGRESS.get(job_id)
    if entry is None:
        raise HTTPException(404, "unknown job")
    return entry


@app.post("/spot", response_model=SpotResponse)
async def spot(
    video: UploadFile = File(...),
    start_timecode: str = Form("00:00:00:00"),
    fps: float = Form(30.0),
    hints: str | None = Form(None),
    sample_fps: float | None = Form(None),
    model: str | None = Form(None),
    job_id: str | None = Form(None),
):
    t0 = time.time()
    model = model or VLM_MODEL
    sample_fps = sample_fps or SAMPLE_FPS
    caller = CALLERS.get(VLM_API)
    if caller is None:
        raise HTTPException(500, f"unknown VLM_API {VLM_API!r}; use ollama or openai")

    workdir = Path(tempfile.mkdtemp(prefix="spot_"))
    try:
        video_path = workdir / (Path(video.filename or "input.mp4").name)
        with video_path.open("wb") as handle:
            shutil.copyfileobj(video.file, handle)
        set_progress(job_id, stage="extracting frames", fraction=0.0, detail="")
        duration = probe_duration(video_path)
        frames = extract_frames(video_path, workdir, sample_fps)
        if not frames:
            raise HTTPException(400, "no frames could be extracted from the video")
        log.info("spotting %s: %.2fs, %d frames at %.1f fps, model %s",
                 video_path.name, duration, len(frames), sample_fps, model)

        # Every call sees FRAMES_PER_CALL frames if the video has that many; the
        # last chunk is pulled back to full size and overlaps the previous one.
        # Smaller chunks lose events: with 4-5 frames per call the model dropped
        # the first half of a test clip that it found reliably with 8. A call
        # that sees one frame cannot judge duration at all. Overlaps are folded
        # by merge_events.
        size = min(FRAMES_PER_CALL, len(frames))
        starts = sorted({min(i, len(frames) - size) for i in range(0, len(frames), size)})
        set_progress(job_id, stage="analysing", fraction=0.0,
                     detail=f"step 1 of {len(starts)}: frames 1-{size} of {len(frames)}",
                     frames_total=len(frames), calls_total=len(starts), calls_done=0)

        raw_events: list[dict] = []
        async with httpx.AsyncClient(timeout=VLM_TIMEOUT_S) as client:
            for k, i in enumerate(starts, start=1):
                chunk = frames[i:i + size]
                # What is being analysed now; the fraction counts model calls, each of which
                # costs the same, so the bar moves in steps of one call.
                set_progress(job_id, stage="analysing", fraction=(k - 1) / len(starts),
                             detail=f"step {k} of {len(starts)}: frames {i + 1}-{i + size} of {len(frames)}",
                             calls_done=k - 1)
                try:
                    result = await caller(client, chunk, hints, model, duration, sample_fps)
                except httpx.HTTPStatusError as exc:
                    raise HTTPException(502, f"model endpoint returned {exc.response.status_code}: "
                                             f"{exc.response.text[:300]}")
                except httpx.HTTPError as exc:
                    raise HTTPException(502, f"model endpoint {VLM_URL} failed: {exc}")
                found = result.get("events", [])
                log.info("frames %.1fs-%.1fs: %d events: %s", chunk[0][0], chunk[-1][0], len(found),
                         "; ".join(f"{e.get('label')} {e.get('start_seconds')}-{e.get('end_seconds')}"
                                   for e in found))
                raw_events.extend(found)
                set_progress(job_id, stage="analysing", fraction=k / len(starts),
                             detail=f"step {k} of {len(starts)} done, {len(frames)} frames", calls_done=k)

        set_progress(job_id, stage="done", fraction=1.0, detail=f"{len(starts)} steps, {len(frames)} frames")
        events = merge_events(raw_events, duration)
        for event in events:
            # A sound seen in one frame is audible at least until the next sample.
            if event["end_seconds"] - event["start_seconds"] < 1 / sample_fps:
                event["end_seconds"] = round(min(duration, event["start_seconds"] + 1 / sample_fps), 3)
            event["start_timecode"] = to_timecode(event["start_seconds"], fps, start_timecode)
            event["end_timecode"] = to_timecode(event["end_seconds"], fps, start_timecode)
        return SpotResponse(events=events, video_duration_seconds=round(duration, 3),
                            frames_analysed=len(frames), model=model,
                            seconds_taken=round(time.time() - t0, 2))
    except subprocess.CalledProcessError as exc:
        raise HTTPException(400, f"could not read the video: {exc}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        if job_id and PROGRESS.get(job_id, {}).get("stage") != "done":
            set_progress(job_id, stage="failed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("API_HOST", "0.0.0.0"),
                port=int(os.environ.get("API_PORT", "8003")))
