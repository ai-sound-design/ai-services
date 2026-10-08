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
import re
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
# A hosted OpenAI-compatible service (OpenAI itself, a gateway) wants a key; a local
# LM Studio or vLLM does not. Sent as a bearer token with every request when set.
VLM_API_KEY = os.environ.get("VLM_API_KEY", "").strip()
VLM_HEADERS = {"Authorization": f"Bearer {VLM_API_KEY}"} if VLM_API_KEY else {}
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
# The longest answer a call may produce. A grammar-constrained generation can run
# away; without this cap one such call filled the context for ten minutes and took
# the whole run down. The events of eight frames fit in a few hundred tokens.
VLM_NUM_PREDICT = int(os.environ.get("VLM_NUM_PREDICT", "2048"))
FRAME_STAMP = os.environ.get("FRAME_STAMP", "1") == "1"
STAMP_FONT = os.environ.get("STAMP_FONT", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
# Cuts inside a clip: a foley or sound-effect event that runs across a picture cut is
# split there, one event per shot, so each sound is made for the shot it belongs to.
# Ambience, music and dialogue carry on across the cuts of a scene. The detector
# compares perceptual hashes of consecutive frames (PySceneDetect's HashDetector):
# at 0.3 it found every cut of the test clips and none of their fast camera moves or
# frame-rate stutter (those stay below 0.25). Pieces shorter than CUT_MIN_PIECE_S go.
CUT_DETECTION = os.environ.get("CUT_DETECTION", "1") == "1"
CUT_THRESHOLD = float(os.environ.get("CUT_THRESHOLD", "0.3"))
CUT_MIN_PIECE_S = float(os.environ.get("CUT_MIN_PIECE_S", "0.3"))
CUT_SPLIT_CATEGORIES = ("foley", "sfx")
# The scene-context pass: frames a few seconds apart over the whole video, from which
# the model describes each scene (place, what the people do, expected sounds, the
# ambience). Every dense chunk call then gets its scene's description, and the
# ambience comes from the scene, not from 4-second chunks.
SCENE_CONTEXT = os.environ.get("SCENE_CONTEXT", "1") == "1"
SCENE_FPS = float(os.environ.get("SCENE_FPS", "0.25"))
SCENE_CHECKLIST = os.environ.get("SCENE_CHECKLIST", "0") == "1"    # expected sounds go into the chunk calls (more found, more invented)
# How much of the scene description a chunk call is given: "full" (place and what
# happens), "place" (the place only), "none" (the scene pass only sets the ambience
# and the scene boundaries). A narrative of actions makes a small model list those
# actions in every chunk, whether the frames show them or not.
SCENE_HINT = os.environ.get("SCENE_HINT", "place")
# A scene shorter than this is a shot, not a scene: it joins its neighbour.
MIN_SCENE_S = float(os.environ.get("MIN_SCENE_S", "10"))
# Every event must cite the frames that show its source acting; an event the model
# cannot ground in a frame is not listed. Costs a few tokens per event.
EVENT_EVIDENCE = os.environ.get("EVENT_EVIDENCE", "0") == "1"
# The model first writes, per frame, what people and things visibly do, then lists
# the events: an event has to come from an observation, not from the place.
EVENT_OBSERVE = os.environ.get("EVENT_OBSERVE", "0") == "1"
# Same-label events closer than this are one event: a run of steps or of typing is
# one marker, as a sound editor would lay one clip. Cuts still split it.
MERGE_GAP_S = float(os.environ.get("MERGE_GAP_S", "2.0"))
# After the chunk pass the model tidies the list the way an editor would: one marker
# per source and action, the engine, tyres and rolling of one car are one pass-by,
# and what belongs to the atmosphere (distant traffic, street bustle) is not a marker.
CONSOLIDATE = os.environ.get("CONSOLIDATE", "0") == "1"    # off: a small model drops beds and renames sounds

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
    evidence: str | None = Field(None, description="The frames that show the source acting, when asked for")
    start_timecode: str | None = None
    end_timecode: str | None = None


class SceneContext(BaseModel):
    start_seconds: float
    end_seconds: float
    place: str
    what_happens: str
    expected_sounds: list[str] = Field(default_factory=list)


class SpotResponse(BaseModel):
    events: list[Event]
    cuts: list[float] = Field(default_factory=list, description="Picture cuts, seconds from the start of the sent video")
    scenes: list[SceneContext] = Field(default_factory=list, description="The scene-context pass, if it ran")
    warnings: list[str] = Field(default_factory=list, description="Calls that failed and were skipped")
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
                    "temporal": {"type": "string", "enum": ["discrete", "continuous", "stationary"]},
                },
                "required": ["label", "category", "description",
                             "start_seconds", "end_seconds", "confidence", "temporal"],
            },
        }
    },
    "required": ["events"],
}
def with_observations(schema: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = json.loads(json.dumps(schema))
    observations = {"type": "array", "items": {"type": "object", "properties": {
        "time_seconds": {"type": "number"}, "seen": {"type": "string"}}, "required": ["time_seconds", "seen"]}}
    out["properties"] = {"observations": observations, **out["properties"]}   # written before the events
    out["required"] = ["observations"] + out["required"]
    return out


EVENT_SCHEMA_EVIDENCE: dict[str, Any] = json.loads(json.dumps(EVENT_SCHEMA))
EVENT_SCHEMA_EVIDENCE["properties"]["events"]["items"]["properties"]["evidence"] = {"type": "string"}
EVENT_SCHEMA_EVIDENCE["properties"]["events"]["items"]["required"].append("evidence")

# The kinds of sound an editor can ask for, and the words the model is given for them
EVENT_CATEGORIES = ("dialogue", "foley", "sfx", "ambience", "music")
CATEGORY_WORDS = {"dialogue": "dialogue (human speech)",
                  "foley": "foley (footsteps, clothes, objects handled by people)",
                  "sfx": "sound effects (machines, vehicles, animals, impacts, weather)",
                  "ambience": "ambience (the room tone or atmosphere of the place)",
                  "music": "music with a visible source"}


def wanted_categories(categories: str | None) -> list[str]:
    """The kinds named in a comma-separated `categories`; empty when all or none are named."""
    named = [c.strip().lower() for c in (categories or "").split(",") if c.strip()]
    wanted = [c for c in EVENT_CATEGORIES if c in named]
    return wanted if 0 < len(wanted) < len(EVENT_CATEGORIES) else []


def category_hint(wanted: list[str], hints: str | None) -> str | None:
    """The editor's hint to the model, extended by which kinds of sound are wanted."""
    if not wanted:
        return hints
    text = "Only these kinds of sound are needed: " + "; ".join(CATEGORY_WORDS[c] for c in wanted) + "."
    return f"{hints.strip()} {text}" if hints and hints.strip() else text
SYSTEM_PROMPT = """You are a film sound editor doing a spotting pass on a silent picture.
You see consecutive frames of one shot, each labelled with its time in seconds.
There is no audio: the soundtrack has yet to be designed. Predict, from what is
visible, which sounds this scene would produce, so that each one can be created
or found later. Go through the frames and name every visible sound source
(people, animals, machines, vehicles, weather, objects being handled, surfaces
being walked on, the room or landscape itself) and list the sound each would
make: actions (foley), effects (sfx), the ambience of the location, speech
(dialogue) and on-screen music. For each person check their hands (typing,
writing, turning pages, picking up or putting down objects), their body
(sitting down, getting up, a chair taking weight or rolling, clothes moving) and
their feet (steps, and on what surface). Anything that moves fast or flies
through the frame makes a whoosh or pass-by. Breathing counts only when it is
visibly strong (panting, gasping). One event per source and action: the engine,
the tyres and the rolling of one car are one event ("taxi passes"), not three.
What belongs to the atmosphere of the place (distant or background traffic,
other vehicles in the background, street bustle, crowd murmur, a room's hum) is
the ambience event, not separate events. Dialogue means human speech only; animal
vocalisations, machine noises and impacts are sfx. Give each event the time span
in seconds during which it would be heard, judged from the frames in which its
source is active. Be concrete: "dog barks twice", not "animal sounds". Only list
sounds a visible source would make; do not invent off-screen sounds. If a sound
plausibly continues between two frames, span it. Use the label as a marker name:
two to five words, specific, no punctuation, naming the sound and not only its
source: "robot vacuum whirring" and "cat meow", never just "robot" or "cat". Say for each event how it behaves in time:
"discrete" for separate repeated events (footsteps, barks, knocks, clicks),
"continuous" for a sound that runs on with changes (an engine, rain on a roof,
a hum that swells), "stationary" for a steady bed that hardly changes (room
tone, wind, distant traffic)."""


CONTEXT_SYSTEM_PROMPT = """You are a film sound editor preparing a spotting session on a silent picture.
You see frames sampled a few seconds apart, each labelled with its time in seconds.
Divide what you see into scenes: a scene is one place and one continuous stretch of
story time; a new camera angle of the same place at the same moment is the same
scene, a different place or a clear jump in time starts a new one. For each scene
give the place, and describe the characters and what they do, with attention to
hands (typing, writing, turning pages, handling objects), bodies (sitting down,
getting up, chairs, clothes) and feet (walking, on what surface), and to anything
that moves fast or flies. Name the ambience of the place: the steady background
sound a sound editor lays under the whole scene (the room tone of an office, wind
on a mountain, the traffic of a street): ambience_label names the sound in two to
four words ("light mountain wind", "office room tone"), ambience_description is
one sentence about its character. List the sounds a sound editor would expect to spot in the scene,
concrete and short ("keyboard typing", "page turns", "office chair creaks",
"footsteps on snow", "whoosh of the flying star")."""

CONTEXT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scenes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_seconds": {"type": "number"},
                    "end_seconds": {"type": "number"},
                    "place": {"type": "string"},
                    "what_happens": {"type": "string"},
                    "ambience_label": {"type": "string"},
                    "ambience_description": {"type": "string"},
                    "expected_sounds": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["start_seconds", "end_seconds", "place", "what_happens",
                             "ambience_label", "ambience_description", "expected_sounds"],
            },
        }
    },
    "required": ["scenes"],
}


def fold_scenes(scenes: list[dict], duration: float, gap: float) -> list[dict]:
    """Scenes from consecutive context windows that name the same place and touch
    (within `gap`, the sampling interval) are one scene; the scenes then tile the video."""
    cleaned = []
    for s in scenes:
        try:
            start, end = max(0.0, float(s["start_seconds"])), min(duration, float(s["end_seconds"]))
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start or not str(s.get("place", "")).strip():
            continue
        cleaned.append({"start_seconds": start, "end_seconds": end, "place": str(s["place"]).strip(),
                        "what_happens": str(s.get("what_happens", "")).strip(),
                        "ambience_label": str(s.get("ambience_label", "")).strip(),
                        "ambience_description": str(s.get("ambience_description", "")).strip(),
                        "expected_sounds": [str(x).strip() for x in (s.get("expected_sounds") or []) if str(x).strip()]})
    cleaned.sort(key=lambda s: s["start_seconds"])
    def join(last: dict, s: dict) -> None:
        last["end_seconds"] = max(last["end_seconds"], s["end_seconds"])
        if s["what_happens"] and s["what_happens"] not in last["what_happens"]:
            last["what_happens"] = (last["what_happens"] + " " + s["what_happens"]).strip()
        for x in s["expected_sounds"]:
            if x.lower() not in {y.lower() for y in last["expected_sounds"]}:
                last["expected_sounds"].append(x)
        if not last["ambience_label"]:
            last["ambience_label"], last["ambience_description"] = s["ambience_label"], s["ambience_description"]

    def same_place(a: dict, b: dict) -> bool:
        # "Cluttered office workspace" and "Office desk" are one place: any shared word.
        # The model's windows are 32 s, so one place comes back in several wordings; a
        # "Desk surface" after an office whose description mentions the desk is a close-up.
        wa, wb = set(_label_words(a["place"])), set(_label_words(b["place"]))
        return bool(wa & wb) or bool(wb & set(_label_words(a["what_happens"])))

    folded: list[dict] = []
    for s in cleaned:
        last = folded[-1] if folded else None
        if last and s["start_seconds"] <= last["end_seconds"] + gap + 0.5 and same_place(last, s):
            join(last, s)
            continue
        folded.append(dict(s))
    # A "scene" of a few seconds is a shot the model singled out (a close-up of hands,
    # a pan across the desk): it belongs to the scene around it.
    changed = True
    while changed and len(folded) > 1:
        changed = False
        for n, s in enumerate(folded):
            if s["end_seconds"] - s["start_seconds"] < MIN_SCENE_S:
                if n > 0:
                    join(folded[n - 1], s)
                else:
                    folded[1]["start_seconds"] = s["start_seconds"]
                    if not folded[1]["ambience_label"]:
                        folded[1]["ambience_label"], folded[1]["ambience_description"] = s["ambience_label"], s["ambience_description"]
                del folded[n]
                changed = True
                break
    # No gaps and no overlaps: a scene lasts until the next one starts.
    for a, b in zip(folded, folded[1:]):
        a["end_seconds"] = b["start_seconds"]
    if folded:
        folded[0]["start_seconds"] = 0.0
        folded[-1]["end_seconds"] = duration
    folded = [s for s in folded if s["end_seconds"] > s["start_seconds"]]
    # Title cards, credits and black are not a place: nothing to spot in them, and
    # they carry the ambience of the scene next to them, as a sound designer lays
    # the first scene's bed under the opening titles.
    for s in folded:
        s["is_title"] = bool(TITLE_WORDS.search(s["place"] + " " + s["what_happens"]))
    for n, s in enumerate(folded):
        if s["is_title"]:
            s["ambience_label"], s["ambience_description"] = "", ""
    return folded


TITLE_WORDS = re.compile(r"\b(title card|title cards|title sequence|titles|credits|black screen|black frame|"
                         r"blank screen|end card|intertitle)\b", re.IGNORECASE)


def scene_ambiences(scenes: list[dict]) -> list[dict]:
    """One ambience event per scene that has one; title scenes extend their neighbour's."""
    events: list[dict] = []
    spans: list[tuple[int, float, float]] = []       # (scene index with a bed, start, end)
    for n, s in enumerate(scenes):
        if s["ambience_label"]:
            spans.append((n, s["start_seconds"], s["end_seconds"]))
    if not spans:
        return events
    for n, s in enumerate(scenes):
        if s["ambience_label"]:
            continue
        before = [i for i, (k, _, _) in enumerate(spans) if k < n]
        after = [i for i, (k, _, _) in enumerate(spans) if k > n]
        if before:
            k, a, _ = spans[before[-1]]
            spans[before[-1]] = (k, a, max(spans[before[-1]][2], s["end_seconds"]))
        elif after:
            k, _, b = spans[after[0]]
            spans[after[0]] = (k, min(spans[after[0]][1], s["start_seconds"]), b)
    for k, start, end in spans:
        s = scenes[k]
        events.append({"label": s["ambience_label"], "category": "ambience",
                       "description": s["ambience_description"] or s["ambience_label"],
                       "start_seconds": start, "end_seconds": end, "confidence": 0.7, "temporal": "stationary"})
    return events


async def describe_scenes(client: httpx.AsyncClient, video: Path, workdir: Path, duration: float,
                          model: str, hints: str | None, job_id: str | None) -> list[dict]:
    """The context pass: sparse frames in windows of FRAMES_PER_CALL, one structured
    answer per window, folded into scenes that tile the video."""
    (workdir / "context").mkdir(exist_ok=True)      # its own folder: extract_frames globs the folder
    frames = extract_frames(video, workdir / "context", SCENE_FPS)
    if not frames:
        return []
    size = min(FRAMES_PER_CALL, len(frames))
    starts = sorted({min(i, len(frames) - size) for i in range(0, len(frames), size)})
    scenes: list[dict] = []
    for k, i in enumerate(starts, start=1):
        chunk = frames[i:i + size]
        set_progress(job_id, stage="scene context", fraction=(k - 1) / len(starts),
                     detail=f"scenes: step {k} of {len(starts)}", calls_done=k - 1)
        first, last = chunk[0][0], chunk[-1][0]
        text = (f"These {len(chunk)} frames were sampled every {1 / SCENE_FPS:.1f} s from a video of "
                f"{duration:.1f} s; they cover {first:.1f}s to {last:.1f}s and each frame shows its time "
                f"in the top-left corner. Frame times: {', '.join(f'{t:.1f}s' for t, _ in chunk)}.\n"
                f"Describe the scenes in these frames. start_seconds and end_seconds must lie between "
                f"{first:.1f} and {min(duration, last + 1 / SCENE_FPS):.1f}.")
        if hints:
            text += f"\nContext from the editor: {hints}"
        text += "\nReturn JSON matching the schema."
        answer = await ask_vlm(client, [path for _, path in chunk], CONTEXT_SYSTEM_PROMPT, text,
                               CONTEXT_SCHEMA, model)
        found = answer.get("scenes", []) if isinstance(answer, dict) else []
        log.info("context %.1fs-%.1fs: %s", first, last,
                 "; ".join(f"{s.get('place')} {s.get('start_seconds')}-{s.get('end_seconds')}: "
                           f"{', '.join(s.get('expected_sounds') or [])}" for s in found))
        scenes.extend(found)
    return fold_scenes(scenes, duration, 1 / SCENE_FPS)


def scene_hint(scene: dict | None, hints: str | None, checklist: bool = True, level: str = "full") -> str | None:
    """What a chunk call is told about its scene, in front of the editor's hints."""
    if not scene or level == "none":
        return hints
    text = f"Scene: {scene['place']}." if level == "place" else f"Scene: {scene['place']}. {scene['what_happens']}".strip()
    if checklist and scene["expected_sounds"]:
        text += (" Sounds expected somewhere in this scene: " + ", ".join(scene["expected_sounds"])
                 + ". List only those whose source is active in these frames, at the times the frames show.")
    return f"{text} {hints.strip()}" if hints and hints.strip() else text


CONSOLIDATE_SYSTEM_PROMPT = """You are a film sound editor cleaning up a spotting list before the session is
built. Make it the list an editor would lay: one marker per sound source and action.
Aspects of one sound are one marker: the engine, the tyres and the rolling of one
passing car are one "car passes", a person's steps and the rustle of the same
person's clothes while walking are one "footsteps". Sounds that are part of the
general atmosphere of the place (distant traffic, background vehicles, street
bustle, crowd murmur, the hum of a room, general machinery) are carried by the
ambience bed and are removed as separate markers; keep the ambience events
themselves. Keep every sound of a specific visible source doing something (a cat
meowing, a door closing, a robot vacuum running, the steps of a person in frame),
with its times; where two markers describe the same source over the same time,
keep one with the wider span. Never invent a sound that is not in the list. Labels
stay short and name the sound."""


async def consolidate(client: httpx.AsyncClient, events: list[dict], scenes: list[dict], model: str,
                      duration: float) -> list[dict]:
    """One text-only call that turns the chunk pass's list into an editor's list."""
    if len(events) < 2:
        return events
    listed = [{k: e[k] for k in ("label", "category", "description", "start_seconds", "end_seconds") if k in e}
              for e in events]
    places = "; ".join(f"{s['place']} ({s['start_seconds']:.1f}-{s['end_seconds']:.1f} s)" for s in scenes)
    text = (f"The clip is {duration:.1f} s long" + (f"; its scenes: {places}" if places else "") + ".\n"
            f"The spotting list:\n{json.dumps(listed, ensure_ascii=False)}\n"
            "Return the cleaned list as JSON matching the schema, with start_seconds, end_seconds, confidence "
            "and temporal for every event.")
    answer = await ask_vlm(client, [], CONSOLIDATE_SYSTEM_PROMPT, text, EVENT_SCHEMA, model)
    cleaned = answer.get("events", []) if isinstance(answer, dict) else []
    if not cleaned:
        log.warning("consolidation returned nothing; the list stays as it was")
        return events
    log.info("consolidated %d events -> %d: %s", len(events), len(cleaned),
             "; ".join(f"{e.get('label')} {e.get('start_seconds')}-{e.get('end_seconds')}" for e in cleaned))
    return merge_events(cleaned, duration)


# ── Video handling ───────────────────────────────────────────────────────────

def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def extract_frames(video: Path, out_dir: Path, sample_fps: float, width: int | None = None) -> list[tuple[float, Path]]:
    """Sample frames at `sample_fps`, scaled to FRAME_WIDTH. Returns (seconds, file).

    Each frame gets its time burned into the corner. The model aligns events to
    what it can read far better than to a list of times in the prompt: in a
    side-by-side test the stamped frames yielded more events with correct spans.
    """
    pattern = out_dir / "f_%05d.jpg"
    filters = f"fps={sample_fps},scale={width or FRAME_WIDTH}:-2"
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


def detect_cuts(video: Path) -> list[float]:
    """Where the picture cuts, in seconds from the start of the clip.

    Consecutive frames are compared by perceptual hash (PySceneDetect's HashDetector,
    threshold CUT_THRESHOLD): a cut changes the whole image, a fast camera move or the
    stutter of a frame-rate conversion changes far less of the hash. Without the
    library, or when the clip cannot be read, there are no cuts and nothing is split.
    """
    try:
        from scenedetect import HashDetector, SceneManager, open_video
    except ImportError:
        log.warning("cut detection is off: scenedetect is not installed")
        return []
    try:
        manager = SceneManager()
        manager.add_detector(HashDetector(threshold=CUT_THRESHOLD, min_scene_len=6))
        manager.detect_scenes(open_video(str(video)), show_progress=False)
        shots = manager.get_scene_list()
    except Exception as exc:  # a cut we miss costs less than a spotting that fails
        log.warning("cut detection failed for %s: %s", video.name, exc)
        return []
    return [round(float(start.seconds), 3) for start, _ in shots[1:]]


def split_at_cuts(events: list[dict], cuts: list[float]) -> list[dict]:
    """One event per shot: a foley or sound-effect event that runs across a cut is
    cut there too, since the sound made for it must change with the picture. The
    pieces keep label, description and confidence. Ambience, music and dialogue
    stay whole. A piece shorter than CUT_MIN_PIECE_S is dropped, so an event that
    starts a few frames before a cut simply starts at the cut."""
    if not cuts:
        return events
    out: list[dict] = []
    for event in events:
        start, end = float(event["start_seconds"]), float(event["end_seconds"])
        inside = [c for c in cuts if start < c < end]
        if event.get("category") not in CUT_SPLIT_CATEGORIES or not inside:
            out.append(event)
            continue
        edges = [start] + inside + [end]
        pieces = [(a, b) for a, b in zip(edges, edges[1:]) if b - a >= CUT_MIN_PIECE_S]
        if not pieces:
            out.append(event)
            continue
        for a, b in pieces:
            out.append({**event, "start_seconds": round(a, 3), "end_seconds": round(b, 3)})
    out.sort(key=lambda e: e["start_seconds"])
    return out


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
                   duration: float, sample_fps: float, evidence: bool = False, observe: bool = False) -> str:
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
    if evidence:
        text += ("\nFor every event give `evidence`: the frame times, as printed in the corner, in which "
                 "its source is seen doing what makes the sound. An event without such a frame must not be listed.")
    if observe:
        text += ("\nFirst fill `observations`: one entry per frame, with its time and what is visibly happening "
                 "in it (what hands, feet and bodies do, what moves, what changes on screens). Then list the "
                 "events: every event must follow from those observations, and a frame in which nothing "
                 "happens yields no event. Sounds that are typical for the place but not seen are not listed.")
    text += "\nReturn JSON matching the schema."
    return text


async def call_ollama(client: httpx.AsyncClient, frames: list[tuple[float, Path]],
                      hints: str | None, model: str, duration: float, sample_fps: float,
                      evidence: bool = False, observe: bool = False) -> dict:
    schema = EVENT_SCHEMA_EVIDENCE if evidence else EVENT_SCHEMA
    body = {
        "model": model,
        "stream": False,
        "format": with_observations(schema) if observe else schema,
        "options": {"temperature": VLM_TEMPERATURE, "num_ctx": VLM_NUM_CTX, "num_predict": VLM_NUM_PREDICT},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _frame_caption(frames, hints, duration, sample_fps, evidence, observe),
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
                      hints: str | None, model: str, duration: float, sample_fps: float,
                      evidence: bool = False, observe: bool = False) -> dict:
    content: list[dict] = [{"type": "text", "text": _frame_caption(frames, hints, duration, sample_fps, evidence, observe)
                            + "\nSchema: " + json.dumps(EVENT_SCHEMA)}]
    for _, path in frames:
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64," + _b64(path)}})
    body = {
        "model": model,
        "temperature": VLM_TEMPERATURE,
        "max_tokens": VLM_NUM_PREDICT,
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
                and event["start_seconds"] <= last["end_seconds"] + MERGE_GAP_S:
            last["end_seconds"] = max(last["end_seconds"], event["end_seconds"])
            last["confidence"] = max(last["confidence"], event["confidence"])
        else:
            merged.append(dict(event))
    merged = _drop_background(_fold_near_duplicates(merged))
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

    "footsteps" / "footsteps on floor" and "cat" / "cat vocalization" are one
    marker; "cat footsteps" / "cat meow" are two sounds of one subject and stay
    apart, as do "cat walking" / "dog walking". So one label must be a wording of
    the other (its content words a subset), or both must name the same subject
    (first content word) and share a word for the sound, and the spans must touch.
    """
    if a["category"] != b["category"]:
        return False
    if b["start_seconds"] > a["end_seconds"] + MERGE_GAP_S or a["start_seconds"] > b["end_seconds"] + MERGE_GAP_S:
        return False
    if a["category"] == "ambience":
        # Two chunks' wordings of one bed overlap in time; two scenes' beds only touch,
        # and an office room tone must not swallow the wind of the mountain after it.
        shorter = min(a["end_seconds"] - a["start_seconds"], b["end_seconds"] - b["start_seconds"])
        overlap = min(a["end_seconds"], b["end_seconds"]) - max(a["start_seconds"], b["start_seconds"])
        if overlap > 0.5 * shorter:
            return True
    wa, wb = _label_words(a["label"]), _label_words(b["label"])
    if not wa or not wb:
        return False

    def covers(xs: list[str], ys: list[str]) -> bool:
        # every word of xs has a word of ys with the same stem ("meow" / "meowing")
        return all(any(_same_stem(x, y) for y in ys) for x in xs)

    if covers(wa, wb) or covers(wb, wa):
        return True
    if not _same_stem(wa[0], wb[0]):
        return False
    if any(_same_stem(x, y) for x in wa[1:] for y in wb[1:]):
        return True
    # The same subject at the same time, worded as two sounds ("cat breathing" and
    # "cat meowing" over one close-up): an editor lays one clip there.
    shorter = min(a["end_seconds"] - a["start_seconds"], b["end_seconds"] - b["start_seconds"])
    overlap = min(a["end_seconds"], b["end_seconds"]) - max(a["start_seconds"], b["start_seconds"])
    return shorter > 0 and overlap >= 0.5 * shorter


def _same_stem(a: str, b: str) -> bool:
    return a == b or (min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a)))


# Words that describe a presence rather than a sound; a label made of them loses to one that names a sound.
_WEAK_WORDS = frozenset("breathing breath shifting presence movement moving motion activity sounds noise".split())


def _label_score(label: str) -> tuple[int, int]:
    words = _label_words(label)
    return (len([w for w in words if w not in _WEAK_WORDS]), -len(label))


def _drop_background(events: list[dict]) -> list[dict]:
    """A foley or effect worded as background ("distant traffic", "background vehicles")
    while an ambience bed runs is part of that bed, not a marker of its own."""
    beds = [e for e in events if e["category"] == "ambience"]
    out = []
    for e in events:
        words = set(_label_words(e["label"]))
        if e["category"] in CUT_SPLIT_CATEGORIES and words & _BACKGROUND_WORDS and any(
                min(b["end_seconds"], e["end_seconds"]) > max(b["start_seconds"], e["start_seconds"]) for b in beds):
            continue
        out.append(e)
    return out


_BACKGROUND_WORDS = frozenset("distant background ambient general faint far".split())


def _fold_near_duplicates(events: list[dict]) -> list[dict]:
    folded: list[dict] = []
    for event in sorted(events, key=lambda e: (e["start_seconds"], -e["confidence"])):
        for kept in folded:
            if _same_event(kept, event):
                kept["start_seconds"] = min(kept["start_seconds"], event["start_seconds"])
                kept["end_seconds"] = max(kept["end_seconds"], event["end_seconds"])
                kept["confidence"] = max(kept["confidence"], event["confidence"])
                # The more telling wording makes the better marker name: "cat meowing" over
                # "cat breathing and shifting" over "cat", and among equals the shorter.
                if _label_score(event["label"]) > _label_score(kept["label"]):
                    kept["label"] = event["label"]
                break
        else:
            folded.append(dict(event))
    return folded


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Reachability of the model endpoint, and whether the model is present."""
    try:
        async with httpx.AsyncClient(timeout=10, headers=VLM_HEADERS) as client:
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
                  "fps": "timecode frame rate", "hints": "optional free text",
                  "categories": "optional, comma-separated kinds to keep (dialogue, foley, sfx, ambience, music)"},
        "output": "events[] with start/end seconds and timecodes",
        "categories": list(CATEGORIES),
        "model": VLM_MODEL,
        "sample_fps": SAMPLE_FPS,
        "scene_context": SCENE_CONTEXT,
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
    categories: str | None = Form(None),
    scene_context: bool | None = Form(None, description="Run the scene-context pass first (default: the service setting)"),
    scene_checklist: bool | None = Form(None, description="Give the chunk calls the scene's expected sounds (default: the service setting)"),
    evidence: bool | None = Form(None, description="Every event must cite the frames that show it (default: the service setting)"),
    frames_per_call: int | None = Form(None, description="Frames shown to the model per call (default: the service setting)"),
    scene_hint_level: str | None = Form(None, description="full, place or none: how much of the scene a chunk call is told (default: the service setting)"),
    observe: bool | None = Form(None, description="The model writes per-frame observations before the events (default: the service setting)"),
    frame_width: int | None = Form(None, description="Width of the frames sent to the model (default: the service setting)"),
    consolidate_events: bool | None = Form(None, description="Tidy the list to one marker per source and action (default: the service setting)"),
):
    """`categories` (optional, comma-separated: dialogue, foley, sfx, ambience, music):
    the model is told which kinds are wanted and other events are dropped."""
    t0 = time.time()
    wanted = wanted_categories(categories)
    hints = category_hint(wanted, hints)
    model = model or VLM_MODEL
    sample_fps = sample_fps or SAMPLE_FPS
    with_context = SCENE_CONTEXT if scene_context is None else scene_context
    with_checklist = SCENE_CHECKLIST if scene_checklist is None else scene_checklist
    with_evidence = EVENT_EVIDENCE if evidence is None else evidence
    hint_level = (scene_hint_level or SCENE_HINT).strip().lower()
    with_observe = EVENT_OBSERVE if observe is None else observe
    with_consolidate = CONSOLIDATE if consolidate_events is None else consolidate_events
    if hint_level not in ("full", "place", "none"):
        raise HTTPException(400, f"scene_hint_level must be full, place or none, not {hint_level!r}")
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
        frames = extract_frames(video_path, workdir, sample_fps, frame_width)
        if not frames:
            raise HTTPException(400, "no frames could be extracted from the video")
        log.info("spotting %s: %.2fs, %d frames at %.1f fps, model %s",
                 video_path.name, duration, len(frames), sample_fps, model)

        # Every call sees FRAMES_PER_CALL frames if the video has that many; the
        # last chunk is pulled back to full size and overlaps the previous one.
        # Smaller chunks lose events: with 4-5 frames per call the model dropped
        # the first half of a test clip that it found reliably with 8. A call
        # that sees one frame cannot judge duration at all. Overlaps are folded
        # by merge_events. With scene context the chunks are made per scene, so
        # no call straddles two places.
        size = min(max(1, frames_per_call or FRAMES_PER_CALL), len(frames))

        def chunk_starts(lo: int, hi: int) -> list[int]:
            n = min(size, hi - lo)
            return sorted({min(lo + i, hi - n) for i in range(0, hi - lo, n)}) if hi > lo else []

        scenes: list[dict] = []
        raw_events: list[dict] = []
        warnings: list[str] = []
        async with httpx.AsyncClient(timeout=VLM_TIMEOUT_S, headers=VLM_HEADERS) as client:
            if with_context:
                try:
                    scenes = await describe_scenes(client, video_path, workdir, duration, model, hints, job_id)
                except (httpx.HTTPError, ValueError, KeyError) as exc:
                    log.warning("scene context failed, spotting without it: %s", exc)
                    scenes = []
                log.info("%s: %d scene(s): %s", video_path.name, len(scenes),
                         "; ".join(f"{s['place']} {s['start_seconds']:.1f}-{s['end_seconds']:.1f}"
                                   + (" (titles)" if s["is_title"] else "") for s in scenes))
            if scenes:
                calls: list[tuple[int, dict | None]] = []
                for s in scenes:
                    if s["is_title"]:
                        continue
                    lo = next((n for n, (t, _) in enumerate(frames) if t >= s["start_seconds"]), len(frames))
                    hi = next((n for n, (t, _) in enumerate(frames) if t >= s["end_seconds"]), len(frames))
                    calls.extend((i, s) for i in chunk_starts(lo, hi))
            else:
                calls = [(i, None) for i in chunk_starts(0, len(frames))]
            set_progress(job_id, stage="analysing", fraction=0.0,
                         detail=f"step 1 of {len(calls)}: frames 1-{size} of {len(frames)}",
                         frames_total=len(frames), calls_total=max(1, len(calls)), calls_done=0)
            for k, (i, scene) in enumerate(calls, start=1):
                chunk = frames[i:i + size]
                # What is being analysed now; the fraction counts model calls, each of which
                # costs the same, so the bar moves in steps of one call.
                set_progress(job_id, stage="analysing", fraction=(k - 1) / len(calls),
                             detail=f"step {k} of {len(calls)}: frames {i + 1}-{i + len(chunk)} of {len(frames)}",
                             calls_done=k - 1)
                # One call that fails (a runaway generation, a model error, a truncated
                # answer) is retried once and then skipped: the rest of the clip is still
                # spotted, and the response says what was skipped. An endpoint that is
                # down fails the request on the first chunk as before.
                result = None
                for attempt in (1, 2):
                    try:
                        result = await caller(client, chunk, scene_hint(scene, hints, with_checklist, hint_level), model,
                                              duration, sample_fps, with_evidence, with_observe)
                        break
                    except httpx.HTTPStatusError as exc:
                        problem = f"model endpoint returned {exc.response.status_code}: {exc.response.text[:200]}"
                        if exc.response.status_code < 500 or (k == 1 and attempt == 2):
                            raise HTTPException(502, problem)
                    except httpx.HTTPError as exc:
                        problem = f"model endpoint {VLM_URL} failed: {exc}"
                        if k == 1 and attempt == 2:
                            raise HTTPException(502, problem)
                    except (ValueError, KeyError) as exc:
                        problem = f"model answer unusable: {exc}"
                    log.warning("frames %.1fs-%.1fs, attempt %d: %s", chunk[0][0], chunk[-1][0], attempt, problem)
                if result is None:
                    warnings.append(f"frames {chunk[0][0]:.1f}-{chunk[-1][0]:.1f} s skipped: {problem}")
                    set_progress(job_id, stage="analysing", fraction=k / len(calls),
                                 detail=f"step {k} of {len(calls)} skipped", calls_done=k)
                    continue
                found = result.get("events", [])
                if scenes:
                    # The ambience is the scene's, from the context pass, not the chunk's.
                    found = [e for e in found if str(e.get("category", "")).lower() != "ambience"]
                log.info("frames %.1fs-%.1fs: %d events: %s", chunk[0][0], chunk[-1][0], len(found),
                         "; ".join(f"{e.get('label')} {e.get('start_seconds')}-{e.get('end_seconds')}"
                                   for e in found))
                raw_events.extend(found)
                set_progress(job_id, stage="analysing", fraction=k / len(calls),
                             detail=f"step {k} of {len(calls)} done, {len(frames)} frames", calls_done=k)

        set_progress(job_id, stage="done", fraction=1.0, detail=f"{len(calls)} steps, {len(frames)} frames")
        raw_events.extend(scene_ambiences(scenes))
        events = merge_events(raw_events, duration)
        if with_consolidate and events:
            set_progress(job_id, stage="tidying", fraction=1.0, detail="one marker per source and action")
            try:
                async with httpx.AsyncClient(timeout=VLM_TIMEOUT_S, headers=VLM_HEADERS) as client:
                    events = await consolidate(client, events, scenes, model, duration)
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                log.warning("consolidation failed, the list stays as it was: %s", exc)
        if wanted:
            events = [e for e in events if str(e.get("category", "")).lower() in wanted]
        cuts = detect_cuts(video_path) if CUT_DETECTION else []
        if cuts:
            before = len(events)
            events = split_at_cuts(events, cuts)
            log.info("%s: %d cut(s) at %s; %d events -> %d", video_path.name, len(cuts),
                     ", ".join(f"{c:.1f}s" for c in cuts), before, len(events))
        for event in events:
            # A sound seen in one frame is audible at least until the next sample.
            if event["end_seconds"] - event["start_seconds"] < 1 / sample_fps:
                event["end_seconds"] = round(min(duration, event["start_seconds"] + 1 / sample_fps), 3)
            event["start_timecode"] = to_timecode(event["start_seconds"], fps, start_timecode)
            event["end_timecode"] = to_timecode(event["end_seconds"], fps, start_timecode)
        return SpotResponse(events=events, cuts=cuts, warnings=warnings,
                            scenes=[SceneContext(start_seconds=round(s["start_seconds"], 3),
                                                 end_seconds=round(s["end_seconds"], 3), place=s["place"],
                                                 what_happens=s["what_happens"], expected_sounds=s["expected_sounds"])
                                    for s in scenes],
                            video_duration_seconds=round(duration, 3),
                            frames_analysed=len(frames), model=model,
                            seconds_taken=round(time.time() - t0, 2))
    except subprocess.CalledProcessError as exc:
        raise HTTPException(400, f"could not read the video: {exc}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        if job_id and PROGRESS.get(job_id, {}).get("stage") != "done":
            set_progress(job_id, stage="failed")


@app.post("/cuts")
async def cuts(video: UploadFile = File(...)):
    """The picture cuts of a clip, in seconds from its start, for a caller that has
    its own events (markers) and wants them split per shot like /spot does."""
    workdir = Path(tempfile.mkdtemp(prefix="cuts_"))
    try:
        video_path = workdir / (Path(video.filename or "input.mp4").name)
        with video_path.open("wb") as handle:
            shutil.copyfileobj(video.file, handle)
        duration = probe_duration(video_path)
        return {"cuts": detect_cuts(video_path), "video_duration_seconds": round(duration, 3),
                "threshold": CUT_THRESHOLD}
    except subprocess.CalledProcessError as exc:
        raise HTTPException(400, f"could not read the video: {exc}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ── Scenes ───────────────────────────────────────────────────────────────────
# Which consecutive clips of a range belong to one scene (same place, continuous
# time), so that sounds can be budgeted and organised per scene rather than per
# clip. A small model mixes up four pictures at once, so the question goes in two easy
# steps: every clip is described alone from its first and last frame (place, indoors
# or outdoors, time of day), then two neighbours are compared by those descriptions
# only. Indoors next to outdoors is a new scene without asking. Scene names come from
# the descriptions too.

SCENE_SYSTEM_PROMPT = """You are a film editor looking at consecutive shots of a silent picture.
A scene is one place and one continuous stretch of story time. A new camera angle,
a close-up or a reverse shot of the same place at the same moment is the SAME scene.
A different place, or a clear jump in time (day to night, a new day, a different
event), starts a NEW scene."""

SCENE_CLIP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "place": {"type": "string"},
        "setting": {"type": "string", "enum": ["indoors", "outdoors", "unclear"]},
        "time_of_day": {"type": "string", "enum": ["day", "night", "unclear"]},
        "what_is_seen": {"type": "string"},
    },
    "required": ["place", "setting", "time_of_day", "what_is_seen"],
}
SCENE_PAIR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"same_scene": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["same_scene", "reason"],
}
SCENE_NAME_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "description": {"type": "string"}},
    "required": ["name", "description"],
}


class Scene(BaseModel):
    index: int = Field(description="1-based")
    name: str = Field(description="Two to five words: place and, if visible, time of day")
    description: str
    first_clip: int = Field(description="0-based index into the sent clips")
    last_clip: int


class ScenesResponse(BaseModel):
    scenes: list[Scene]
    clip_scenes: list[int] = Field(description="Scene index per sent clip")
    model: str
    seconds_taken: float


def extract_frame_at(video: Path, seconds: float, dst: Path) -> Path | None:
    subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, seconds):.3f}", "-i", str(video),
                    "-frames:v", "1", "-vf", f"scale={FRAME_WIDTH}:-2", "-q:v", "4", str(dst)], check=False)
    return dst if dst.exists() else None


async def ask_vlm(client: httpx.AsyncClient, frames: list[Path], system: str, text: str,
                  schema: dict[str, Any], model: str) -> dict:
    """One structured question about some frames, in either API dialect."""
    if VLM_API == "ollama":
        body = {"model": model, "stream": False, "format": schema,
                "options": {"temperature": VLM_TEMPERATURE, "num_ctx": VLM_NUM_CTX, "num_predict": 2 * VLM_NUM_PREDICT},
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": text, "images": [_b64(p) for p in frames]}]}
        response = await client.post(f"{VLM_URL}/api/chat", json=body)
        response.raise_for_status()
        return json.loads(response.json()["message"]["content"])
    content: list[dict] = [{"type": "text", "text": text + "\nSchema: " + json.dumps(schema)}]
    for path in frames:
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + _b64(path)}})
    body = {"model": model, "temperature": VLM_TEMPERATURE, "response_format": {"type": "json_object"},
            "max_tokens": 2 * VLM_NUM_PREDICT,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}]}
    response = await client.post(f"{VLM_URL}/v1/chat/completions", json=body)
    response.raise_for_status()
    return json.loads(response.json()["choices"][0]["message"]["content"])


@app.post("/scenes", response_model=ScenesResponse)
async def scenes(
    videos: list[UploadFile] = File(..., description="the clips of the range, in timeline order"),
    names: str | None = Form(None, description="JSON list of clip names, same order"),
    model: str | None = Form(None),
    job_id: str | None = Form(None),
):
    """Group consecutive clips into scenes and name each scene."""
    t0 = time.time()
    model = model or VLM_MODEL
    try:
        clip_names = [str(n) for n in json.loads(names)] if names else []
    except ValueError:
        clip_names = []
    workdir = Path(tempfile.mkdtemp(prefix="scenes_"))
    try:
        # Two frames per clip, from just after its start and just before its end.
        clips: list[dict] = []
        for n, upload in enumerate(videos):
            path = workdir / f"clip_{n}.mp4"
            with path.open("wb") as handle:
                shutil.copyfileobj(upload.file, handle)
            duration = probe_duration(path)
            head = extract_frame_at(path, min(0.5, duration / 4), workdir / f"c{n}_head.jpg")
            tail = extract_frame_at(path, max(duration - 0.5, duration * 3 / 4), workdir / f"c{n}_tail.jpg")
            frames = [f for f in (head, tail) if f is not None]
            if not frames:
                raise HTTPException(400, f"no frame could be extracted from clip {n}")
            clips.append({"index": n, "name": clip_names[n] if n < len(clip_names) else f"clip {n + 1}",
                          "head": frames[0], "tail": frames[-1]})
        set_progress(job_id, stage="scenes", fraction=0.0, detail=f"{len(clips)} clips")

        # Step 1: every clip on its own, then step 2: neighbours compared by text.
        calls_total = len(clips) + max(0, len(clips) - 1)
        breaks: list[bool] = []
        async with httpx.AsyncClient(timeout=VLM_TIMEOUT_S, headers=VLM_HEADERS) as client:
            for n, clip in enumerate(clips):
                set_progress(job_id, stage="scenes", fraction=n / max(1, calls_total + 1),
                             detail=f"clip {n + 1} of {len(clips)}: where does it take place?")
                text = ("Frames 1 and 2 are the beginning and the end of one shot. Where does it take place? "
                        "Give the place in a few words (for example 'living room', 'city street', 'forest'), "
                        "whether it is indoors or outdoors, the time of day, and in one sentence what is seen. "
                        "Return JSON.")
                try:
                    answer = await ask_vlm(client, [clip["head"], clip["tail"]], SCENE_SYSTEM_PROMPT, text,
                                           SCENE_CLIP_SCHEMA, model)
                except httpx.HTTPError as exc:
                    raise HTTPException(502, f"model endpoint {VLM_URL} failed: {exc}")
                clip["place"] = str(answer.get("place") or "").strip()[:80]
                clip["setting"] = str(answer.get("setting") or "unclear").lower()
                clip["time"] = str(answer.get("time_of_day") or "unclear").lower()
                clip["seen"] = str(answer.get("what_is_seen") or "").strip()[:200]
                log.info("scenes: %s: %s, %s, %s (%s)", clip["name"], clip["place"], clip["setting"],
                         clip["time"], clip["seen"][:100])

            def described(c: dict) -> str:
                return f"{c['place']} ({c['setting']}, {c['time']}): {c['seen']}"

            for n in range(1, len(clips)):
                set_progress(job_id, stage="scenes", fraction=(len(clips) + n - 1) / max(1, calls_total + 1),
                             detail=f"cut {n} of {len(clips) - 1}: clips {n} and {n + 1} the same scene?")
                prev, cur = clips[n - 1], clips[n]
                known = {"indoors", "outdoors"}
                if prev["setting"] in known and cur["setting"] in known and prev["setting"] != cur["setting"]:
                    same, reason = False, f"{prev['setting']} then {cur['setting']}"
                elif prev["time"] in ("day", "night") and cur["time"] in ("day", "night") and prev["time"] != cur["time"]:
                    same, reason = False, f"{prev['time']} then {cur['time']}"
                else:
                    text = (f"Shot A: {described(prev)}\nShot B, which follows A directly in the cut: "
                            f"{described(cur)}\nIs B the same place as A at the same moment (the same scene), "
                            "or a different place? Only the place matters, not what happens there. Return JSON.")
                    try:
                        answer = await ask_vlm(client, [], SCENE_SYSTEM_PROMPT, text, SCENE_PAIR_SCHEMA, model)
                    except httpx.HTTPError as exc:
                        raise HTTPException(502, f"model endpoint {VLM_URL} failed: {exc}")
                    same, reason = bool(answer.get("same_scene")), str(answer.get("reason", ""))
                log.info("scenes: %s -> %s: %s (%s)", prev["name"], cur["name"],
                         "same" if same else "NEW", reason[:120])
                breaks.append(not same)

            groups: list[list[dict]] = [[clips[0]]] if clips else []
            for n in range(1, len(clips)):
                if breaks[n - 1]:
                    groups.append([clips[n]])
                else:
                    groups[-1].append(clips[n])

            # A name per scene from its clips' descriptions; one clip names itself.
            out: list[Scene] = []
            for k, group in enumerate(groups, start=1):
                set_progress(job_id, stage="scenes", fraction=(calls_total + k / max(1, len(groups))) / (calls_total + 1),
                             detail=f"naming scene {k} of {len(groups)}")
                first = group[0]
                name = first["place"].capitalize() if first["place"] else f"Scene {k}"
                if first["time"] in ("day", "night"):
                    name += ", " + ("daytime" if first["time"] == "day" else "night")
                description = first["seen"]
                if len(group) > 1:
                    text = ("These shots form one scene:\n" + "\n".join(f"- {described(c)}" for c in group)
                            + "\nName the scene in two to five words (the place and, if known, the time of "
                              "day), then describe in one sentence what happens there. Return JSON.")
                    try:
                        answer = await ask_vlm(client, [], SCENE_SYSTEM_PROMPT, text, SCENE_NAME_SCHEMA, model)
                        name = str(answer.get("name") or name).strip()
                        description = str(answer.get("description") or description).strip()
                    except httpx.HTTPError as exc:
                        raise HTTPException(502, f"model endpoint {VLM_URL} failed: {exc}")
                out.append(Scene(index=k, name=name[:60] or f"Scene {k}", description=description,
                                 first_clip=group[0]["index"], last_clip=group[-1]["index"]))

        clip_scenes = [next(s.index for s in out if s.first_clip <= c["index"] <= s.last_clip) for c in clips]
        set_progress(job_id, stage="done", fraction=1.0, detail=f"{len(out)} scenes in {len(clips)} clips")
        log.info("scenes: %d clips -> %d scene(s): %s", len(clips), len(out), "; ".join(s.name for s in out))
        return ScenesResponse(scenes=out, clip_scenes=clip_scenes, model=model,
                              seconds_taken=round(time.time() - t0, 2))
    except subprocess.CalledProcessError as exc:
        raise HTTPException(400, f"could not read a clip: {exc}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        if job_id and PROGRESS.get(job_id, {}).get("stage") != "done":
            set_progress(job_id, stage="failed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("API_HOST", "0.0.0.0"),
                port=int(os.environ.get("API_PORT", "8003")))
