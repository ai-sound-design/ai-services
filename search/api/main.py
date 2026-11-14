"""
Sound Search API

Video-to-sound and text-to-sound retrieval with X-CLIP over any indexed
library: the BBC archive, folders of your own recordings, whatever the indexer
has described (see search/README.md). Sound-to-sound retrieval with CLAP over
the 10 s windows embed_audio.py made of the files on disk (/search/by_audio):
the answer names the file and the second it matches at.

A sound's audio is served from disk when it is there and otherwise fetched
from its media URL on first use (AUDIO_FETCH=on_demand, the default), so a
library like the BBC archive is searchable without downloading it first.
AUDIO_FETCH=local_only restricts results to sounds already on disk, for
machines without internet access.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import tempfile
import threading
import time
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from common import audio, db
from common.device import torch_device
from utils.db_client import DatabaseClient
from utils.xclip_encoder import XCLIPEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATABASE_URL = db.DATABASE_URL
MODEL_NAME = os.getenv("XCLIP_MODEL", "microsoft/xclip-base-patch32-16-frames")
AUDIO_FETCH = os.getenv("AUDIO_FETCH", "on_demand").strip().lower()      # on_demand | local_only
AUDIO_DIR = db.AUDIO_DIR
FETCH_USER_AGENT = "ai-sound-services-search"
DEVICE = torch_device()

encoder: Optional[XCLIPEncoder] = None
clap: Optional[audio.ClapEncoder] = None
clap_lock = threading.Lock()
REFINE_TOP = int(os.getenv("REFINE_TOP", "0"))          # candidates that get the fine search, 0: all returned
QUIET_BELOW_QUERY_DB = float(os.getenv("QUIET_BELOW_QUERY_DB", "20"))  # refine skips stretches this much quieter
# Which audio index the search by sound uses: segments (the adaptive index of
# embed_segments.py), windows (the 10 s windows), or auto: the segments once every
# sound with windows has them, checked every few minutes while the indexer runs.
AUDIO_INDEX = os.getenv("AUDIO_INDEX", "auto").strip().lower()
_segments_state = {"ready": False, "checked": 0.0}
db_client: Optional[DatabaseClient] = None
_fetch_locks: dict[int, asyncio.Lock] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global encoder, db_client
    logger.info("Sound Search API starting (audio fetch: %s)", AUDIO_FETCH)
    logger.info("Loading X-CLIP model %s on %s", MODEL_NAME, DEVICE)
    encoder = XCLIPEncoder(model_name=MODEL_NAME, device=DEVICE)
    global clap
    try:
        clap = audio.ClapEncoder(device=DEVICE)
    except Exception as exc:  # noqa: BLE001  (the video/text search must not depend on it)
        logger.error("CLAP not loaded, search by sound is off: %s", exc)
        clap = None
    db_client = DatabaseClient(DATABASE_URL)
    db_client.set_embedding_column(encoder.embedding_dim)
    for entry in db_client.libraries():
        logger.info("library %-12s %6d sounds, %6d embedded, %6d on disk, %6d audio-indexed",
                    entry["library"], entry["sounds"], entry["embedded"], entry["local_files"],
                    entry.get("audio_indexed", 0))
    if not db_client.libraries():
        logger.warning("The index is empty. Run the indexer (see search/README.md).")
    logger.info("API ready")
    yield
    if db_client:
        db_client.close()


app = FastAPI(title="Sound Search API", version="2.0.0",
              description="X-CLIP based video/text to sound retrieval over indexed libraries",
              lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


# ── Service information ──────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"service": "Sound Search API", "version": "2.0.0", "model": MODEL_NAME,
            "device": DEVICE, "audio_fetch": AUDIO_FETCH, "status": "ready"}


@app.get("/health")
async def health_check():
    try:
        stats = db_client.get_stats()
        windows = db_client.audio_windows()
        segments = db_client.audio_segments()
        return {"status": "ok", "model": encoder.model_name if encoder else MODEL_NAME, "device": DEVICE,
                "database": "connected", "audio_fetch": AUDIO_FETCH,
                "libraries": {s["library"]: {"sounds": s["sounds"], "embedded": s["embedded"],
                                             "local_files": s["local_files"],
                                             "audio_indexed": s.get("audio_indexed", 0)} for s in stats["libraries"]},
                "available_sounds": stats["sounds_with_embeddings"],
                "sounds_with_embeddings": stats["sounds_with_embeddings"],
                # Search by sound: needs the CLAP model and an audio index, either one: the
                # setup page builds only the segments (embed_segments.py), the older windows
                # come from embed_audio.py. The hybrid service reports this as database_match.
                "audio_search": {"available": clap is not None and (windows["windows"] > 0 or segments["segments"] > 0),
                                 "model": clap.model_name if clap else None,
                                 "windows": windows["windows"], "sounds": windows["sounds"],
                                 "index": "segments" if _use_segments() else "windows",
                                 "segments": segments["segments"], "segmented_sounds": segments["sounds"],
                                 "reason": "CLAP model not loaded" if clap is None else
                                           None if windows["windows"] > 0 or segments["segments"] > 0 else
                                           "no audio index yet: the setup page's 'Audio index' step builds it"}}
    except Exception as exc:
        logger.error("Health check failed: %s", exc)
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.get("/libraries")
async def list_libraries():
    """The indexed libraries with their sizes; the plugin's search profile may name one."""
    return {"libraries": db_client.libraries()}


@app.get("/stats")
async def get_statistics():
    return db_client.get_stats()


@app.get("/categories")
async def get_categories(library: Optional[str] = None, limit: int = 50):
    return {"categories": db_client.get_categories(library, limit)}


@app.post("/admin/switch-model")
async def switch_model(model_name: str = Form(...)):
    """Load another X-CLIP variant at runtime (base: 512 dims, large: 768 dims)."""
    global encoder
    try:
        if encoder:
            del encoder
            torch.cuda.empty_cache()
        encoder = XCLIPEncoder(model_name, DEVICE)
        db_client.set_embedding_column(encoder.embedding_dim)
        return {"status": "ok", "model": model_name, "embedding_dim": encoder.embedding_dim, "device": DEVICE}
    except Exception as exc:
        logger.error("Model switch failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Model switch failed: {exc}")


# ── Search ───────────────────────────────────────────────────────────────────

@app.post("/search/sounds")
async def search_sounds(
    video: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    limit: int = Form(5),
    threshold: float = Form(0.0),
    num_frames: int = Form(16),
    text_weight: float = Form(0.6),
    library: Optional[str] = Form(None),
):
    """
    Search by video and/or text.

    library: restrict to one library, or several separated by commas; empty
             means all. With both video and text the query embedding is
             (1 - text_weight) * video + text_weight * text.
    """
    if video is None and not text:
        raise HTTPException(status_code=400, detail="Provide a video, a text, or both")
    libraries = [name.strip() for name in (library or "").split(",") if name.strip()] or None

    try:
        if video and text:
            video_embedding = await encoder.encode_video(video, num_frames=num_frames)
            text_embedding = encoder.encode_text(text)
            query_embedding = (1 - text_weight) * video_embedding + text_weight * text_embedding
            query_type = f"hybrid (video {1 - text_weight:.0%} + text {text_weight:.0%})"
        elif video:
            query_embedding = await encoder.encode_video(video, num_frames=num_frames)
            query_type = "video"
        else:
            query_embedding = encoder.encode_text(text)
            query_type = "text"

        results = db_client.vector_search(query_embedding, limit=limit, threshold=threshold,
                                          libraries=libraries, local_only=AUDIO_FETCH == "local_only")
        logger.info("%d results for %s query%s", len(results), query_type,
                    f" in {libraries}" if libraries else "")
        return {"query_type": query_type, "query": text if text else video.filename,
                "libraries": libraries, "count": len(results), "results": results}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Search failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}")


# ── Search by sound ──────────────────────────────────────────────────────────

def _fused_query(samples: np.ndarray, text: Optional[str], text_weight: float) -> np.ndarray:
    """CLAP embedding of the audio, blended with the text's when a weight is given."""
    query = clap.encode_long_audio(samples)
    if text and text_weight > 0:
        text_vec = clap.encode_text([text])[0]
        query = (1 - text_weight) * query + text_weight * text_vec
        query = query / (np.linalg.norm(query) or 1.0)
    return query


def _envelope(samples: np.ndarray, hop: float = 0.05) -> np.ndarray:
    """Loudness over time: RMS per `hop` seconds, normalised by its mean, so that only the
    shape counts (a steady hum is flat, a machine winding up is a ramp)."""
    size = max(1, int(hop * audio.SAMPLE_RATE))
    count = max(1, len(samples) // size)
    frames = samples[:count * size].reshape(count, size).astype(np.float64)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    return rms / (rms.mean() or 1.0)


def _rms_db(samples: np.ndarray) -> float:
    return float(10.0 * np.log10(np.mean(np.asarray(samples, dtype=np.float64) ** 2) + 1e-12))


def _envelope_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """1 for the same loudness shape, towards 0 the more the shapes differ."""
    n = min(len(a), len(b))
    if n < 2:
        return 0.5
    # Resample both to the same number of points, compare the mean absolute difference.
    ia = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a)
    ib = np.interp(np.linspace(0, len(b) - 1, n), np.arange(len(b)), b)
    return float(1.0 / (1.0 + np.abs(ia - ib).mean()))


def _decode_failed(candidate: dict, exc: Exception) -> None:
    """A file that is not on disk (the prebuilt index, a sound not fetched yet) is no
    warning; one that is there and cannot be read is."""
    path = candidate.get("file_path")
    if path and os.path.exists(path):
        logger.warning("refine: could not decode %s: %s", path, exc)
    else:
        logger.debug("refine: %s is not on disk; the coarse match stands", path)


def _refine(candidate: dict, query: np.ndarray, piece_length: float,
            query_samples: Optional[np.ndarray] = None, envelope_weight: float = 0.0) -> dict:
    """Locate the best `piece_length` seconds inside (and just around) a matched 10 s window.

    The score there is CLAP similarity, blended with the similarity of the loudness
    envelopes when `envelope_weight` is above 0: CLAP hears what a sound is, the envelope
    says how it moves, and a steady hum should not be replaced by a machine winding up."""
    region_start = max(0.0, float(candidate["offset_seconds"]) - 2.0)
    region_length = float(candidate["length_seconds"]) + 4.0
    try:
        region = audio.decode(candidate["file_path"], start=region_start, length=region_length)
    except Exception as exc:  # noqa: BLE001
        _decode_failed(candidate, exc)
        return candidate
    length = min(piece_length, len(region) / audio.SAMPLE_RATE)
    if length <= 0:
        return candidate
    hop = max(0.25, length / 4)
    offsets = [round(i * hop, 3) for i in range(int((len(region) / audio.SAMPLE_RATE - length) // hop) + 1)]
    pieces = [audio.slice_window(region, o, length) for o in offsets]
    # A stretch far quieter than the query (a pause between two miaows) is no match,
    # whatever CLAP makes of near-silence; kept only if the whole region is that quiet
    if query_samples is not None and len(pieces) > 1:
        wanted = _rms_db(query_samples[:int(length * audio.SAMPLE_RATE)])
        loud = [(o, p) for o, p in zip(offsets, pieces) if _rms_db(p) >= wanted - QUIET_BELOW_QUERY_DB]
        if loud:
            offsets, pieces = [o for o, _ in loud], [p for _, p in loud]
    vectors = clap.encode_audio(pieces)
    scores = vectors @ query
    if envelope_weight > 0 and query_samples is not None:
        wanted = _envelope(query_samples[:int(length * audio.SAMPLE_RATE)])
        shapes = np.array([_envelope_similarity(wanted, _envelope(piece)) for piece in pieces])
        scores = (1 - envelope_weight) * scores + envelope_weight * shapes
    best = int(np.argmax(scores))
    refined = dict(candidate)
    refined["window_similarity"] = candidate["similarity"]
    refined["similarity"] = round(float(scores[best]), 4)
    refined["offset_seconds"] = round(region_start + offsets[best], 3)
    refined["length_seconds"] = round(length, 3)
    return refined


def _use_segments() -> bool:
    if AUDIO_INDEX in ("segments", "windows"):
        return AUDIO_INDEX == "segments"
    # Looked at again every half minute until it is ready: a search right after the setup
    # page built the index must not answer from the empty window index for five minutes
    if not _segments_state["ready"] and time.time() - _segments_state["checked"] > 30:
        _segments_state["checked"] = time.time()
        try:
            _segments_state["ready"] = db_client.segments_ready()
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not check the segment index: %s", exc)
        if _segments_state["ready"]:
            logger.info("the segment index covers every sound: search by sound uses it from now on")
    return _segments_state["ready"]


def _refine_fine(candidate: dict, query: np.ndarray, piece_length: float,
                 query_samples: Optional[np.ndarray] = None, envelope_weight: float = 0.0) -> dict:
    """Locate the best `piece_length` seconds of a matched sound. The stored 2 s windows
    pick the best window without a CLAP run (a piece up to 2 s is scored by the window
    it lies in, a longer one by the mean of the windows it covers); the whole sound is
    searched, not only the segment that matched, since a segment's vector is the mean
    of its windows and the best single stretch can lie in a long, even segment that
    ranked lower. Then only that window and half a second around it is decoded, CLAP
    scores the stretches of the piece's length in it, and the loudness envelope blends
    in as in _refine: the precision of the old fine search at a fraction of its cost."""
    fine = db_client.fine_windows(int(candidate["id"]))
    if fine is None:
        return _refine(candidate, query, min(piece_length, audio.WINDOW_SECONDS), query_samples, envelope_weight)
    offsets, levels, vectors, width = fine["offsets"], fine["levels"], fine["vectors"], fine["window_seconds"]
    duration = float(offsets[-1]) + width
    length = min(piece_length, duration)
    places = [i for i, o in enumerate(offsets) if o + min(length, width) <= duration + 0.01]         or list(range(len(offsets)))
    # Pauses are no match, whatever CLAP made of them (as in _refine)
    if query_samples is not None:
        wanted_db = _rms_db(query_samples[:int(min(length, width) * audio.SAMPLE_RATE)])
        places = [i for i in places if levels[i] >= wanted_db - QUIET_BELOW_QUERY_DB] or places
    if length <= width:
        clap_scores = vectors[places] @ query
    else:
        means = []
        for i in places:
            inside = [j for j in range(i, len(offsets)) if offsets[j] + width <= offsets[i] + length + 0.01]
            mean = vectors[inside or [i]].mean(axis=0)
            means.append(mean / (float(np.linalg.norm(mean)) or 1.0))
        clap_scores = np.asarray(means) @ query
    best = places[int(np.argmax(clap_scores))]
    clap_score = float(np.max(clap_scores))
    offset, score = float(offsets[best]), clap_score
    if query_samples is not None and length <= audio.WINDOW_SECONDS:
        # The exact stretch inside the best window (and half a second around it): CLAP on
        # every stretch of the piece's length, the loudness shape blended in
        region_start = max(0.0, offset - 0.5)
        try:
            region = audio.decode(candidate["file_path"], start=region_start, length=max(length, width) + 1.0)
        except Exception as exc:  # noqa: BLE001
            # No file on disk is the normal case with the prebuilt index: the coarse match stands
            _decode_failed(candidate, exc)
            region = None
        if region is not None and len(region) / audio.SAMPLE_RATE >= length:
            hop = 0.1 if length <= 3.0 else 0.25
            steps = [round(k * hop, 3) for k in range(int((len(region) / audio.SAMPLE_RATE - length) / hop) + 1)]
            pieces = [audio.slice_window(region, step, length) for step in steps]
            wanted_db = _rms_db(query_samples[:int(length * audio.SAMPLE_RATE)])
            loud = [k for k, piece in enumerate(pieces) if _rms_db(piece) >= wanted_db - QUIET_BELOW_QUERY_DB]
            if loud:
                steps, pieces = [steps[k] for k in loud], [pieces[k] for k in loud]
            scores = clap.encode_audio(pieces) @ query
            if envelope_weight > 0:
                wanted = _envelope(query_samples[:int(length * audio.SAMPLE_RATE)])
                shapes = np.array([_envelope_similarity(wanted, _envelope(piece)) for piece in pieces])
                scores = (1 - envelope_weight) * scores + envelope_weight * shapes
            k = int(np.argmax(scores))
            offset, score = region_start + steps[k], float(scores[k])
    refined = dict(candidate)
    refined["window_similarity"] = candidate["similarity"]
    refined["similarity"] = round(float(score), 4)
    refined["offset_seconds"] = round(offset, 3)
    refined["length_seconds"] = round(length, 3)
    return refined


@app.post("/search/by_audio")
async def search_by_audio(
    audio_file: UploadFile = File(..., alias="audio"),
    text: Optional[str] = Form(None),
    text_weight: float = Form(0.0),
    limit: int = Form(10),
    library: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    refine: bool = Form(True),
    exclude: Optional[str] = Form(None),
    envelope_weight: float = Form(0.3),
    only: Optional[str] = Form(None),
    refine_top: Optional[int] = Form(None),
    index: Optional[str] = Form(None),
):
    """
    Sounds that sound like the uploaded audio.

    The audio is embedded with CLAP and compared with the segments of the
    adaptive index (2 s windows merged where they sound alike), or with the 10 s
    windows of the indexed files while that index is incomplete (AUDIO_INDEX). With `text` and a `text_weight` above 0 the description is
    blended in (CLAP shares one space for both). `refine` (default on) then
    looks inside each matched window for the best stretch of the query's
    length, so a two-second event is located to the quarter second.
    `exclude`: comma-separated sound ids to leave out. `envelope_weight` (0..1,
    default 0.3) blends the similarity of the loudness envelopes into the refined
    score, so the match also moves like the query. `only`: comma-separated sound
    ids to search inside, every one of them refined: a sketch search finds its
    candidates by description first and lets the loudness shape pick the stretch.
    `refine_top`: how many of the strongest candidates get the fine search (default
    REFINE_TOP, 0: every result, since a caller cuts a result at its offset); the others
    keep the start of their segment or window and its score, marked `refined: false`.
    On the segment index the fine search reads the stored 2 s windows instead of
    running CLAP again, so it costs a short decode per candidate. `index`
    (segments or windows) overrides AUDIO_INDEX for this request.
    Only sounds on disk are indexed, so every result is available immediately.
    """
    if clap is None:
        raise HTTPException(status_code=503, detail="Search by sound is off: the CLAP model is not loaded")
    libraries = [name.strip() for name in (library or "").split(",") if name.strip()] or None
    excluded = [int(x) for x in (exclude or "").split(",") if x.strip().isdigit()] or None
    only_ids = [int(x) for x in (only or "").split(",") if x.strip().isdigit()] or None
    suffix = Path(audio_file.filename or "query.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(await audio_file.read())
        query_path = Path(handle.name)
    try:
        samples = audio.decode(query_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not decode the audio: {exc}")
    finally:
        query_path.unlink(missing_ok=True)
    if len(samples) < audio.SAMPLE_RATE // 10:
        raise HTTPException(status_code=400, detail="The audio is shorter than 0.1 s")
    query_length = len(samples) / audio.SAMPLE_RATE

    def work() -> dict:
        segments = index == "segments" or (index != "windows" and _use_segments())
        search, fine = (db_client.segment_search, _refine_fine) if segments else (db_client.window_search, _refine)
        with clap_lock:
            query = _fused_query(samples, text, text_weight)
            windows = search(query, limit=max(limit * 4, 20) if not only_ids else len(only_ids) * 8,
                             libraries=libraries, category=category, exclude=excluded, only=only_ids)
            best_per_sound: dict[int, dict] = {}
            for window in windows:
                if window["id"] not in best_per_sound:
                    best_per_sound[window["id"]] = window
            candidates = list(best_per_sound.values())[:limit]
            if refine:
                top = len(candidates) if only_ids else min(len(candidates), refine_top or REFINE_TOP or len(candidates))
                length = query_length if segments else min(query_length, audio.WINDOW_SECONDS)
                candidates = [fine(c, query, length, samples, max(0.0, min(1.0, envelope_weight)))
                              for c in candidates[:top]] + candidates[top:]
                for candidate in candidates:
                    candidate["refined"] = "window_similarity" in candidate
                candidates.sort(key=lambda c: c["similarity"], reverse=True)
        for candidate in candidates:
            candidate.pop("file_path", None)
        return {"query_seconds": round(query_length, 3), "text": text, "text_weight": text_weight,
                "libraries": libraries, "index": "segments" if segments else "windows",
                "count": len(candidates), "results": candidates}

    try:
        return await asyncio.to_thread(work)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Search by audio failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search by audio failed: {exc}")


# ── Single sounds and their audio ────────────────────────────────────────────

def _media_type_for(path: Path) -> str:
    return {".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac",
            ".aiff": "audio/aiff", ".aif": "audio/aiff", ".ogg": "audio/ogg",
            ".m4a": "audio/mp4"}.get(path.suffix.lower(), "application/octet-stream")


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": FETCH_USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        temporary.write_bytes(response.read())
    temporary.replace(target)


async def _resolve_audio(sound: dict) -> Path:
    """The sound's file on disk, fetching it from its media URL first if allowed."""
    path = Path(sound["file_path"]) if sound.get("file_path") else None
    if path and path.exists():
        return path
    if AUDIO_FETCH != "on_demand" or not sound.get("media_url"):
        raise HTTPException(status_code=404, detail="Audio is not available locally"
                            + ("" if sound.get("media_url") else " and has no media URL"))

    lock = _fetch_locks.setdefault(sound["id"], asyncio.Lock())
    async with lock:
        if path and path.exists():
            return path
        url = sound["media_url"]
        suffix = Path(url.split("?")[0]).suffix or ".mp3"
        target = AUDIO_DIR / sound["library"] / (Path(str(sound["external_id"])).name + suffix)
        logger.info("fetching sound %d from %s", sound["id"], url)
        try:
            await asyncio.to_thread(_download, url, target)
        except Exception as exc:
            logger.error("fetch of sound %d failed: %s", sound["id"], exc)
            raise HTTPException(status_code=502, detail=f"Could not fetch the audio: {exc}")
        db_client.set_file(sound["id"], str(target))
        return target


def _load_sound(sound_id: int) -> dict:
    sound = db_client.get_sound_by_id(sound_id)
    if not sound:
        raise HTTPException(status_code=404, detail="Sound not found")
    return sound


@app.get("/sounds/{sound_id}")
async def get_sound_metadata(sound_id: int):
    sound = _load_sound(sound_id)
    sound.pop("file_path", None)          # a container path means nothing to the caller
    return sound


@app.get("/sounds/{sound_id}/download")
async def download_sound(sound_id: int):
    sound = _load_sound(sound_id)
    path = await _resolve_audio(sound)
    return FileResponse(path=path, media_type=_media_type_for(path), filename=path.name)


@app.get("/sounds/{sound_id}/snippet")
async def sound_snippet(sound_id: int, start: float = 0.0, length: float = 10.0, fade_ms: int = 20,
                        channels: int = 0):
    """`length` seconds of the sound from `start`, as 48 kHz WAV with short fades:
    what a search by sound points at, ready to drop on a track. `channels` (1 or 2)
    forces a channel count so that pieces of different recordings can share one
    track; 0 keeps the file's own."""
    sound = _load_sound(sound_id)
    path = await _resolve_audio(sound)
    length = max(0.1, min(length, 600.0))
    fade = max(0.0, fade_ms / 1000.0)
    handle, out_name = tempfile.mkstemp(suffix=".wav")
    os.close(handle)
    filters = f"afade=t=in:st=0:d={fade:.3f},afade=t=out:st={max(0.0, length - fade):.3f}:d={fade:.3f}"
    command = ["ffmpeg", "-y", "-v", "error", "-nostdin", "-ss", f"{start:.3f}", "-i", str(path),
               "-t", f"{length:.3f}", "-af", filters, "-ar", "48000"]
    if channels in (1, 2):
        command += ["-ac", str(channels)]
    command += ["-c:a", "pcm_s24le", out_name]
    try:
        await asyncio.to_thread(subprocess.run, command, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        Path(out_name).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Could not cut the snippet: {exc.stderr.decode()[:300]}")
    filename = f"{Path(path).stem}_{start:.1f}s.wav"
    return FileResponse(path=out_name, media_type="audio/wav", filename=filename,
                        background=BackgroundTask(lambda: Path(out_name).unlink(missing_ok=True)))


@app.get("/sounds/{sound_id}/preview")
async def preview_sound(sound_id: int, duration: int = 5):
    """The audio for listening before importing. Currently the whole file."""
    sound = _load_sound(sound_id)
    path = await _resolve_audio(sound)
    return FileResponse(path=path, media_type=_media_type_for(path), filename=f"preview_{path.name}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
