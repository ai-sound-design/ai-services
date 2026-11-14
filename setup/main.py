"""
The setup page of the stack (http://localhost:8020).

One page that shows what is installed and what is still being fetched or
indexed, with progress: the services and whether they have their models, the
vision-language model's download, the BBC metadata, the text embeddings, the
BBC audio files, the audio index, and the folder libraries. The long steps are
the indexer's own scripts, run here as subprocesses; progress is read from the
database, not from their output. Every step can be paused and resumed, and it
carries on where it stopped: the scripts only ever process what is still
missing. "Set everything up" runs the steps in order and, after a restart of
the machine, picks up at the step that was interrupted.

Environment (all with the stack's defaults):
    DATABASE_URL, AUDIO_DIR, LIBRARIES_DIR, BBC_AUDIO_FORMAT, VLM_URL, VLM_MODEL, VLM_API
    SETUP_LOG_DIR     where the steps' output goes (default /data/setup/logs)
    MMAUDIO_URL, GENERATION_URL, SEARCH_URL, SPOTTING_URL, HYBRID_URL
                      the services' addresses (default: their compose names)
    MODELS_DIR, OLLAMA_MODELS_DIR
                      where the models are mounted (default /models, /models-ollama)

Outside Docker (install_mac.sh runs the services natively on a Mac) these point at
localhost and at host folders; the steps run with the Python running this page.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import httpx
import psycopg2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from psycopg2.extras import Json, RealDictCursor

from common import db

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)        # one line per probe otherwise
log = logging.getLogger("setup")

OLLAMA_URL = os.getenv("VLM_URL", "http://ollama:11434").rstrip("/")
# Where the models and the audio live, as the page shows it: a store is a Docker volume
# name or a host folder (see MODELS_STORE in .env); the folders are mounted at fixed paths.
MODELS_STORE = os.getenv("MODELS_STORE", "models")
OLLAMA_MODELS_STORE = os.getenv("OLLAMA_MODELS_STORE", "ollama-models")
AUDIO_HOST_DIR = os.getenv("AUDIO_HOST_DIR", "./data/audio")
LIBRARIES_HOST_DIR = os.getenv("LIBRARIES_HOST_DIR", "./data/libraries")
MODELS_DIR = Path(os.getenv("MODELS_DIR", "/models"))
OLLAMA_DIR = Path(os.getenv("OLLAMA_MODELS_DIR", "/models-ollama"))
VLM_MODEL = os.getenv("VLM_MODEL", "gemma4:e4b-it-qat")
VLM_API = os.getenv("VLM_API", "ollama").strip().lower()
BBC_FORMAT = os.getenv("BBC_AUDIO_FORMAT", "mp3")
BBC_EXPECTED = 33_000                      # records in the archive, for the bar before the harvest is done
LOG_DIR = Path(os.getenv("SETUP_LOG_DIR", "/data/setup/logs"))
DOCKER_SOCKET = Path(os.getenv("DOCKER_SOCKET", "/var/run/docker.sock"))
DATABASE_STORE = os.getenv("DATABASE_STORE", "postgres-data")      # a volume name, or a folder outside Docker
# The indexer's scripts sit beside this file (/app in the image). Not resolved: outside
# Docker the file is a link in a folder laid out like the image, the scripts beside it.
APP_DIR = Path(__file__).parent
AUDIO_SUFFIXES = {".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg", ".m4a", ".caf"}

# The services, as the page lists them: title, compose service, health URL
SERVICES = [
    ("Generation (MMAudio)", "mmaudio-api", os.getenv("MMAUDIO_URL", "http://mmaudio-api:8000").rstrip("/") + "/health"),
    ("Generation gateway", "generation-gateway",
     os.getenv("GENERATION_URL", "http://generation-gateway:8010").rstrip("/") + "/health"),
    ("Sound search", "sound-search-api", os.getenv("SEARCH_URL", "http://sound-search-api:8002").rstrip("/") + "/health"),
    ("Spotting", "spotting-api", os.getenv("SPOTTING_URL", "http://spotting-api:8003").rstrip("/") + "/health"),
    ("Hybrid", "hybrid-api", os.getenv("HYBRID_URL", "http://hybrid-api:8004").rstrip("/") + "/health"),
    ("Ollama (hosts the vision-language model)", "ollama", f"{OLLAMA_URL}/api/tags"),
]

# The steps: the indexer's scripts. Progress comes from the database (see progress()).
STEPS: dict[str, dict[str, Any]] = {
    "bbc_index": {"title": "BBC Sound Effects: prebuilt index",
                  "about": "The archive's records, their text embeddings and the audio index (segment vectors of "
                           "every recording), fetched from Hugging Face (about 6 GB, no audio) and merged into the "
                           "database: searchable by video, prompt and sound in minutes, instead of a 60 GB "
                           "download and hours of indexing. A recording is fetched when it is used. Free for "
                           "personal, educational and research use under the BBC's terms.",
                  "command": [sys.executable, "load_index.py"]},
    "bbc_metadata": {"title": "BBC Sound Effects: metadata",
                     "about": "The archive's records (about 33,000), so it can be searched by video or prompt; "
                              "no audio is downloaded. Free for research use under the BBC's terms.",
                     "command": [sys.executable, "sources/bbc/harvest.py"]},
    "text_embeddings": {"title": "Text embeddings",
                        "about": "One vector per description, for the search by video or prompt.",
                        "command": [sys.executable, "embed.py"]},
    "bbc_archive": {"title": "BBC Sound Effects: existing archive",
                    "about": "The copy of the archive in the folder given at installation (BBC_ARCHIVE_HOST_DIR), read "
                             "only: its files are registered as the sounds' audio, nothing is downloaded or copied.",
                    "command": [sys.executable, "sources/bbc/register.py"]},
    "bbc_audio": {"title": f"BBC Sound Effects: download the audio files ({BBC_FORMAT})",
                  "about": "Optional. The archive on disk (about 60 GB as mp3). Not required for searching, since a "
                           "sound is fetched the first time it is used, but needed for the audio index below; an "
                           "existing archive folder makes this step unnecessary.",
                  "command": [sys.executable, "sources/bbc/download.py", "--format", BBC_FORMAT]},
    "audio_index": {"title": "Audio index",
                    "about": "CLAP vectors of every file on disk, for the search by sound and the Hybrid mode's "
                             "library pieces. Hours for the whole archive; the partial index is used right away.",
                    "command": [sys.executable, "embed_segments.py"]},
}
PIPELINE = ["bbc_index", "bbc_metadata", "text_embeddings", "bbc_archive", "bbc_audio", "audio_index"]
ARCHIVE_DIR = Path(os.getenv("BBC_ARCHIVE_DIR", "/data/archive"))
OWN_SOUNDS_DIR = Path(os.getenv("OWN_SOUNDS_DIR", "/data/own-sounds"))     # the folder given at installation
OWN_SOUNDS_NAME = os.getenv("OWN_SOUNDS_NAME", "own")


def pipeline_order(pipeline: dict) -> list[str]:
    """The steps "Set everything up" runs: the prebuilt index unless the index is to be
    built here (metadata, embeddings, the audio index over the files on disk); an existing
    archive is registered instead of downloaded; the download only when it was asked for."""
    order = ["bbc_index"] if pipeline.get("prebuilt", True) else ["bbc_metadata", "text_embeddings"]
    if _disk_cache["archive_files"]:
        order.append("bbc_archive")
    elif pipeline.get("download", True):
        order.append("bbc_audio")
    return order + ["audio_index"]

app = FastAPI(title="AI Sound Design setup", version="1.0.0")


# ── database ─────────────────────────────────────────────────────────────────

class Store:
    """The jobs' state and the pipeline flag, in two small tables beside the index."""

    def __init__(self) -> None:
        self.conn = None

    def connect(self) -> None:
        for attempt in range(60):
            try:
                self.conn = db.connect()
                self.conn.autocommit = True
                break
            except psycopg2.OperationalError as exc:
                if attempt == 59:
                    raise
                log.info("waiting for the database: %s", str(exc).strip()[:80])
                time.sleep(5)
        db.ensure_schema(self.conn)
        with self.conn.cursor() as cursor:
            cursor.execute("CREATE TABLE IF NOT EXISTS setup_jobs (name TEXT PRIMARY KEY, status TEXT NOT NULL, "
                           "started_at TIMESTAMPTZ, finished_at TIMESTAMPTZ, error TEXT)")
            cursor.execute("CREATE TABLE IF NOT EXISTS setup_state (key TEXT PRIMARY KEY, value JSONB NOT NULL)")
            # A job that was running when the container stopped
            cursor.execute("UPDATE setup_jobs SET status = 'interrupted' WHERE status = 'running'")

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(sql, params)
                return [dict(r) for r in cursor.fetchall()] if cursor.description else []
        except psycopg2.OperationalError:
            self.connect()
            raise

    def jobs(self) -> dict[str, dict]:
        return {r["name"]: r for r in self.query("SELECT * FROM setup_jobs")}

    def set_job(self, name: str, status: str, error: Optional[str] = None) -> None:
        self.query("INSERT INTO setup_jobs (name, status, started_at, finished_at, error) VALUES (%s, %s, "
                   "  CASE WHEN %s = 'running' THEN now() END, CASE WHEN %s IN ('done', 'error') THEN now() END, %s) "
                   "ON CONFLICT (name) DO UPDATE SET status = EXCLUDED.status, error = EXCLUDED.error, "
                   "  started_at = COALESCE(EXCLUDED.started_at, setup_jobs.started_at), "
                   "  finished_at = EXCLUDED.finished_at", (name, status, status, status, error))

    def state(self, key: str, default: Any = None) -> Any:
        rows = self.query("SELECT value FROM setup_state WHERE key = %s", (key,))
        return rows[0]["value"] if rows else default

    def set_state(self, key: str, value: Any) -> None:
        self.query("INSERT INTO setup_state (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                   (key, Json(value)))

    def libraries(self) -> list[dict]:
        rows = self.query("SELECT * FROM library_stats")
        for r in rows:
            r["hours"] = round(float(r["hours"] or 0), 1)
            for k in ("sounds", "embedded", "local_files", "audio_indexed", "segment_indexed", "local_indexed"):
                r[k] = int(r.get(k) or 0)
        return rows

    def segments(self) -> dict[str, Any]:
        rows = self.query("SELECT (SELECT COUNT(*) FROM sound_segments) AS segments, "
                          "(SELECT pg_total_relation_size('sound_segments') + pg_total_relation_size('sound_fine')) AS bytes")
        return {"segments": int(rows[0]["segments"]), "bytes": int(rows[0]["bytes"])} if rows else {"segments": 0, "bytes": 0}


store = Store()


# ── the steps as subprocesses ────────────────────────────────────────────────

class Runner:
    def __init__(self) -> None:
        self.procs: dict[str, subprocess.Popen] = {}
        self.lock = threading.Lock()

    def command_of(self, name: str) -> list[str]:
        if name in STEPS:
            return STEPS[name]["command"]
        if name.startswith("folder:"):
            library = name.split(":", 1)[1]
            path = _disk_cache.get("paths", {}).get(library)
            return [sys.executable, "index_folder.py", library] + (["--path", path] if path else [])
        raise HTTPException(404, f"no such step: {name}")

    def start(self, name: str) -> None:
        command = self.command_of(name)
        with self.lock:
            if name in self.procs and self.procs[name].poll() is None:
                return
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            handle = open(LOG_DIR / f"{name}.log", "ab")
            handle.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(command)}\n".encode())
            self.procs[name] = subprocess.Popen(command, cwd=APP_DIR, stdout=handle, stderr=subprocess.STDOUT)
            store.set_job(name, "running")
            log.info("started %s: %s", name, " ".join(command))

    def pause(self, name: str) -> None:
        with self.lock:
            proc = self.procs.get(name)
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            if proc:
                del self.procs[name]
        store.set_job(name, "paused")
        log.info("paused %s", name)

    def running(self) -> list[str]:
        return [n for n, p in self.procs.items() if p.poll() is None]

    def reap(self) -> None:
        """Record the steps that ended since the last look."""
        with self.lock:
            for name, proc in list(self.procs.items()):
                code = proc.poll()
                if code is None:
                    continue
                del self.procs[name]
                if code == 0:
                    store.set_job(name, "done")
                else:
                    store.set_job(name, "error", f"exit code {code}: {log_tail(name, 1)[-1] if log_tail(name, 1) else ''}")
                log.info("%s ended with code %s", name, code)


runner = Runner()


def log_tail(name: str, lines: int = 3) -> list[str]:
    path = LOG_DIR / f"{name}.log"
    if not path.exists():
        return []
    try:
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - 16_000))
            text = handle.read().decode("utf-8", "replace")
    except OSError:
        return []
    return [line for line in text.splitlines() if line.strip() and not line.startswith("===")][-lines:]


# ── progress of every step, from the database ────────────────────────────────

_disk_cache: dict[str, Any] = {"at": 0.0, "bbc_bytes": 0, "archive_files": 0, "folders": {}, "busy": False}


def kick_disk_refresh() -> None:
    """Sizes and file counts on disk, refreshed in the background every ten minutes: a stat
    per file is slow on a bind mount (minutes for the archive on Windows), and the page
    must not wait for 33,000 of them."""
    if _disk_cache["busy"] or time.time() - _disk_cache["at"] < 600:
        return
    _disk_cache["busy"] = True
    threading.Thread(target=refresh_disk, daemon=True).start()


def refresh_disk() -> None:
    try:
        _scan_disk()
    except Exception as exc:  # noqa: BLE001
        log.warning("disk scan: %s", exc)
    finally:
        _disk_cache["at"] = time.time()
        _disk_cache["busy"] = False


def folder_bytes(folder: Path) -> int:
    total = 0
    if folder.exists():
        for root, _, files in os.walk(folder):
            for name in files:
                try:
                    # lstat: the Hugging Face cache links snapshots to blobs; stat would count both
                    total += os.lstat(os.path.join(root, name)).st_size
                except OSError:
                    pass
    return total


def _scan_disk() -> None:
    _disk_cache["bbc_bytes"] = folder_bytes(db.AUDIO_DIR / "bbc")
    _disk_cache["audio_bytes"] = folder_bytes(db.AUDIO_DIR)
    _disk_cache["libraries_bytes"] = folder_bytes(db.LIBRARIES_DIR)
    _disk_cache["models_bytes"] = folder_bytes(MODELS_DIR)
    _disk_cache["ollama_bytes"] = folder_bytes(OLLAMA_DIR)
    archive = 0
    if ARCHIVE_DIR.is_dir():
        for _, _, files in os.walk(ARCHIVE_DIR):
            archive += sum(1 for name in files if Path(name).suffix.lower() in AUDIO_SUFFIXES)
    _disk_cache["archive_files"] = archive
    folders: dict[str, int] = {}
    paths: dict[str, str] = {}
    if db.LIBRARIES_DIR.exists():
        for folder in os.scandir(db.LIBRARIES_DIR):
            if folder.is_dir() and not folder.name.startswith("."):
                count = sum(1 for p in Path(folder.path).rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)
                if count:
                    folders[folder.name] = count
    if OWN_SOUNDS_DIR.is_dir():
        count = sum(1 for p in OWN_SOUNDS_DIR.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)
        if count:                               # the mount of an unused OWN_SOUNDS_HOST_DIR is an empty folder
            folders[OWN_SOUNDS_NAME] = count
            paths[OWN_SOUNDS_NAME] = str(OWN_SOUNDS_DIR)
    _disk_cache["folders"] = folders
    _disk_cache["paths"] = paths


def progress(libraries: list[dict]) -> list[dict]:
    by_name = {r["library"]: r for r in libraries}
    bbc = by_name.get("bbc", {})
    sounds = sum(r["sounds"] for r in libraries)
    embedded = sum(r["embedded"] for r in libraries)
    local = sum(r["local_files"] for r in libraries)
    local_indexed = sum(r["local_indexed"] for r in libraries)
    jobs = store.jobs()
    seg = store.segments()
    steps = []

    def step(name: str, title: str, about: str, done: int, total: int, unit: str, detail: str, complete: bool) -> None:
        job = jobs.get(name, {})
        status = job.get("status", "idle")
        if status in ("done", "error", "idle", "interrupted", "paused") and complete:
            status = "complete"
        elif status in ("idle", "done") and 0 < done < total:
            # Work done outside this page (the command-line indexer, an archive folder)
            # or a few files that arrived after the step ran: not "not started"
            status = "partial"
        steps.append({"name": name, "title": title, "about": about, "done": done, "total": total, "unit": unit,
                      "detail": detail, "status": status, "error": job.get("error"), "log": log_tail(name),
                      "started_at": str(job.get("started_at") or ""), "finished_at": str(job.get("finished_at") or "")})

    n = bbc.get("sounds", 0)
    index = store.state("bbc_index", {}) or {}
    # Complete once (nearly) every BBC sound has its segments: loaded by load_index.py, or
    # built here before the prebuilt index existed
    bbc_segmented = bbc.get("segment_indexed", 0)
    index_complete = n >= BBC_EXPECTED * 0.95 and bbc_segmented >= n * 0.99
    index_detail = index.get("message") or (f"{bbc_segmented:,} of {n:,} sounds with audio index" if n else "not loaded")
    step("bbc_index", STEPS["bbc_index"]["title"], STEPS["bbc_index"]["about"], int(index.get("done") or 0),
         int(index.get("total") or 0), "bytes" if index.get("phase") in ("download", "import") else "",
         index_detail, index_complete)
    step("bbc_metadata", STEPS["bbc_metadata"]["title"], STEPS["bbc_metadata"]["about"], n, max(BBC_EXPECTED, n),
         "records", f"{n:,} records", n >= BBC_EXPECTED * 0.95 or (jobs.get("bbc_metadata", {}).get("status") == "done" and n > 0))
    step("text_embeddings", STEPS["text_embeddings"]["title"], STEPS["text_embeddings"]["about"], embedded, sounds,
         "sounds", f"{embedded:,} of {sounds:,} sounds", sounds > 0 and embedded >= sounds)
    gb = _disk_cache["bbc_bytes"] / 1e9
    archive = _disk_cache["archive_files"]
    bbc_sounds, bbc_files = bbc.get("sounds", 0), bbc.get("local_files", 0)
    missing = bbc_sounds - bbc_files
    # A few sounds the archive no longer serves do not keep the step from being complete
    have_all = bbc_sounds > 0 and missing <= max(0, int(bbc_sounds * 0.005))
    # A handful missing: sounds the archive no longer serves; many: the audio was not downloaded
    gap = (f", {missing:,} without a file (the archive does not serve every sound)" if 0 < missing <= max(1, int(bbc_sounds * 0.005))
           else f", {missing:,} without a file" if missing > 0 else "")
    if archive:
        step("bbc_archive", STEPS["bbc_archive"]["title"], STEPS["bbc_archive"]["about"], bbc_files, bbc_sounds, "files",
             f"{archive:,} audio files in the archive folder, {bbc_files:,} of {bbc_sounds:,} sounds have a file{gap}",
             have_all or jobs.get("bbc_archive", {}).get("status") == "done")
    step("bbc_audio", STEPS["bbc_audio"]["title"], STEPS["bbc_audio"]["about"], bbc_files, bbc_sounds, "files",
         f"{bbc_files:,} of {bbc_sounds:,} files, {gb:.1f} GB downloaded{gap}"
         + (" (not needed: the archive folder is used)" if archive else ""), have_all)
    hours = sum(r["hours"] for r in libraries if r["local_files"])
    in_index = f"{seg['segments']:,} segments, {seg['bytes'] / 1e9:.1f} GB in the index"
    if local:
        step("audio_index", STEPS["audio_index"]["title"], STEPS["audio_index"]["about"], local_indexed, local, "sounds",
             f"{local_indexed:,} of {local:,} sounds on disk, {in_index}", local_indexed >= local)
    else:
        # Nothing on disk to index: the prebuilt index serves the search by sound, a recording
        # is fetched when it is used; complete once that index is in place (not while it loads)
        step("audio_index", STEPS["audio_index"]["title"], STEPS["audio_index"]["about"], 0, 0, "sounds",
             f"no audio on disk; {in_index}" if seg["segments"] else "no audio on disk, nothing to index yet",
             seg["segments"] > 0 and jobs.get("bbc_index", {}).get("status") != "running")
    for folder, files in sorted(_disk_cache["folders"].items()):
        row = by_name.get(folder, {})
        done = min(row.get("embedded", 0), row.get("segment_indexed", 0))
        where = "the folder given at installation" if folder in _disk_cache.get("paths", {}) else f"data/libraries/{folder}"
        step(f"folder:{folder}", f"Library folder: {folder}",
             f"Your own sounds in {where}, read only: described from a sidecar index.csv, the files' tags or "
             "their names, then embedded and audio-indexed.", done, max(files, row.get("sounds", 0)), "files",
             f"{files:,} audio files, {row.get('sounds', 0):,} indexed, {done:,} with embeddings and audio index",
             files > 0 and row.get("sounds", 0) >= files and done >= row.get("sounds", 0))
    return steps


# ── the services, their models, and Docker's view of them ────────────────────

def where(store: str, project: str) -> str:
    """A store from .env as the page names it: a folder, or the Docker volume with its full name."""
    if "/" in store or "\\" in store:
        return f"folder {store}"
    return f"Docker volume {project + '_' if project else ''}{store}"


def storage(project: str) -> list[dict]:
    """Where the big things live and how much room they take; sizes from the last disk scan."""
    rows = [
        {"title": "MMAudio and search models", "where": where(MODELS_STORE, project), "bytes": _disk_cache.get("models_bytes")},
        {"title": "Vision-language model", "where": where(OLLAMA_MODELS_STORE, project), "bytes": _disk_cache.get("ollama_bytes")},
        {"title": "Sound index (database)", "where": where(DATABASE_STORE, project), "bytes": None},
        {"title": "Audio (fetched and downloaded)", "where": f"folder {AUDIO_HOST_DIR}", "bytes": _disk_cache.get("audio_bytes")},
        {"title": "Libraries", "where": f"folder {LIBRARIES_HOST_DIR}", "bytes": _disk_cache.get("libraries_bytes")},
    ]
    try:
        with db.connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT pg_database_size(current_database())")
            rows[2]["bytes"] = int(cursor.fetchone()[0])
    except Exception as exc:  # noqa: BLE001
        log.debug("database size: %s", exc)
    return rows


def docker_containers() -> dict[str, dict]:
    """State and last log line of the stack's containers, by compose service; {} without the socket."""
    if not DOCKER_SOCKET.is_socket():
        return {}
    try:
        with httpx.Client(transport=httpx.HTTPTransport(uds=str(DOCKER_SOCKET)), base_url="http://docker", timeout=5) as client:
            me = client.get(f"/containers/{socket.gethostname()}/json").json()
            project = me["Config"]["Labels"].get("com.docker.compose.project")
            _disk_cache["project"] = project or ""
            if not project:
                return {}
            rows = client.get("/containers/json", params={"all": "true", "filters": json.dumps(
                {"label": [f"com.docker.compose.project={project}"]})}).json()
            out = {}
            for row in rows:
                service = row["Labels"].get("com.docker.compose.service")
                if not service:
                    continue
                last = ""
                try:
                    raw = client.get(f"/containers/{row['Id']}/logs", params={"stdout": "true", "stderr": "true", "tail": "1"}).content
                    last = demux(raw).strip().splitlines()[-1] if raw else ""
                except Exception:  # noqa: BLE001
                    pass
                out[service] = {"state": row["State"], "status": row["Status"], "last_log": last[-200:]}
            return out
    except Exception as exc:  # noqa: BLE001
        log.debug("docker: %s", exc)
        return {}


def demux(raw: bytes) -> str:
    """Docker's log stream: 8-byte frame headers unless the container has a TTY."""
    if len(raw) < 8 or raw[0] not in (0, 1, 2) or raw[1:4] != b"\x00\x00\x00":
        return raw.decode("utf-8", "replace")
    out, i = [], 0
    while i + 8 <= len(raw):
        size = int.from_bytes(raw[i + 4:i + 8], "big")
        out.append(raw[i + 8:i + 8 + size].decode("utf-8", "replace"))
        i += 8 + size
    return "".join(out)


async def probe(url: str) -> tuple[bool, Any]:
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            answer = await client.get(url)
            return answer.status_code == 200, (answer.json() if answer.status_code == 200 else None)
    except Exception:  # noqa: BLE001
        return False, None


vlm: dict[str, Any] = {"task": None, "status": "", "completed": 0, "total": 0, "error": None}


async def pull_vlm() -> None:
    """Ollama fetches the model with progress; a pull the Ollama service started itself shares
    the same download, so this only adds the numbers."""
    vlm.update(status="starting", error=None)
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", f"{OLLAMA_URL}/api/pull", json={"name": VLM_MODEL, "stream": True}) as answer:
                async for line in answer.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    vlm["status"] = data.get("status", "")
                    if data.get("total"):
                        vlm["completed"], vlm["total"] = int(data.get("completed", 0)), int(data["total"])
                    if data.get("error"):
                        vlm["error"] = data["error"]
    except Exception as exc:  # noqa: BLE001
        vlm["error"] = str(exc)[:200]
    finally:
        vlm["task"] = None


@app.get("/status")
async def status() -> dict:
    containers, libraries = await asyncio.gather(asyncio.to_thread(docker_containers), asyncio.to_thread(store.libraries))
    kick_disk_refresh()
    healths = await asyncio.gather(*(probe(url) for _, _, url in SERVICES))
    services, models, search = [], {}, {}
    for (title, service, _), (ok, body) in zip(SERVICES, healths):
        box = containers.get(service, {})
        if ok:
            state, detail = "ok", "running"
        elif box.get("state") == "running":
            state, detail = "starting", box.get("last_log") or box.get("status", "")
        elif box:
            state, detail = "down", box.get("status", box.get("state", ""))
        else:
            state, detail = "down", "not reachable"
        if ok and service == "sound-search-api" and isinstance(body, dict):
            a = body.get("audio_search", {})
            search = {"sounds": int(body.get("available_sounds", 0)), "by_sound": bool(a.get("available")),
                      "index": a.get("index", "-"), "segments": int(a.get("segments", 0))}
            models["clap"] = {"title": f"Search models ({a.get('model', 'CLAP')}, {body.get('model', 'X-CLIP')})",
                              "ready": True, "detail": "ready"}
        if ok and service == "ollama" and isinstance(body, dict):
            names = [m.get("name", "") for m in body.get("models", [])]
            present = VLM_MODEL in names or f"{VLM_MODEL}:latest" in names
            models["vlm"] = {"title": f"Vision-language model ({VLM_MODEL})", "ready": present,
                             "detail": "ready" if present else (vlm["status"] or "waiting for the download"),
                             "completed": vlm["completed"], "total": vlm["total"], "error": vlm["error"]}
            if not present and VLM_API == "ollama" and vlm["task"] is None:
                vlm["task"] = asyncio.create_task(pull_vlm())
        if service == "mmaudio-api":
            models["mmaudio"] = {"title": "MMAudio weights (about 10 GB)", "ready": ok,
                                 "detail": "ready" if ok else (box.get("last_log") or "the container fetches them at its first start")}
        services.append({"title": title, "service": service, "state": state, "detail": detail})
    if "vlm" not in models and VLM_API != "ollama":
        models["vlm"] = {"title": "Vision-language model", "ready": True, "detail": f"external: {OLLAMA_URL}"}
    if "clap" not in models:
        models["clap"] = {"title": "Search models (CLAP, X-CLIP)", "ready": False, "detail": "loaded when the search service is up"}
    pipeline = store.state("pipeline", {"active": False, "download": True})
    # An indexer started from the command line (docker compose run indexer ...) works on the
    # same tables; the page says so instead of starting the same step a second time
    external = containers.get("indexer", {}).get("state") == "running"
    return {"services": services, "models": models, "steps": await asyncio.to_thread(progress, libraries),
            "libraries": libraries, "pipeline": pipeline, "running": runner.running(), "search": search,
            "storage": await asyncio.to_thread(storage, _disk_cache.get("project", "")),
            "external_indexer": containers.get("indexer", {}).get("last_log", "") if external else None,
            "disk_scanning": _disk_cache["busy"] and _disk_cache["at"] == 0.0, "time": time.strftime("%H:%M:%S")}


# ── control ──────────────────────────────────────────────────────────────────

@app.get("/")
async def page():
    return FileResponse(Path(__file__).with_name("page.html"))


@app.post("/jobs/{name}/start")
async def start_job(name: str) -> dict:
    runner.start(name)
    return {"ok": True}


@app.post("/jobs/{name}/pause")
async def pause_job(name: str) -> dict:
    runner.pause(name)
    if name in PIPELINE:
        store.set_state("pipeline", {**store.state("pipeline", {}), "active": False})
    return {"ok": True}


@app.get("/jobs/{name}/log")
async def job_log(name: str) -> dict:
    return {"name": name, "lines": log_tail(name, 200)}


@app.post("/pipeline/start")
async def pipeline_start(body: dict) -> dict:
    # "ran": the steps this run has started, each at most once (see supervise)
    store.set_state("pipeline", {"active": True, "download": bool(body.get("download", True)),
                                 "prebuilt": bool(body.get("prebuilt", True)), "ran": []})
    return {"ok": True}


@app.post("/pipeline/stop")
async def pipeline_stop() -> dict:
    store.set_state("pipeline", {**store.state("pipeline", {}), "active": False})
    for name in runner.running():
        if name in PIPELINE:
            runner.pause(name)
    return {"ok": True}


@app.post("/vlm/pull")
async def vlm_pull() -> dict:
    if vlm["task"] is None:
        vlm["task"] = asyncio.create_task(pull_vlm())
    return {"ok": True}


async def supervise() -> None:
    """Every few seconds: note the steps that ended, and keep the pipeline moving."""
    while True:
        try:
            runner.reap()
            pipeline = store.state("pipeline", {"active": False})
            if pipeline.get("active") and not runner.running():
                steps = {s["name"]: s for s in progress(store.libraries())}
                ran = pipeline.get("ran") or []
                for name in pipeline_order(pipeline):
                    # Each step once per run: one that ran through with a few items left (files
                    # the archive no longer serves, files ffmpeg cannot read) shows as "partial"
                    # and must not be started again and again
                    if name in ran or name not in steps or steps[name]["status"] in ("complete", "done", "error"):
                        continue
                    runner.start(name)
                    store.set_state("pipeline", {**pipeline, "ran": ran + [name]})
                    break
                else:
                    store.set_state("pipeline", {**pipeline, "active": False})
                    log.info("every step of the pipeline is complete")
        except Exception as exc:  # noqa: BLE001
            log.warning("supervise: %s", exc)
        await asyncio.sleep(3)


@app.on_event("startup")
async def startup() -> None:
    await asyncio.to_thread(store.connect)
    pipeline = await asyncio.to_thread(store.state, "pipeline")
    if pipeline and pipeline.get("ran"):
        # After a restart every step of a running pipeline gets one more go: the one that
        # was interrupted carries on where it stopped
        await asyncio.to_thread(store.set_state, "pipeline", {**pipeline, "ran": []})
    asyncio.create_task(supervise())
    log.info("setup page ready on port 8020")
