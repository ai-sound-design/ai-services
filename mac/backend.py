#!/usr/bin/env python3
"""
The backend on a Mac, without Docker.

Docker on macOS cannot reach the Apple GPU, so the services that run models
(MMAudio, the search, the indexer) would run on the CPU inside the Docker VM. Here
every service runs natively instead, each in a virtualenv of its own with the
versions its Dockerfile installs, from the same source files: the same ports, the
same .env, the same setup page. The models run on the Apple GPU (MPS).

    python3 mac/backend.py install     # everything; install_mac.sh runs it after the Homebrew part
    python3 mac/backend.py prepare     # only the virtualenvs, the MMAudio source, the folders
    python3 mac/backend.py start [service ...]   # as launchd agents (they also start at login)
    python3 mac/backend.py stop [service ...]
    python3 mac/backend.py restart [service ...]
    python3 mac/backend.py status
    python3 mac/backend.py logs <service> [lines]
    python3 mac/backend.py uninstall   # removes the launchd agents; data and .mac/ stay
    python3 mac/backend.py database [--audio]   # fills the sound index, showing its progress

Services: postgres, ollama, mmaudio, gateway, search, spotting, hybrid, setup, webui.

The web UI (https://github.com/ai-sound-design/ai-sound-webui, the plugin's four modes
in the browser, no Pro Tools needed) is installed alongside: its source is fetched
into .mac/src, the frontend built with Node, and its session server serves it on
port 8030 (WEBUI_PORT) and forwards /svc/* to the services. MAC_WEBUI=0 leaves it out.

Where things live (all from .env, as for Docker):
    the data folder        AUDIO_HOST_DIR/.. (default ./data): audio, libraries, the
                           database (postgres/), the models (models/) and the
                           hybrid service's output (hybrid/)
    the models             MODELS_STORE when it names a folder, else <data>/models
    the Ollama models      OLLAMA_MODELS_STORE when it names a folder, else Ollama's
                           own ~/.ollama/models
    everything installed   .mac/ in this repository: virtualenvs, the MMAudio source,
                           the services' folders laid out as in their images, logs

A few settings only mean something here (put them in .env to change them):
    MAC_BIND_HOST   the address the services listen on (default 127.0.0.1: this Mac only;
                    0.0.0.0 makes them reachable from the network, as Docker does)
    OLLAMA_PORT     the port of the Ollama this script runs (default 11435, beside a
                    separately installed Ollama on 11434)
    WEBUI_SESSIONS_DIRS  the folders with Pro Tools sessions the web UI shows (several,
                    separated by ";"; default <data>/sessions)
    WEBUI_REPO, WEBUI_REF  where the web UI comes from (default its GitHub main)
"""
from __future__ import annotations

import hashlib
import json
import os
import plistlib
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNTIME = REPO / ".mac"
VENVS = RUNTIME / "venv"
APPS = RUNTIME / "app"
LOGS = RUNTIME / "logs"
BREW = Path("/opt/homebrew")
PYTHON = BREW / "opt/python@3.12/bin/python3.12"
PG_BIN = BREW / "opt/postgresql@17/bin"
LABEL = "com.ai-sound-design.backend"
# Run by the installer app (github.com/ai-sound-design/installer): it opens what it wants itself, and reads
# the database's progress from @@ lines
GUI = os.environ.get("AISD_GUI") == "1"
AGENTS = Path.home() / "Library/LaunchAgents"
ORDER = ["postgres", "ollama", "mmaudio", "gateway", "search", "spotting", "hybrid", "setup", "webui"]
NODE = BREW / "bin/node"
NPM = BREW / "bin/npm"
WEBUI_SRC = RUNTIME / "src/ai-sound-webui"

# The fonts ffmpeg burns the frame time with (spotting); a path without spaces is linked to one
STAMP_FONTS = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
               "/Library/Fonts/Arial Bold.ttf"]


# ── configuration ────────────────────────────────────────────────────────────

def read_env() -> dict[str, str]:
    """KEY=value lines of .env (or .env.example before the first install)."""
    path = REPO / ".env" if (REPO / ".env").exists() else REPO / ".env.example"
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def folder(value: str) -> Path:
    path = Path(os.path.expanduser(value))
    return path if path.is_absolute() else (REPO / path).resolve()


def is_folder(value: str) -> bool:
    return "/" in value or "\\" in value


class Config:
    def __init__(self) -> None:
        env = self.env = read_env()
        get = lambda key, default: env.get(key) or default  # noqa: E731  (empty counts as unset)
        audio = folder(get("AUDIO_HOST_DIR", "./data/audio"))
        self.data = audio.parent
        self.audio = audio
        self.libraries = folder(get("LIBRARIES_HOST_DIR", str(self.data / "libraries")))
        self.archive = folder(get("BBC_ARCHIVE_HOST_DIR", str(self.data / "archive")))
        self.own_sounds = folder(get("OWN_SOUNDS_HOST_DIR", str(self.data / "own-sounds")))
        self.models = folder(env["MODELS_STORE"]) if is_folder(env.get("MODELS_STORE", "")) else self.data / "models"
        store = env.get("OLLAMA_MODELS_STORE", "")
        self.ollama_models = folder(store) if is_folder(store) else Path.home() / ".ollama/models"
        self.ollama_models_custom = is_folder(store)
        self.pgdata = folder(get("POSTGRES_DATA_DIR", str(self.data / "postgres")))
        self.hybrid_output = self.data / "hybrid"
        self.webui = get("MAC_WEBUI", "1") != "0"
        self.webui_repo = get("WEBUI_REPO", "https://github.com/ai-sound-design/ai-sound-webui.git")
        self.webui_ref = get("WEBUI_REF", "main")
        self.webui_sessions = [folder(p) for p in get("WEBUI_SESSIONS_DIRS", str(self.data / "sessions")).split(";") if p.strip()]
        self.webui_cache = self.data / "webui-cache"
        self.host = get("MAC_BIND_HOST", "127.0.0.1")
        self.ports = {"mmaudio": int(get("MMAUDIO_PORT", "8000")), "gateway": int(get("GENERATION_PORT", "8010")),
                      "search": int(get("SOUND_SEARCH_PORT", "8002")), "spotting": int(get("SPOTTING_PORT", "8003")),
                      "hybrid": int(get("HYBRID_PORT", "8004")), "setup": int(get("SETUP_PORT", "8020")),
                      "postgres": int(get("POSTGRES_PORT", "5432")), "ollama": int(get("OLLAMA_PORT", "11435")),
                      "webui": int(get("WEBUI_PORT", "8030"))}
        self.db_name, self.db_user = get("POSTGRES_DB", "sounds"), get("POSTGRES_USER", "sound_user")
        self.db_password = get("POSTGRES_PASSWORD", "change_me")
        self.vlm_model = get("VLM_MODEL", "gemma4:e4b-it-qat")
        self.vlm_api = get("VLM_API", "ollama").lower()
        # The Docker defaults name containers; here the bundled Ollama is this script's own
        self.bundled_ollama = get("BUNDLED_OLLAMA", "1") != "0"
        vlm_url = get("VLM_URL", "http://ollama:11434")
        if self.bundled_ollama and re.match(r"^https?://ollama(:\d+)?/?$", vlm_url):
            vlm_url = f"http://127.0.0.1:{self.ports['ollama']}"
        self.vlm_url = vlm_url.replace("host.docker.internal", "127.0.0.1")

    def url(self, service: str) -> str:
        return f"http://127.0.0.1:{self.ports[service]}"

    @property
    def database_url(self) -> str:
        return f"postgresql://{self.db_user}:{self.db_password}@127.0.0.1:{self.ports['postgres']}/{self.db_name}"


# ── what each service is ─────────────────────────────────────────────────────

def base_env(cfg: Config) -> dict[str, str]:
    env = {
        "HOME": str(Path.home()), "USER": os.environ.get("USER", ""), "LANG": "en_US.UTF-8", "LC_ALL": "en_US.UTF-8",
        "PATH": f"{BREW}/bin:{PG_BIN}:/usr/bin:/bin:/usr/sbin:/sbin",
        "PYTHONUNBUFFERED": "1", "LOG_LEVEL": cfg.env.get("LOG_LEVEL", "info"),
        "HF_HOME": str(cfg.models / "huggingface"), "TORCH_HOME": str(cfg.models / "torch"),
        # Operators MPS lacks run on the CPU instead of failing the request
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "FORCE_DEVICE": cfg.env.get("FORCE_DEVICE") or "auto",
    }
    return env


def search_env(cfg: Config) -> dict[str, str]:
    """What the search service, the indexer and the setup page share (as in docker-compose.yml)."""
    e = cfg.env
    return {"DATABASE_URL": cfg.database_url, "AUDIO_DIR": str(cfg.audio), "LIBRARIES_DIR": str(cfg.libraries),
            "BBC_ARCHIVE_DIR": str(cfg.archive), "OWN_SOUNDS_DIR": str(cfg.own_sounds),
            "OWN_SOUNDS_NAME": e.get("OWN_SOUNDS_NAME") or "own", "BBC_AUDIO_FORMAT": e.get("BBC_AUDIO_FORMAT") or "mp3",
            "CLAP_MODEL": e.get("CLAP_MODEL") or "laion/clap-htsat-fused", "AUDIO_FETCH": e.get("AUDIO_FETCH") or "on_demand",
            **{k: e[k] for k in ("AUDIO_INDEX", "SEGMENT_THRESHOLD", "REFINE_TOP", "XCLIP_MODEL", "CLAP_CPU_FEATURES") if e.get(k)}}


def uvicorn(venv: str, cfg: Config, service: str, module: str = "main:app") -> list[str]:
    return [str(VENVS / venv / "bin/python"), "-m", "uvicorn", module, "--host", cfg.host,
            "--port", str(cfg.ports[service]), "--log-level", "warning" if service == "setup" else "info"]


def service_spec(name: str, cfg: Config) -> tuple[list[str], Path, dict[str, str]]:
    """argv, working directory and environment of one service."""
    env, e = base_env(cfg), cfg.env
    if name == "postgres":
        return ([str(PG_BIN / "postgres"), "-D", str(cfg.pgdata), "-c", "listen_addresses=localhost",
                 "-c", f"port={cfg.ports['postgres']}"], cfg.pgdata, env)
    if name == "ollama":
        env.update(OLLAMA_HOST=f"127.0.0.1:{cfg.ports['ollama']}", OLLAMA_KEEP_ALIVE=e.get("OLLAMA_KEEP_ALIVE") or "1h")
        if cfg.ollama_models_custom:
            env["OLLAMA_MODELS"] = str(cfg.ollama_models)
        return [str(BREW / "bin/ollama"), "serve"], Path.home(), env
    if name == "mmaudio":
        env.update(API_HOST=cfg.host, API_PORT=str(cfg.ports["mmaudio"]), MMAUDIO_PATH=str(RUNTIME / "src/MMAudio"),
                   MMAUDIO_MODEL=e.get("MMAUDIO_MODEL") or "large_44k_v2",
                   MMAUDIO_PREFETCH=e.get("MMAUDIO_PREFETCH") or "1",
                   FORCE_DTYPE=e.get("FORCE_DTYPE") or "float32",
                   VIDEO_CACHE_MAX_GB=e.get("VIDEO_CACHE_MAX_GB") or "8",
                   VIDEO_CACHE_TTL_MIN=e.get("VIDEO_CACHE_TTL_MIN") or "60",
                   ENABLE_TRACEMALLOC=e.get("ENABLE_TRACEMALLOC") or "false")
        return [str(VENVS / "mmaudio/bin/python"), "main.py"], APPS / "mmaudio/api", env
    if name == "gateway":
        env.update(GENERATION_MODEL_URL=cfg.url("mmaudio"), GATEWAY_MAX_SECONDS=e.get("GATEWAY_MAX_SECONDS") or "600",
                   GATEWAY_OVERLAP_SECONDS=e.get("GATEWAY_OVERLAP_SECONDS") or "1.0",
                   GATEWAY_WORK_DIR=str(RUNTIME / "work/gateway"), API_PORT=str(cfg.ports["gateway"]))
        return uvicorn("gateway", cfg, "gateway"), REPO / "generation/gateway", env
    if name == "search":
        env.update(search_env(cfg), PYTHONPATH=str(APPS / "search"))
        return uvicorn("search", cfg, "search"), APPS / "search", env
    if name == "spotting":
        # The compose file's names for the spotting settings (SPOTTING_* in .env)
        renamed = {"SPOTTING_SAMPLE_FPS": "SAMPLE_FPS", "SPOTTING_FRAMES_PER_CALL": "FRAMES_PER_CALL",
                   "SPOTTING_CUT_DETECTION": "CUT_DETECTION", "SPOTTING_CUT_THRESHOLD": "CUT_THRESHOLD",
                   "SPOTTING_CUT_MIN_PIECE_S": "CUT_MIN_PIECE_S", "SPOTTING_SCENE_CONTEXT": "SCENE_CONTEXT",
                   "SPOTTING_SCENE_FPS": "SCENE_FPS", "SPOTTING_SCENE_CHECKLIST": "SCENE_CHECKLIST",
                   "SPOTTING_EVENT_EVIDENCE": "EVENT_EVIDENCE", "SPOTTING_EVENT_OBSERVE": "EVENT_OBSERVE",
                   "SPOTTING_SCENE_HINT": "SCENE_HINT", "SPOTTING_MIN_SCENE_S": "MIN_SCENE_S",
                   "SPOTTING_MERGE_GAP_S": "MERGE_GAP_S", "SPOTTING_VLM_NUM_PREDICT": "VLM_NUM_PREDICT"}
        env.update({new: e[old] for old, new in renamed.items() if e.get(old)})
        env.update({k: e[k] for k in ("VLM_NUM_CTX", "FRAME_STAMP", "VLM_TEMPERATURE", "VLM_API_KEY") if e.get(k)})
        # ffmpeg with drawtext (the frame time burnt into each frame) comes first: Homebrew's
        # ffmpeg is built without freetype
        env.update(VLM_URL=cfg.vlm_url, VLM_MODEL=cfg.vlm_model, VLM_API=cfg.vlm_api,
                   STAMP_FONT=str(RUNTIME / "fonts/stamp.ttf"), PATH=f"{RUNTIME / 'bin/spotting'}:{env['PATH']}")
        return uvicorn("spotting", cfg, "spotting"), REPO / "spotting/api", env
    if name == "hybrid":
        env.update(GENERATION_URL=cfg.url("gateway"), SPOTTING_URL=cfg.url("spotting"), SEARCH_URL=cfg.url("search"),
                   CANDIDATES_BY_TEXT=e.get("HYBRID_CANDIDATES_BY_TEXT") or "20", OUTPUT_DIR=str(cfg.hybrid_output),
                   API_PORT=str(cfg.ports["hybrid"]))
        return uvicorn("hybrid", cfg, "hybrid"), REPO / "hybrid/api", env
    if name == "setup":
        env.update(search_env(cfg), PYTHONPATH=str(APPS / "setup"),
                   MMAUDIO_URL=cfg.url("mmaudio"), GENERATION_URL=cfg.url("gateway"), SEARCH_URL=cfg.url("search"),
                   SPOTTING_URL=cfg.url("spotting"), HYBRID_URL=cfg.url("hybrid"),
                   VLM_URL=cfg.vlm_url, VLM_API=cfg.vlm_api, VLM_MODEL=cfg.vlm_model,
                   MODELS_DIR=str(cfg.models), OLLAMA_MODELS_DIR=str(cfg.ollama_models),
                   MODELS_STORE=str(cfg.models), OLLAMA_MODELS_STORE=str(cfg.ollama_models),
                   DATABASE_STORE=str(cfg.pgdata), AUDIO_HOST_DIR=str(cfg.audio), LIBRARIES_HOST_DIR=str(cfg.libraries),
                   SETUP_LOG_DIR=str(LOGS / "setup"), DOCKER_SOCKET=str(RUNTIME / "no-docker.sock"))
        return uvicorn("search", cfg, "setup"), APPS / "setup", env
    if name == "webui":
        # The session server of the web UI: serves the built frontend (dist/), the Pro Tools
        # sessions in WEBUI_SESSIONS_DIRS, and forwards /svc/* to the services
        env.update(STATIC_DIR=str(WEBUI_SRC / "dist"), CACHE_DIR=str(cfg.webui_cache),
                   SESSIONS_DIRS=";".join(str(p) for p in cfg.webui_sessions),
                   GENERATION_URL=cfg.url("gateway"), SEARCH_URL=cfg.url("search"),
                   SPOTTING_URL=cfg.url("spotting"), HYBRID_URL=cfg.url("hybrid"),
                   FFMPEG=str(BREW / "bin/ffmpeg"), FFPROBE=str(BREW / "bin/ffprobe"))
        return uvicorn("webui", cfg, "webui", "app.main:app") + ["--proxy-headers"], WEBUI_SRC / "server", env
    raise SystemExit(f"unknown service: {name}")


def health_url(name: str, cfg: Config) -> str | None:
    if name == "postgres":
        return None
    if name == "ollama":
        return f"http://127.0.0.1:{cfg.ports['ollama']}/api/tags"
    return cfg.url(name) + {"setup": "/status", "webui": "/api/health"}.get(name, "/health")


# ── prepare: virtualenvs, MMAudio, the services' folders ─────────────────────

def say(text: str) -> None:
    print(f"   {text}", flush=True)


def run(command: list, **kwargs) -> None:
    subprocess.run([str(c) for c in command], check=True, **kwargs)


def dockerfile_arg(path: Path, name: str) -> str:
    match = re.search(rf"^ARG {name}=(\S+)", path.read_text(), re.M)
    if not match:
        raise SystemExit(f"{name} not found in {path}")
    return match.group(1)


def pip_lines(*paths: Path, skip: tuple[str, ...] = ()) -> list[str]:
    lines = []
    for path in paths:
        for line in path.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line and not any(line.startswith(s) for s in skip):
                lines.append(line)
    return lines


def venv_packages() -> dict[str, list[str]]:
    """Per virtualenv what its Dockerfile installs; torch as in the images (2.5.1)."""
    torch = ["torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1"]
    return {
        # MMAudio itself is installed from its pinned source below; nvidia-ml-py3 is CUDA-only
        "mmaudio": torch + ["torchcodec==0.2.1"] + pip_lines(REPO / "generation/mmaudio/requirements.txt", skip=("nvidia-ml-py3",)),
        # The search service, the indexer and the setup page share one image base
        "search": pip_lines(REPO / "search/api/requirements.txt") + [
            "transformers>=4.40,<5", "psycopg2-binary>=2.9", "sentencepiece>=0.1.99", "mutagen>=1.47",
            "fastapi>=0.110", "uvicorn[standard]>=0.27", "httpx>=0.27"],
        "gateway": pip_lines(REPO / "generation/gateway/requirements.txt"),
        "spotting": pip_lines(REPO / "spotting/api/requirements.txt") + ["imageio-ffmpeg>=0.5"],
        "hybrid": pip_lines(REPO / "hybrid/api/requirements.txt"),
    }


def prepare_venvs() -> None:
    mmaudio_ref = dockerfile_arg(REPO / "generation/mmaudio/Dockerfile", "MMAUDIO_REF")
    source = RUNTIME / "src/MMAudio"
    if not (source / ".git").exists():
        say("fetching the MMAudio source")
        source.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "-q", dockerfile_arg(REPO / "generation/mmaudio/Dockerfile", "MMAUDIO_REPO"), source])
    if subprocess.run(["git", "-C", source, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() != mmaudio_ref:
        run(["git", "-C", source, "fetch", "-q", "origin"])
        run(["git", "-C", source, "checkout", "-q", mmaudio_ref])
    for name, packages in venv_packages().items():
        venv = VENVS / name
        stamp = venv / ".installed"
        # The torch pins hold for every later install into the same virtualenv: MMAudio
        # asks only for torch>=2.5.1, and pip once resolved its dependencies to torch 2.15,
        # whose libraries torchaudio 2.5.1 can no longer load
        pins = [p for p in packages if re.match(r"(torch|torchvision|torchaudio)==", p)]
        wanted = hashlib.sha256(json.dumps([packages, mmaudio_ref if name == "mmaudio" else "", "pinned"]).encode()).hexdigest()
        if stamp.exists() and stamp.read_text() == wanted:
            say(f"virtualenv {name}: up to date")
            continue
        say(f"virtualenv {name}: installing (the first time downloads PyTorch, several hundred MB)")
        if not (venv / "bin/python").exists():
            run([PYTHON, "-m", "venv", venv])
        pip = [venv / "bin/python", "-m", "pip", "install", "-q", "--disable-pip-version-check"]
        run(pip + ["--upgrade", "pip"])
        if pins:
            constraints = venv / "constraints.txt"
            constraints.write_text("\n".join(pins) + "\n")
            pip += ["-c", constraints]
        run(pip + packages)
        if name == "mmaudio":
            run(pip + ["-e", source])
        if any(p.startswith("torchaudio==") for p in pins):
            # Fails here, not later in the service, if torch and torchaudio do not fit
            run([venv / "bin/python", "-c", "import torch, torchaudio"])
        stamp.write_text(wanted)


def prepare_webui(cfg: Config) -> None:
    """The web UI: its source at WEBUI_REF, the frontend built with Node (again only when
    the source changed), a virtualenv for its session server."""
    git = ["git", "-C", WEBUI_SRC]
    if not (WEBUI_SRC / ".git").exists():
        say(f"web UI: fetching {cfg.webui_repo}")
        WEBUI_SRC.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "-q", cfg.webui_repo, WEBUI_SRC])
    else:
        run(git + ["remote", "set-url", "origin", cfg.webui_repo])
        run(git + ["fetch", "-q", "origin"])
    # A managed copy: always exactly the wanted revision (a branch, a tag or a commit)
    ref = cfg.webui_ref
    target = f"origin/{ref}" if subprocess.run(git + ["rev-parse", "-q", "--verify", f"origin/{ref}"],
                                              capture_output=True).returncode == 0 else ref
    run(git + ["checkout", "-q", "--detach", target])
    head = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    say(f"web UI: {head[:7]} ({ref})")

    built = WEBUI_SRC / "dist/.built-from"
    if not (built.exists() and built.read_text() == head):
        say("web UI: building the frontend (npm ci, npm run build)")
        env = {**os.environ, "PATH": f"{BREW}/bin:/usr/bin:/bin:/usr/sbin:/sbin"}
        run([NPM, "ci", "--no-audit", "--no-fund", "--loglevel=error"], cwd=WEBUI_SRC, env=env)
        run([NPM, "run", "build"], cwd=WEBUI_SRC, env=env, stdout=subprocess.DEVNULL)
        built.write_text(head)

    requirements = WEBUI_SRC / "server/requirements.txt"
    venv = VENVS / "webui"
    stamp = venv / ".installed"
    wanted = hashlib.sha256(requirements.read_bytes()).hexdigest()
    if not (stamp.exists() and stamp.read_text() == wanted):
        say("virtualenv webui: installing")
        if not (venv / "bin/python").exists():
            run([PYTHON, "-m", "venv", venv])
        pip = [venv / "bin/python", "-m", "pip", "install", "-q", "--disable-pip-version-check"]
        run(pip + ["--upgrade", "pip"])
        run(pip + ["-r", requirements])
        stamp.write_text(wanted)
    else:
        say("virtualenv webui: up to date")
    for path in cfg.webui_sessions:
        if path == cfg.data / "sessions":
            path.mkdir(parents=True, exist_ok=True)
        elif not path.is_dir():
            say(f"web UI: the sessions folder {path} does not exist (WEBUI_SESSIONS_DIRS in .env)")
        elif protected_folder(path):
            say(f"web UI: {path} is inside {protected_folder(path)}, which macOS keeps background services out of;")
            say("        the web UI cannot list its sessions. Use a folder elsewhere, or give")
            say(f"        {PYTHON.resolve()} Full Disk Access (System Settings > Privacy & Security).")


def protected_folder(path: Path) -> str | None:
    """The folder macOS keeps launchd agents out of that `path` is in, if any."""
    home = Path.home()
    for name in ("Desktop", "Documents", "Downloads", "Library/Mobile Documents"):
        root = home / name
        if path == root or root in path.resolve().parents:
            return str(root)
    return None


def link(target: Path, at: Path) -> None:
    at.parent.mkdir(parents=True, exist_ok=True)
    if at.is_symlink() or at.exists():
        if at.is_symlink() and os.readlink(at) == str(target):
            return
        if at.is_dir() and not at.is_symlink():
            shutil.rmtree(at)
        else:
            at.unlink()
    at.symlink_to(target)


def common_folder(at: Path) -> None:
    """search/common with schema.sql beside it, as the images copy them."""
    at.mkdir(parents=True, exist_ok=True)
    wanted = {p.name: p for p in (REPO / "search/common").glob("*.py")}
    wanted["schema.sql"] = REPO / "search/schema.sql"
    for entry in at.iterdir():
        if entry.is_symlink() and entry.name not in wanted:
            entry.unlink()
    for name, source in wanted.items():
        link(source, at / name)


def prepare_layout(cfg: Config) -> None:
    """The services' folders laid out as in their images, as links to this repository:
    an edit in the repository is what the next start runs."""
    say("laying out the services' folders")
    mm = APPS / "mmaudio"
    link(REPO / "shared", mm / "shared")
    for script in ("main.py", "prefetch.py", "mps_compat.py"):
        link(REPO / "generation/mmaudio" / script, mm / "api" / script)
    # MMAudio resolves its checkpoints relative to its working directory (as entrypoint.sh links them)
    for sub in ("weights", "ext_weights"):
        (cfg.models / "mmaudio" / sub).mkdir(parents=True, exist_ok=True)
        link(cfg.models / "mmaudio" / sub, mm / "api" / sub)

    search = APPS / "search"
    link(REPO / "search/api/main.py", search / "main.py")
    link(REPO / "search/api/utils", search / "utils")
    common_folder(search / "common")

    setup = APPS / "setup"
    for name in ("main.py", "page.html"):
        link(REPO / "setup" / name, setup / name)
    for script in (REPO / "search/indexer").glob("*.py"):
        link(script, setup / script.name)
    link(REPO / "search/sources", setup / "sources")
    common_folder(setup / "common")

    # spotting: an ffmpeg with drawtext first in its PATH, and a font path without spaces
    ffmpeg = subprocess.run([VENVS / "spotting/bin/python", "-c", "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"],
                            capture_output=True, text=True, check=True).stdout.strip()
    link(Path(ffmpeg), RUNTIME / "bin/spotting/ffmpeg")
    font = next((Path(f) for f in STAMP_FONTS if Path(f).exists()), None)
    if font:
        link(font, RUNTIME / "fonts/stamp.ttf")
    else:
        say("no Arial font found: spotting sends its frames without the burnt-in time")

    for path in (cfg.audio, cfg.libraries, cfg.archive, cfg.own_sounds, cfg.hybrid_output, cfg.webui_cache,
                 cfg.models / "huggingface", cfg.models / "torch", LOGS / "setup", RUNTIME / "work/gateway"):
        path.mkdir(parents=True, exist_ok=True)


def prepare_database(cfg: Config) -> None:
    if (cfg.pgdata / "PG_VERSION").exists():
        say(f"database cluster: {cfg.pgdata}")
        return
    say(f"creating the database cluster in {cfg.pgdata}")
    cfg.pgdata.parent.mkdir(parents=True, exist_ok=True)
    # The macOS user owns the cluster (local socket, peer); the services log in over TCP with a password
    run([PG_BIN / "initdb", "-D", cfg.pgdata, "--encoding=UTF8", "--locale=en_US.UTF-8",
         "--auth-local=peer", "--auth-host=scram-sha-256"], stdout=subprocess.DEVNULL)


def ensure_database(cfg: Config) -> None:
    """Role, database and the vector extension; the services apply schema.sql themselves."""
    def psql(database: str, sql: str) -> str:
        return subprocess.run([PG_BIN / "psql", "-h", "/tmp", "-p", str(cfg.ports["postgres"]), "-d", database,
                               "-v", "ON_ERROR_STOP=1", "-Atqc", sql], capture_output=True, text=True, check=True).stdout
    for _ in range(30):
        if subprocess.run([PG_BIN / "pg_isready", "-q", "-h", "/tmp", "-p", str(cfg.ports["postgres"])]).returncode == 0:
            break
        time.sleep(1)
    password = cfg.db_password.replace("'", "''")
    user = cfg.db_user.replace('"', '""')
    exists = psql("postgres", f"SELECT 1 FROM pg_roles WHERE rolname = '{cfg.db_user}'").strip()
    verb = "ALTER" if exists else "CREATE"
    psql("postgres", f"{verb} ROLE \"{user}\" LOGIN PASSWORD '{password}'")
    if not psql("postgres", f"SELECT 1 FROM pg_database WHERE datname = '{cfg.db_name}'").strip():
        psql("postgres", f"CREATE DATABASE \"{cfg.db_name}\" OWNER \"{user}\"")
    psql(cfg.db_name, "CREATE EXTENSION IF NOT EXISTS vector")


# ── launchd ──────────────────────────────────────────────────────────────────

def label(name: str) -> str:
    return f"{LABEL}.{name}"


def plist_path(name: str) -> Path:
    return AGENTS / f"{label(name)}.plist"


def domain() -> str:
    return f"gui/{os.getuid()}"


def wanted_services(cfg: Config) -> list[str]:
    return [s for s in ORDER if (s != "ollama" or (cfg.bundled_ollama and cfg.vlm_api == "ollama"))
            and (s != "webui" or cfg.webui)]


def write_agent(name: str) -> bool:
    """The launchd agent of one service: runs `backend.py run <name>`, restarted when it
    exits, started at login. Returns whether the file changed."""
    LOGS.mkdir(parents=True, exist_ok=True)
    content = plistlib.dumps({
        "Label": label(name),
        "ProgramArguments": [str(PYTHON), str(Path(__file__).resolve()), "run", name],
        "WorkingDirectory": str(REPO),
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 15,
        "ProcessType": "Interactive" if name in ("mmaudio", "search", "ollama") else "Standard",
        "StandardOutPath": str(LOGS / f"{name}.log"),
        "StandardErrorPath": str(LOGS / f"{name}.log"),
    })
    path = plist_path(name)
    AGENTS.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == content:
        return False
    path.write_bytes(content)
    return True


def loaded(name: str) -> bool:
    return subprocess.run(["launchctl", "print", f"{domain()}/{label(name)}"], capture_output=True).returncode == 0


def pid_of(name: str) -> int | None:
    out = subprocess.run(["launchctl", "print", f"{domain()}/{label(name)}"], capture_output=True, text=True).stdout
    match = re.search(r"^\s*pid = (\d+)", out, re.M)
    return int(match.group(1)) if match else None


def probe(url: str | None, cfg: Config, name: str) -> bool:
    if url is None:
        return subprocess.run([PG_BIN / "pg_isready", "-q", "-h", "/tmp", "-p", str(cfg.ports["postgres"])]).returncode == 0
    try:
        with urllib.request.urlopen(url, timeout=4) as answer:
            return answer.status == 200
    except Exception:  # noqa: BLE001
        return False


def port_busy(port: int) -> bool:
    return subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"], capture_output=True).returncode == 0


def start(names: list[str], cfg: Config) -> None:
    for name in names:
        changed = write_agent(name)
        if loaded(name):
            if changed:
                subprocess.run(["launchctl", "bootout", f"{domain()}/{label(name)}"], capture_output=True)
                time.sleep(1)
            else:
                say(f"{name}: already running")
                continue
        port = cfg.ports.get(name)
        if port and port_busy(port):
            say(f"{name}: port {port} is taken by another program (a Docker stack?); not started")
            continue
        subprocess.run(["launchctl", "bootstrap", domain(), str(plist_path(name))], check=True)
        say(f"{name}: started (log: {LOGS / (name + '.log')})")
        if name == "postgres":
            ensure_database(cfg)


def stop(names: list[str]) -> None:
    for name in reversed(names):
        if loaded(name):
            subprocess.run(["launchctl", "bootout", f"{domain()}/{label(name)}"], capture_output=True)
            say(f"{name}: stopped")


def status(cfg: Config) -> int:
    print(f"{'service':<10} {'agent':<9} {'pid':>7}  {'answers':<8} where")
    bad = 0
    for name in wanted_services(cfg):
        is_loaded = loaded(name)
        ok = probe(health_url(name, cfg), cfg, name)
        bad += 0 if ok else 1
        where = f"localhost:{cfg.ports[name]}"
        print(f"{name:<10} {'loaded' if is_loaded else '-':<9} {pid_of(name) or '-':>7}  {'yes' if ok else 'no':<8} {where}")
    if cfg.vlm_api != "ollama" or not cfg.bundled_ollama:
        print(f"(spotting uses the vision-language model at {cfg.vlm_url})")
    print(f"\nsetup page: http://localhost:{cfg.ports['setup']}/    logs: {LOGS}")
    return 1 if bad else 0


# ── install: configuration, models, first start ──────────────────────────────

def set_env_value(key: str, value: str) -> None:
    """Replace or append KEY=value in .env (quoted when it holds spaces)."""
    path = REPO / ".env"
    text = path.read_text(encoding="utf-8")
    line = f'{key}="{value}"' if re.search(r"\s", value) else f"{key}={value}"
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=.*$", re.M)
    text = pattern.sub(lambda _: line, text) if pattern.search(text) else text.rstrip("\n") + f"\n{line}\n"
    path.write_text(text, encoding="utf-8")


def ask_folder(prompt: str, must_exist: bool) -> str:
    while True:
        answer = input(prompt).strip().strip("'\"")
        if not answer:
            return ""
        path = Path(os.path.expanduser(answer))
        if path.is_dir() or not must_exist:
            return str(path.resolve())
        say(f"{answer} is not a folder")


def configure(options: dict[str, str], interactive: bool) -> None:
    """.env as install.ps1 writes it: from the example with a random database password on
    the first run, the folders given as parameters or asked for. Values only Docker reads
    (container names) are left alone; this script translates them."""
    env_path = REPO / ".env"
    fresh = not env_path.exists()
    if fresh:
        shutil.copy(REPO / ".env.example", env_path)
        os.chmod(env_path, 0o600)
        set_env_value("POSTGRES_PASSWORD", hashlib.sha256(os.urandom(32)).hexdigest()[:24])
        # bfloat16 is emulated on the M1/M2 GPU; float32 is the safe default on a Mac
        set_env_value("FORCE_DTYPE", "float32")
        say("created .env from .env.example with a random database password")
        if not options.get("data_dir") and interactive:
            print()
            say("The data folder takes what the backend writes: the database, the models (about 21 GB), fetched or")
            say(f"downloaded audio (the BBC archive is about 60 GB) and your libraries. Default: {REPO / 'data'}")
            options["data_dir"] = ask_folder("   Data folder, Enter for the default: ", must_exist=False)
    else:
        say(".env exists (kept; delete it to start from the example again)")
    if options.get("data_dir"):
        data = Path(options["data_dir"]).expanduser().resolve()
        for sub in ("audio", "libraries", "archive", "own-sounds"):
            (data / sub).mkdir(parents=True, exist_ok=True)
        set_env_value("AUDIO_HOST_DIR", str(data / "audio"))
        set_env_value("LIBRARIES_HOST_DIR", str(data / "libraries"))
        set_env_value("BBC_ARCHIVE_HOST_DIR", str(data / "archive"))
        set_env_value("OWN_SOUNDS_HOST_DIR", str(data / "own-sounds"))
        say(f"data folder: {data}")
    if not options.get("models_dir") and interactive and fresh:
        print()
        say("The models live in <data folder>/models; on a Mac a folder loads as fast as anything else.")
        options["models_dir"] = ask_folder("   Another folder for the models, Enter for the default: ", must_exist=False)
    if options.get("models_dir"):
        models = Path(options["models_dir"]).expanduser().resolve()
        models.mkdir(parents=True, exist_ok=True)
        set_env_value("MODELS_STORE", str(models))
        say(f"models: {models}")
    if not options.get("bbc_archive") and interactive:
        print()
        say("The BBC Sound Effects archive (free for research use under the BBC's terms) can be downloaded")
        say("later from the setup page, or an existing copy on disk is used, read-only.")
        options["bbc_archive"] = ask_folder("   Folder of an existing BBC archive (<id>.wav or .mp3 files), Enter for none: ",
                                            must_exist=True)
    if not options.get("own_sounds") and interactive:
        options["own_sounds"] = ask_folder("   Folder with your own sounds to index (read-only), Enter for none: ",
                                           must_exist=True)
    if not options.get("sessions") and interactive and fresh:
        print()
        say("The web UI opens Pro Tools sessions (.ptx) from a folder you choose, read-only. Not in")
        say("Documents, Desktop or Downloads: macOS keeps the background service out of them.")
        options["sessions"] = ask_folder("   Folder with your Pro Tools sessions, Enter for <data folder>/sessions: ",
                                         must_exist=True)
    if options.get("webui") in ("on", "off"):
        set_env_value("MAC_WEBUI", "1" if options["webui"] == "on" else "0")
    if options.get("sessions"):
        sessions = Path(options["sessions"]).expanduser().resolve()
        if not sessions.is_dir():
            raise SystemExit(f"The sessions folder does not exist: {sessions}")
        set_env_value("WEBUI_SESSIONS_DIRS", str(sessions))
        say(f"web UI sessions: {sessions}" + (f" (inside {protected_folder(sessions)}: see the warning below)"
                                              if protected_folder(sessions) else ""))
    if options.get("bbc_archive"):
        if not Path(options["bbc_archive"]).is_dir():
            raise SystemExit(f"The BBC archive folder does not exist: {options['bbc_archive']}")
        set_env_value("BBC_ARCHIVE_HOST_DIR", str(Path(options["bbc_archive"]).resolve()))
        say(f"BBC archive: {options['bbc_archive']} (read-only; registered instead of downloaded)")
    if options.get("own_sounds"):
        own = Path(options["own_sounds"]).resolve()
        if not own.is_dir():
            raise SystemExit(f"The sounds folder does not exist: {own}")
        name = re.sub(r"[^a-z0-9_-]+", "-", own.name.lower()).strip("-") or "own"
        set_env_value("OWN_SOUNDS_HOST_DIR", str(own))
        set_env_value("OWN_SOUNDS_NAME", name)
        say(f"own sounds: {own} (read-only; library '{name}')")


def prefetch_mmaudio(cfg: Config) -> None:
    """MMAudio's weights (about 10 GB) with the download shown here rather than in a log."""
    argv, cwd, env = service_spec("mmaudio", cfg)
    run([argv[0], "prefetch.py"], cwd=cwd, env=env)


def pull_vlm(cfg: Config) -> None:
    env = {**os.environ, "OLLAMA_HOST": f"127.0.0.1:{cfg.ports['ollama']}"}
    if cfg.ollama_models_custom:
        env["OLLAMA_MODELS"] = str(cfg.ollama_models)
    for _ in range(30):
        if probe(health_url("ollama", cfg), cfg, "ollama"):
            break
        time.sleep(1)
    run([BREW / "bin/ollama", "pull", cfg.vlm_model], env=env)


def wait_for(names: list[str], cfg: Config, minutes: float) -> list[str]:
    deadline, waiting = time.time() + minutes * 60, list(names)
    while waiting and time.time() < deadline:
        waiting = [n for n in waiting if not probe(health_url(n, cfg), cfg, n)]
        if waiting:
            time.sleep(3)
    return waiting


def install(options: dict[str, str], interactive: bool) -> int:
    def step(text: str) -> None:
        print(f"\n== {text}", flush=True)

    step("Configuration")
    configure(options, interactive)
    cfg = Config()
    free = shutil.disk_usage(cfg.data if cfg.data.exists() else REPO).free // 2**30
    say(f"{free} GB free on the data folder's volume" + ("" if free >= 90 else
        ": the models and the index take about 40 GB, a downloaded BBC archive another 60 GB"))

    step("Python environments, MMAudio source, folders (the first time several GB of PyTorch)")
    prepare_venvs()
    prepare_layout(cfg)
    prepare_database(cfg)

    if cfg.webui:
        step("Web UI (the four modes in the browser)")
        prepare_webui(cfg)
    elif plist_path("webui").exists():
        # Left out now: its agent goes; its source in .mac/src stays for a later "on"
        stop(["webui"])
        plist_path("webui").unlink()
        say("web UI: switched off (MAC_WEBUI=0)")

    step("Database")
    start(["postgres"], cfg)

    if "ollama" in wanted_services(cfg):
        step(f"Vision-language model for spotting ({cfg.vlm_model}, about 6 GB the first time)")
        start(["ollama"], cfg)
        pull_vlm(cfg)

    step("MMAudio weights (about 10 GB the first time)")
    prefetch_mmaudio(cfg)

    step("Starting the services")
    # A service that already runs is restarted: after an update it has to load the new code
    running = [n for n in APP_SERVICES if n in wanted_services(cfg) and loaded(n)]
    if running:
        stop(running)
        time.sleep(2)
    start(wanted_services(cfg), cfg)
    say("waiting for them to answer (MMAudio loads its model onto the GPU first: a minute or two)")
    late = wait_for(wanted_services(cfg), cfg, minutes=6)
    for name in late:
        say(f"{name} does not answer yet; see: python3 mac/backend.py logs {name}")

    shortcuts = write_shortcuts(cfg)
    if shortcuts:
        step("Shortcuts")
        say(f"in {shortcuts}: " + ("Web UI, " if cfg.webui else "") +
            "Start backend, Stop backend, Backend status, Setup page (double-click)")
        if not GUI:
            subprocess.run(["open", str(shortcuts)])
    launcher = write_launcher(cfg)
    if launcher:
        step("App")
        say(f"{launcher}: opens the web UI (Launchpad, Spotlight, or drag it into the Dock)")

    step("Done")
    say(f"setup page: http://localhost:{cfg.ports['setup']}/ (the services' state, the sound index)")
    if cfg.webui:
        webui = f"http://localhost:{cfg.ports['webui']}/"
        say(f"web UI:     {webui}")
        if "webui" not in late and not GUI:
            subprocess.run(["open", webui])
    say("The services start at every login; 'Stop backend' stops them until the next login or 'Start backend'.")
    return 0


def database(cfg: Config, audio: bool, build: bool = False) -> int:
    """Fills the sound index as "Set everything up" on the setup page does: the prebuilt
    BBC index from Hugging Face (or, with `build`, the BBC metadata and the text
    embeddings made here), the audio (an existing archive folder registered, else
    downloaded when `audio`), the audio index. The setup service does the work and this
    shows its progress. So closing the window does not stop it, it carries on after a
    restart (the setup service's pipeline is kept in the database), and running this
    again shows where it is: finished steps are not repeated, the download skips files on
    disk, the embeddings and the audio index add only what is missing."""
    late = wait_for(["postgres", "search", "setup"], cfg, minutes=5)
    if late:
        say(f"{', '.join(late)} does not answer; see: python3 mac/backend.py status")
        if GUI:
            print(f"@@error The backend services do not answer ({', '.join(late)}); install them again.", flush=True)
        return 1
    base = cfg.url("setup")

    def call(path: str, body: dict | None = None) -> dict:
        request = urllib.request.Request(base + path, method="GET" if body is None else "POST",
                                         data=None if body is None else json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as answer:
            return json.load(answer)

    steps = {s["name"]: s for s in call("/status")["steps"]}
    archive = "bbc_archive" in steps
    wanted = (["bbc_metadata", "text_embeddings"] if build else ["bbc_index"]) \
        + (["bbc_archive"] if archive else ["bbc_audio"] if audio else []) + ["audio_index"]
    if audio and not archive and steps.get("bbc_audio", {}).get("status") != "complete":
        free = shutil.disk_usage(cfg.audio if cfg.audio.exists() else cfg.data).free // 2**30
        if free < 70:
            say(f"The BBC audio takes about 60 GB, and {cfg.audio.parent} has {free} GB free: not started.")
            say("Free some room, or give another data folder (install_mac.sh --data-dir), then run this again.")
            if GUI:
                print(f"@@error The BBC audio needs about 60 GB; {free} GB are free. Free some room and try again.", flush=True)
            return 1
    say("BBC Sound Effects: free for personal, educational and research use under the BBC's terms")
    say("(https://sound-effects.bbcrewind.co.uk/licensing).")
    if audio and not archive:
        say("All of its audio (about 60 GB) is downloaded, then indexed: several hours.")
    elif not archive:
        say("Without its audio: a sound is fetched the first time it is used. The search by sound and the")
        say("Hybrid mode's library pieces need audio on disk (the installer's audio option).")
    say("This window can be closed at any time: the work goes on in the background, also after a")
    say("restart, and running the installer again shows where it is.")
    if GUI:
        print("@@status Starting the sound database", flush=True)
    print(flush=True)

    call("/pipeline/start", {"download": audio, "prebuilt": not build})
    retried: set[str] = set()
    done: set[str] = set()
    line = ""
    try:
        while True:
            time.sleep(4)
            state = call("/status")
            steps = {s["name"]: s for s in state["steps"]}
            for name in wanted:
                s = steps.get(name)
                if s and name not in done and s["status"] in ("complete", "done"):
                    done.add(name)
                    if GUI:
                        print(f"   ✓ {s['title']}: {s['detail']}", flush=True)
                    else:
                        print(f"\r\033[K   \033[32m✓\033[0m {s['title']}: {s['detail']}", flush=True)
                    line = ""
            running = [n for n in wanted if n in state["running"]]
            if running:
                s = steps[running[0]]
                share = f"{100 * s['done'] / s['total']:.0f}%  " if s["total"] else ""
                text = f"   … {s['title']}: {share}{s['detail']}"[:150]
                if text != line:
                    if GUI:
                        print(f"@@status {s['title']}: {s['detail']}", flush=True)
                        print(f"@@progress {100 * s['done'] / s['total']:.1f}" if s["total"] else "@@progress -", flush=True)
                    else:
                        print(f"\r\033[K{text}", end="", flush=True)
                    line = text
                continue
            if state["pipeline"].get("active"):
                # Between two steps: the setup service starts the next one within seconds
                continue
            failed = [n for n in wanted if steps.get(n, {}).get("status") == "error"]
            # A step that failed is not retried by the pipeline itself: once per run here
            retry = [n for n in failed if n not in retried]
            if retry:
                print(("" if GUI else "\r\033[K") + f"   {steps[retry[0]]['title']} had failed before; trying it again", flush=True)
                retried.add(retry[0])
                call(f"/jobs/{retry[0]}/start", {})
                call("/pipeline/start", {"download": audio, "prebuilt": not build})
                continue
            if line and not GUI:
                print()
            if failed:
                for name in failed:
                    s = steps[name]
                    say(f"{s['title']} failed: {s.get('error') or ''}")
                    if GUI:
                        print(f"@@error {s['title']} failed; Try Again carries on where it stopped.", flush=True)
                    for log_line in s.get("log") or []:
                        say(f"  {log_line}")
                say("Run the installer again to retry; the log: python3 mac/backend.py logs setup")
                return 1
            for name in wanted:
                s = steps.get(name)
                if s and name not in done and s["status"] == "partial":
                    # Ran through with a few items left: files the archive does not serve or ffmpeg cannot read
                    say(f"{s['title']}: {s['detail']}")
            search = state.get("search") or {}
            say(f"sound database ready: {search.get('sounds', 0):,} searchable sounds"
                + (", search by sound available" if search.get("by_sound") else ""))
            return 0
    except KeyboardInterrupt:
        print()
        say("This window stops showing it; the sound database keeps loading in the background.")
        return 0


APP_SERVICES = ["mmaudio", "gateway", "search", "spotting", "hybrid", "setup", "webui"]


def write_shortcuts(cfg: Config) -> Path | None:
    """Double-clickable files next to the installation in ~/AI Sound Design, for whoever
    never opens Terminal. Only there: a checkout elsewhere is someone at home on the
    command line. Written here, so macOS does not hold them as downloaded files."""
    folder = REPO.parent
    if folder.name != "AI Sound Design":
        return None
    backend = f'"{PYTHON}" "{Path(__file__).resolve()}"'
    close = 'echo; read -r -n 1 -s -p "Press any key to close this window." && echo'
    scripts = {
        "Start backend.command": f'{backend} start\nsleep 3\nopen "http://localhost:{cfg.ports["setup"]}/"\n{close}',
        "Stop backend.command": f'{backend} stop\necho\necho "Stopped until the next login or Start backend."\n{close}',
        "Backend status.command": f'{backend} status\n{close}',
    }
    for name, body in scripts.items():
        path = folder / name
        path.write_text(f"#!/bin/bash\n# AI Sound Design backend, written by install_mac.sh\nclear\n{body}\n")
        path.chmod(0o755)
    (folder / "Setup page.webloc").write_bytes(plistlib.dumps({"URL": f"http://localhost:{cfg.ports['setup']}/"}))
    webui = folder / "Web UI.webloc"
    if cfg.webui:
        webui.write_bytes(plistlib.dumps({"URL": f"http://localhost:{cfg.ports['webui']}/"}))
    elif webui.exists():
        webui.unlink()
    return folder


LAUNCHER_NAME = "AI Sound Design.app"
LAUNCHER_SCRIPT = """\
-- Opens the AI Sound Design web UI in the browser; written by mac/backend.py.
-- The services run in the background from login on; after "Stop backend" this starts
-- them first.
on run
    set webui to "http://localhost:%(port)s/"
    if not my answers(webui & "api/health") then
        display notification "Starting the services; a minute at most." with title "AI Sound Design"
        try
            do shell script %(start)s
        end try
        repeat 90 times
            if my answers(webui & "api/health") then exit repeat
            delay 1
        end repeat
    end if
    if my answers(webui & "api/health") then
        open location webui
    else
        display alert "AI Sound Design does not answer." message ¬
            "The services did not start. 'Backend status' in the folder AI Sound Design (in your home folder) says which one is missing; the installer sets them up again." as warning
    end if
end run

on answers(address)
    try
        do shell script "/usr/bin/curl -fsS -m 2 -o /dev/null " & quoted form of address
        return true
    on error
        return false
    end try
end answers
"""


def launcher_path() -> Path:
    """/Applications when this account may write there (an administrator's), else
    ~/Applications; Launchpad and Spotlight find either."""
    apps = Path("/Applications")
    return (apps if os.access(apps, os.W_OK) else Path.home() / "Applications") / LAUNCHER_NAME


def write_launcher(cfg: Config) -> Path | None:
    """The app "AI Sound Design": opens the web UI, starting the services first when they
    are stopped. An AppleScript applet, compiled here with osacompile (part of macOS), so
    nothing has to be downloaded or opened past Gatekeeper. Its icon comes from the
    installer app (AISD_ICON), or stays from an earlier one."""
    path = launcher_path()
    for old in {Path("/Applications") / LAUNCHER_NAME, Path.home() / "Applications" / LAUNCHER_NAME} - {path}:
        if (old / "Contents/Resources/Scripts/main.scpt").exists():
            shutil.rmtree(old, ignore_errors=True)        # one of ours from before, elsewhere
    if not cfg.webui:
        if (path / "Contents/Resources/Scripts/main.scpt").exists():
            shutil.rmtree(path, ignore_errors=True)
        return None
    icon = Path(os.environ.get("AISD_ICON", ""))
    kept = RUNTIME / "launcher.icns"
    if icon.is_file():
        shutil.copyfile(icon, kept)
    elif (path / "Contents/Resources/applet.icns").exists() and not kept.exists():
        shutil.copyfile(path / "Contents/Resources/applet.icns", kept)
    start = f'{PYTHON} "{Path(__file__).resolve()}" start >/dev/null 2>&1'
    script = LAUNCHER_SCRIPT % {"port": cfg.ports["webui"], "start": json.dumps(start)}
    source = RUNTIME / "launcher.applescript"
    source.write_text(script, encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    built = path.with_name(".AI Sound Design (new).app")      # osacompile makes an app only for a .app name
    shutil.rmtree(built, ignore_errors=True)
    if subprocess.run(["osacompile", "-o", str(built), str(source)], capture_output=True).returncode != 0:
        say("the app AI Sound Design could not be made (osacompile failed); the web UI is at the address above")
        return None
    with open(built / "Contents/Info.plist", "rb") as handle:
        info = plistlib.load(handle)
    if kept.exists():
        shutil.copyfile(kept, built / "Contents/Resources/applet.icns")
        # The applet's default icon in Assets.car would win over applet.icns
        (built / "Contents/Resources/Assets.car").unlink(missing_ok=True)
        info.pop("CFBundleIconName", None)
    info.update({"CFBundleIdentifier": "com.ai-sound-design.webui", "CFBundleName": "AI Sound Design",
                 "LSMinimumSystemVersion": "14.0"})
    with open(built / "Contents/Info.plist", "wb") as handle:
        plistlib.dump(info, handle)
    subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(built)], capture_output=True)
    shutil.rmtree(path, ignore_errors=True)
    built.rename(path)
    # Finder and the Dock show the new icon without a restart
    subprocess.run(["touch", str(path)])
    return path


# ── run: what a launchd agent executes ───────────────────────────────────────

def run_service(name: str) -> None:
    cfg = Config()
    argv, cwd, env = service_spec(name, cfg)
    if name == "mmaudio" and env.get("MMAUDIO_PREFETCH") == "1":
        # As the image's entrypoint: the weights are on disk before the server starts,
        # or the service does not start at all (launchd tries again)
        result = subprocess.run([argv[0], "prefetch.py"], cwd=cwd, env=env)
        if result.returncode != 0:
            sys.exit(result.returncode)
    os.chdir(cwd)
    os.execve(argv[0], argv, env)


# ── command line ─────────────────────────────────────────────────────────────

def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    command, rest = args[0], args[1:]
    if command == "run":
        run_service(rest[0])
        return 0
    if command == "install":
        options, flags = {}, {"--data-dir": "data_dir", "--bbc-archive": "bbc_archive",
                              "--own-sounds": "own_sounds", "--models-dir": "models_dir",
                              "--sessions": "sessions", "--webui": "webui"}
        i = 0
        while i < len(rest):
            if rest[i] in flags and i + 1 < len(rest):
                options[flags[rest[i]]] = rest[i + 1]
                i += 2
            elif rest[i] == "--no-prompt":
                i += 1
            else:
                raise SystemExit(f"unknown option: {rest[i]}")
        if options.get("webui", "on") not in ("on", "off"):
            raise SystemExit("--webui on|off")
        interactive = "--no-prompt" not in rest and sys.stdin.isatty()
        return install(options, interactive)
    cfg = Config()
    if command == "database":
        if [a for a in rest if a not in ("--audio", "--build-index")]:
            raise SystemExit("database [--audio] [--build-index]")
        return database(cfg, audio="--audio" in rest, build="--build-index" in rest)
    unknown = [n for n in rest if n not in ORDER and not n.isdigit()]
    if unknown and command != "logs":
        raise SystemExit(f"unknown service(s): {', '.join(unknown)}; known: {', '.join(ORDER)}")
    names = [n for n in ORDER if n in rest] or wanted_services(cfg)
    if command == "prepare":
        prepare_venvs()
        if cfg.webui:
            prepare_webui(cfg)
        prepare_layout(cfg)
        prepare_database(cfg)
        return 0
    if command == "start":
        prepare_layout(cfg)
        start(names, cfg)
        return 0
    if command == "stop":
        stop(names)
        return 0
    if command == "restart":
        stop(names)
        time.sleep(2)
        prepare_layout(cfg)
        start(names, cfg)
        return 0
    if command == "status":
        return status(cfg)
    if command == "logs":
        if not rest:
            raise SystemExit("logs <service> [lines]")
        lines = rest[1] if len(rest) > 1 else "60"
        return subprocess.run(["tail", "-n", lines, str(LOGS / f"{rest[0]}.log")]).returncode
    if command == "uninstall":
        stop(ORDER)
        for name in ORDER:
            plist_path(name).unlink(missing_ok=True)
        if (launcher_path() / "Contents/Resources/Scripts/main.scpt").exists():
            shutil.rmtree(launcher_path(), ignore_errors=True)
        say("launchd agents removed; the data folder and .mac/ are left in place")
        return 0
    raise SystemExit(f"unknown command: {command}\n{__doc__}")


if __name__ == "__main__":
    sys.exit(main())
