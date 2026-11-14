#!/usr/bin/env python3
r"""
The backend on Windows, without Docker.

For a PC without Docker Desktop: every service runs natively, each in a virtualenv of
its own with the versions its Dockerfile installs, from the same source files: the
same ports, the same .env, the same setup page. PyTorch comes from its CUDA 12.4 wheels,
PostgreSQL 17 from EDB's binaries with pgvector built for it, the vision-language model
from the Ollama installed on Windows (install_windows.ps1 installs what is missing).

    python windows\backend.py install     # everything; install_windows.ps1 runs it after the prerequisites
    python windows\backend.py prepare     # only PostgreSQL, the virtualenvs, the MMAudio source, the folders
    python windows\backend.py start [service ...]   # kept running by a supervisor; started at every logon
    python windows\backend.py stop [service ...]
    python windows\backend.py restart [service ...]
    python windows\backend.py status
    python windows\backend.py logs <service> [lines]
    python windows\backend.py uninstall   # stops everything, removes the logon task; data and .windows\ stay
    python windows\backend.py database [--audio]   # fills the sound index, showing its progress
    python windows\backend.py webui       # only the web UI (beside a Docker stack, say)

Services: postgres, mmaudio, gateway, search, spotting, hybrid, setup, webui.

The web UI (https://github.com/ai-sound-design/ai-sound-webui, the plugin's four modes
in the browser, no Pro Tools needed) is installed alongside: its source is fetched next
to this repository (AISD_WEBUI_DIR, default ..\ai-sound-webui), the frontend built with
Node, and its session server serves it on port 8030 (WEBUI_PORT) and forwards /svc/* to
the services. WINDOWS_WEBUI=0 leaves it out.

Where things live (all from .env, as for Docker):
    the data folder        AUDIO_HOST_DIR\.. (default .\data): audio, libraries, the
                           database (postgres\), the models (models\) and the hybrid
                           service's output (hybrid\)
    the models             MODELS_STORE when it names a folder, else <data>\models
    the Ollama models      OLLAMA_MODELS_STORE when it names a folder, else Ollama's own
                           %UserProfile%\.ollama\models
    everything installed   .windows\ in this repository: PostgreSQL, the virtualenvs, the
                           MMAudio source, the services' folders laid out as in their
                           images, logs

A few settings only mean something here (put them in .env to change them):
    WINDOWS_BIND_HOST   the address the services listen on (default 127.0.0.1: this PC only;
                        0.0.0.0 makes them reachable from the network, as Docker does)
    WEBUI_SESSIONS_DIRS the folders with Pro Tools sessions the web UI shows (several,
                        separated by ";"; default <data>\sessions)
    WEBUI_REPO, WEBUI_REF  where the web UI comes from (default its GitHub main)
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
import urllib.request
import winreg
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNTIME = REPO / ".windows"
VENVS = RUNTIME / "venv"
APPS = RUNTIME / "app"
LOGS = RUNTIME / "logs"
PIDS = RUNTIME / "pids"
PG = RUNTIME / "pgsql"
PG_BIN = PG / "bin"
# python.exe for everything that runs a service (a hidden console, its output into a log);
# pythonw.exe only for the supervisor, which is started by the logon task without any console
PYTHON = Path(sys.executable).with_name("python.exe")
PYTHONW = PYTHON.with_name("pythonw.exe")
# Run by the installer app (github.com/ai-sound-design/installer): it opens what it wants itself,
# and reads the database's progress from @@ lines
GUI = os.environ.get("AISD_GUI") == "1"
ORDER = ["postgres", "mmaudio", "gateway", "search", "spotting", "hybrid", "setup", "webui"]
APP_SERVICES = ["mmaudio", "gateway", "search", "spotting", "hybrid", "setup", "webui"]
WEBUI_SRC = Path(os.environ.get("AISD_WEBUI_DIR") or (REPO.parent / "ai-sound-webui")).resolve()
TASK_NAME = "AI Sound Design backend"
PG_VERSION = "17.12-1"
PG_URL = f"https://get.enterprisedb.com/postgresql/postgresql-{PG_VERSION}-windows-x64-binaries.zip"
PGVECTOR_URL = os.environ.get("AISD_PGVECTOR_URL") or \
    "https://github.com/ai-sound-design/installer/releases/latest/download/pgvector-pg17-windows-x64.zip"
TORCH_INDEX = "https://download.pytorch.org/whl/cu124"
STAMP_FONTS = [r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeuib.ttf"]
CREATE_NO_WINDOW = 0x08000000
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
USER_AGENT = "ai-sound-design-installer"

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ── configuration ────────────────────────────────────────────────────────────

def read_env() -> dict[str, str]:
    """KEY=value lines of .env (or .env.example before the first install)."""
    path = REPO / ".env" if (REPO / ".env").exists() else REPO / ".env.example"
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
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
    path = Path(os.path.expandvars(os.path.expanduser(value.replace("/", "\\"))))
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
        self.webui = get("WINDOWS_WEBUI", "1") != "0"
        self.webui_repo = get("WEBUI_REPO", "ai-sound-design/ai-sound-webui")
        self.webui_ref = get("WEBUI_REF", "main")
        self.webui_sessions = [folder(p) for p in get("WEBUI_SESSIONS_DIRS", str(self.data / "sessions")).split(";") if p.strip()]
        self.webui_cache = self.data / "webui-cache"
        self.host = get("WINDOWS_BIND_HOST", "127.0.0.1")
        self.ports = {"mmaudio": int(get("MMAUDIO_PORT", "8000")), "gateway": int(get("GENERATION_PORT", "8010")),
                      "search": int(get("SOUND_SEARCH_PORT", "8002")), "spotting": int(get("SPOTTING_PORT", "8003")),
                      "hybrid": int(get("HYBRID_PORT", "8004")), "setup": int(get("SETUP_PORT", "8020")),
                      "postgres": int(get("POSTGRES_PORT", "5432")), "webui": int(get("WEBUI_PORT", "8030"))}
        self.db_name, self.db_user = get("POSTGRES_DB", "sounds"), get("POSTGRES_USER", "sound_user")
        self.db_password = get("POSTGRES_PASSWORD", "change_me")
        self.vlm_model = get("VLM_MODEL", "gemma4:e4b-it-qat")
        self.vlm_api = get("VLM_API", "ollama").lower()
        # The Docker defaults name containers; here the Ollama is the one installed on Windows
        vlm_url = get("VLM_URL", "http://ollama:11434")
        if re.match(r"^https?://ollama(:\d+)?/?$", vlm_url):
            vlm_url = "http://127.0.0.1:11434"
        self.vlm_url = vlm_url.replace("host.docker.internal", "127.0.0.1").rstrip("/")
        self.ollama_local = self.vlm_api == "ollama" and re.match(r"^https?://(127\.0\.0\.1|localhost)(:\d+)?$", self.vlm_url) is not None

    def url(self, service: str) -> str:
        return f"http://127.0.0.1:{self.ports[service]}"

    @property
    def database_url(self) -> str:
        return f"postgresql://{self.db_user}:{self.db_password}@127.0.0.1:{self.ports['postgres']}/{self.db_name}"


# ── programs installed on Windows ────────────────────────────────────────────

def registry_path() -> str:
    """PATH as new programs see it: the machine's and the user's from the registry (a winget
    install extends them, but not this process's), then what this process has on top, each
    folder once. A PATH that repeats itself soon passes the 8 KB cmd.exe can take, and npm's
    scripts then find neither node nor the system's own tools."""
    sources = []
    for hive, key in ((winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
                      (winreg.HKEY_CURRENT_USER, r"Environment")):
        try:
            with winreg.OpenKey(hive, key) as handle:
                sources.append(os.path.expandvars(winreg.QueryValueEx(handle, "Path")[0]))
        except OSError:
            pass
    sources.append(os.environ.get("PATH", ""))
    seen: set[str] = set()
    parts = []
    for source in sources:
        for folder in source.split(";"):
            folder = folder.strip()
            key = folder.rstrip("\\").lower()
            if folder and key not in seen:
                seen.add(key)
                parts.append(folder)
    return ";".join(parts)


def find_exe(name: str, *candidates: str) -> Path | None:
    for candidate in candidates:
        path = Path(os.path.expandvars(candidate))
        if path.exists():
            return path
    found = shutil.which(name, path=registry_path())
    return Path(found) if found else None


class Tools:
    def __init__(self) -> None:
        local = os.environ.get("LOCALAPPDATA", "")
        links = rf"{local}\Microsoft\WinGet\Links"
        self.ffmpeg = find_exe("ffmpeg", rf"{links}\ffmpeg.exe")
        self.ffprobe = find_exe("ffprobe", rf"{links}\ffprobe.exe")
        self.node = find_exe("node", r"%ProgramFiles%\nodejs\node.exe")
        self.npm = find_exe("npm.cmd", r"%ProgramFiles%\nodejs\npm.cmd")
        self.ollama = find_exe("ollama", rf"{local}\Programs\Ollama\ollama.exe")

    def path_prefix(self) -> list[str]:
        dirs = [str(PG_BIN)]
        for exe in (self.ffmpeg, self.node):
            if exe and str(exe.parent) not in dirs:
                dirs.append(str(exe.parent))
        return dirs


# ── what each service is ─────────────────────────────────────────────────────

def base_env(cfg: Config, tools: Tools) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k.upper() not in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV")}
    env.update({
        "PATH": ";".join(tools.path_prefix() + [registry_path()]),
        "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
        "LOG_LEVEL": cfg.env.get("LOG_LEVEL", "info"),
        "HF_HOME": str(cfg.models / "huggingface"), "TORCH_HOME": str(cfg.models / "torch"),
        "FORCE_DEVICE": cfg.env.get("FORCE_DEVICE") or "auto",
    })
    return env


def search_env(cfg: Config) -> dict[str, str]:
    """What the search service, the indexer and the setup page share (as in docker-compose.yml)."""
    e = cfg.env
    return {"DATABASE_URL": cfg.database_url, "AUDIO_DIR": str(cfg.audio), "LIBRARIES_DIR": str(cfg.libraries),
            "BBC_ARCHIVE_DIR": str(cfg.archive), "OWN_SOUNDS_DIR": str(cfg.own_sounds),
            "OWN_SOUNDS_NAME": e.get("OWN_SOUNDS_NAME") or "own", "BBC_AUDIO_FORMAT": e.get("BBC_AUDIO_FORMAT") or "mp3",
            "CLAP_MODEL": e.get("CLAP_MODEL") or "laion/clap-htsat-fused", "AUDIO_FETCH": e.get("AUDIO_FETCH") or "on_demand",
            **{k: e[k] for k in ("AUDIO_INDEX", "SEGMENT_THRESHOLD", "REFINE_TOP", "XCLIP_MODEL", "CLAP_CPU_FEATURES") if e.get(k)}}


def venv_python(name: str) -> Path:
    return VENVS / name / "Scripts" / "python.exe"


def uvicorn(venv: str, cfg: Config, service: str, module: str = "main:app") -> list[str]:
    return [str(venv_python(venv)), "-m", "uvicorn", module, "--host", cfg.host,
            "--port", str(cfg.ports[service]), "--log-level", "warning" if service == "setup" else "info"]


def service_spec(name: str, cfg: Config) -> tuple[list[str], Path, dict[str, str]]:
    """argv, working directory and environment of one service."""
    tools = Tools()
    env, e = base_env(cfg, tools), cfg.env
    if name == "postgres":
        return ([str(PG_BIN / "postgres.exe"), "-D", str(cfg.pgdata), "-c", f"listen_addresses={cfg.host}",
                 "-c", f"port={cfg.ports['postgres']}"], cfg.pgdata, env)
    if name == "mmaudio":
        env.update(API_HOST=cfg.host, API_PORT=str(cfg.ports["mmaudio"]), MMAUDIO_PATH=str(RUNTIME / "src/MMAudio"),
                   MMAUDIO_MODEL=e.get("MMAUDIO_MODEL") or "large_44k_v2",
                   MMAUDIO_PREFETCH=e.get("MMAUDIO_PREFETCH") or "1",
                   FORCE_DTYPE=e.get("FORCE_DTYPE") or "bfloat16",
                   VIDEO_CACHE_MAX_GB=e.get("VIDEO_CACHE_MAX_GB") or "8",
                   VIDEO_CACHE_TTL_MIN=e.get("VIDEO_CACHE_TTL_MIN") or "60",
                   ENABLE_TRACEMALLOC=e.get("ENABLE_TRACEMALLOC") or "false")
        return [str(venv_python("mmaudio")), "main.py"], APPS / "mmaudio/api", env
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
        font = next((f for f in STAMP_FONTS if Path(f).exists()), "")
        env.update(VLM_URL=cfg.vlm_url, VLM_MODEL=cfg.vlm_model, VLM_API=cfg.vlm_api)
        if font:
            env["STAMP_FONT"] = font
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
        # The session server of the web UI: serves the built frontend (dist\), the Pro Tools
        # sessions in WEBUI_SESSIONS_DIRS, and forwards /svc/* to the services
        env.update(STATIC_DIR=str(WEBUI_SRC / "dist"), CACHE_DIR=str(cfg.webui_cache),
                   SESSIONS_DIRS=";".join(str(p) for p in cfg.webui_sessions),
                   GENERATION_URL=cfg.url("gateway"), SEARCH_URL=cfg.url("search"),
                   SPOTTING_URL=cfg.url("spotting"), HYBRID_URL=cfg.url("hybrid"))
        if tools.ffmpeg:
            env["FFMPEG"] = str(tools.ffmpeg)
        if tools.ffprobe:
            env["FFPROBE"] = str(tools.ffprobe)
        return uvicorn("webui", cfg, "webui", "app.main:app") + ["--proxy-headers"], WEBUI_SRC / "server", env
    raise SystemExit(f"unknown service: {name}")


def health_url(name: str, cfg: Config) -> str | None:
    if name == "postgres":
        return None
    if name == "setup":
        return cfg.url("setup") + "/status"
    if name == "webui":
        return cfg.url("webui") + "/api/health"
    return cfg.url(name) + "/health"


def wanted_services(cfg: Config) -> list[str]:
    return [s for s in ORDER if s != "webui" or cfg.webui]


# ── helpers ──────────────────────────────────────────────────────────────────

def say(text: str) -> None:
    print(f"   {text}", flush=True)


def status_line(text: str) -> None:
    if GUI:
        print(f"@@status {text}", flush=True)


def run(command: list, **kwargs) -> None:
    kwargs.setdefault("check", True)
    subprocess.run([str(c) for c in command], **kwargs)


def pip_lines(*paths: Path, skip: tuple[str, ...] = ()) -> list[str]:
    lines = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line and not any(line.lower().startswith(s.lower()) for s in skip):
                lines.append(line)
    return lines


def dockerfile_arg(path: Path, name: str) -> str:
    match = re.search(rf"^ARG {name}=(\S+)", path.read_text(encoding="utf-8"), re.M)
    if not match:
        raise SystemExit(f"{path} has no ARG {name}")
    return match.group(1)


def http_json(url: str, timeout: float = 20) -> dict | None:
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            return json.load(answer)
    except Exception:  # noqa: BLE001
        return None


def download(url: str, target: Path, what: str) -> None:
    """A file with its progress shown (the app's bar, or a percentage in the console)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as answer, part.open("wb") as out:
            total = int(answer.headers.get("Content-Length") or 0)
            done, shown, last = 0, -1, time.time()
            while True:
                chunk = answer.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total and time.time() - last > 0.5:
                    percent = int(100 * done / total)
                    if percent != shown:
                        shown, last = percent, time.time()
                        if GUI:
                            print(f"@@progress {percent}", flush=True)
                        else:
                            print(f"\r   {what}: {percent}%  ", end="", flush=True)
    except Exception as error:  # noqa: BLE001
        part.unlink(missing_ok=True)
        raise SystemExit(f"Could not download {what} from {url}: {error}")
    if not GUI:
        print()
    part.replace(target)


def extract(archive: Path, into: Path, prefix: str, keep: tuple[str, ...]) -> None:
    """The entries under <prefix>/<one of keep>/... of a zip, without the prefix."""
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            name = member.filename
            if not name.startswith(prefix):
                continue
            rest = name[len(prefix):]
            if not any(rest == k or rest.startswith(k + "/") for k in keep):
                continue
            if member.is_dir():
                continue
            target = into / rest
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def junction(target: Path, at: Path) -> None:
    """A directory junction (no privileges needed, unlike a symlink) to a folder elsewhere."""
    if at.is_dir() and not at.is_junction():
        shutil.rmtree(at)
    if at.is_junction():
        if os.path.realpath(at).lower() == os.path.realpath(target).lower():
            return
        at.rmdir()
    at.parent.mkdir(parents=True, exist_ok=True)
    import _winapi
    _winapi.CreateJunction(str(target), str(at))


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == source.stat().st_size \
            and int(target.stat().st_mtime) == int(source.stat().st_mtime):
        return
    shutil.copy2(source, target)


def copy_tree(source: Path, target: Path, only: tuple[str, ...] = ()) -> None:
    for path in source.rglob("*"):
        if path.is_dir() or "__pycache__" in path.parts:
            continue
        if only and path.suffix not in only:
            continue
        copy_file(path, target / path.relative_to(source))


def set_env_value(key: str, value: str) -> None:
    """Replace or append KEY=value in .env (quoted when it holds spaces)."""
    path = REPO / ".env"
    text = path.read_text(encoding="utf-8-sig")
    line = f'{key}="{value}"' if re.search(r"\s", value) else f"{key}={value}"
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=.*$", re.M)
    text = pattern.sub(lambda _: line, text) if pattern.search(text) else text.rstrip("\n") + f"\n{line}\n"
    path.write_text(text, encoding="utf-8")


def probe(url: str | None, cfg: Config) -> bool:
    if url is None:
        return subprocess.run([str(PG_BIN / "pg_isready.exe"), "-q", "-h", "127.0.0.1", "-p", str(cfg.ports["postgres"])],
                              creationflags=CREATE_NO_WINDOW).returncode == 0
    try:
        with urllib.request.urlopen(url, timeout=4) as answer:
            return answer.status == 200
    except Exception:  # noqa: BLE001
        return False


def wait_for(names: list[str], cfg: Config, minutes: float) -> list[str]:
    deadline, waiting = time.time() + minutes * 60, list(names)
    while waiting and time.time() < deadline:
        waiting = [n for n in waiting if not probe(health_url(n, cfg), cfg)]
        if waiting:
            time.sleep(3)
    return waiting


def port_busy(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


# ── PostgreSQL ───────────────────────────────────────────────────────────────

def prepare_postgres() -> None:
    """PostgreSQL 17 (EDB's binaries, the server parts only) with pgvector built for it,
    in .windows\\pgsql."""
    marker = PG / ".installed"
    wanted = f"postgresql {PG_VERSION}, pgvector from {PGVECTOR_URL}"
    if marker.exists() and marker.read_text(encoding="utf-8") == wanted and (PG / "share/extension/vector.control").exists():
        say("PostgreSQL 17 with pgvector: installed")
        return
    downloads = RUNTIME / "downloads"
    archive = downloads / f"postgresql-{PG_VERSION}-windows-x64-binaries.zip"
    if not (PG_BIN / "postgres.exe").exists():
        if not archive.exists():
            say("fetching PostgreSQL 17 from EDB (about 390 MB)")
            status_line("Downloading PostgreSQL 17")
            download(PG_URL, archive, "PostgreSQL 17")
        say("unpacking PostgreSQL (the server, not pgAdmin)")
        extract(archive, PG, "pgsql/", ("bin", "lib", "share"))
        archive.unlink(missing_ok=True)
    vector = downloads / "pgvector-pg17-windows-x64.zip"
    source = os.environ.get("AISD_PGVECTOR_ZIP")
    if source and Path(source).exists():
        shutil.copy2(source, vector)
    else:
        say("fetching pgvector for PostgreSQL 17 on Windows")
        download(PGVECTOR_URL, vector, "pgvector")
    extract(vector, PG, "", ("lib", "share", "include"))
    if not (PG / "share/extension/vector.control").exists() or not (PG / "lib/vector.dll").exists():
        raise SystemExit(f"{vector} does not hold pgvector (lib\\vector.dll, share\\extension\\vector.control)")
    vector.unlink(missing_ok=True)
    version = subprocess.run([str(PG_BIN / "postgres.exe"), "--version"], capture_output=True, text=True).stdout.strip()
    say(version or "PostgreSQL installed")
    marker.write_text(wanted, encoding="utf-8")


def prepare_database(cfg: Config) -> None:
    if (cfg.pgdata / "PG_VERSION").exists():
        say(f"database cluster: {cfg.pgdata}")
        return
    say(f"creating the database cluster in {cfg.pgdata}")
    cfg.pgdata.parent.mkdir(parents=True, exist_ok=True)
    # The services' user owns the cluster and logs in over TCP with the password from .env
    pwfile = RUNTIME / "pw.tmp"
    pwfile.write_text(cfg.db_password + "\n", encoding="utf-8")
    try:
        run([PG_BIN / "initdb.exe", "-D", cfg.pgdata, "-U", cfg.db_user, f"--pwfile={pwfile}",
             "--auth=scram-sha-256", "--encoding=UTF8", "--locale=C"], stdout=subprocess.DEVNULL)
    finally:
        pwfile.unlink(missing_ok=True)


def psql(cfg: Config, database: str, sql: str) -> str:
    env = {**os.environ, "PGPASSWORD": cfg.db_password}
    return subprocess.run([str(PG_BIN / "psql.exe"), "-h", "127.0.0.1", "-p", str(cfg.ports["postgres"]), "-U", cfg.db_user,
                           "-d", database, "-v", "ON_ERROR_STOP=1", "-Atqc", sql],
                          capture_output=True, text=True, check=True, env=env, creationflags=CREATE_NO_WINDOW).stdout


def ensure_database(cfg: Config) -> None:
    """The database and the vector extension; the services apply schema.sql themselves."""
    for _ in range(60):
        if probe(None, cfg):
            break
        time.sleep(1)
    user = cfg.db_user.replace('"', '""')
    exists = f"SELECT 1 FROM pg_database WHERE datname = '{cfg.db_name}'"
    if not psql(cfg, "postgres", exists).strip():
        try:
            psql(cfg, "postgres", f"CREATE DATABASE \"{cfg.db_name}\" OWNER \"{user}\"")
        except subprocess.CalledProcessError:
            # The supervisor creates it too when it starts postgres; one of the two loses
            if not psql(cfg, "postgres", exists).strip():
                raise
    psql(cfg, cfg.db_name, "CREATE EXTENSION IF NOT EXISTS vector")


# ── virtualenvs, sources, layout ─────────────────────────────────────────────

def mmaudio_packages(source: Path) -> list[str]:
    """What MMAudio's pyproject asks for, without torchcodec (unused by the API; no build
    for this torch on Windows)."""
    data = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))
    packages = []
    for dep in data["project"]["dependencies"]:
        name = re.split(r"[\s<>=!~\[;]", dep.strip(), 1)[0].lower()
        if name in ("torchcodec", "torch"):
            continue
        packages.append(dep.strip())
    return packages


def venv_packages() -> dict[str, list[str]]:
    """Per virtualenv what its Dockerfile installs; torch as in the images (2.5.1, CUDA 12.4)."""
    torch = ["torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1"]
    return {
        "mmaudio": torch + pip_lines(REPO / "generation/mmaudio/requirements.txt"),
        # The search service, the indexer and the setup page share one image base
        "search": pip_lines(REPO / "search/api/requirements.txt") + [
            "transformers>=4.40,<5", "psycopg2-binary>=2.9", "sentencepiece>=0.1.99", "mutagen>=1.47",
            "fastapi>=0.110", "uvicorn[standard]>=0.27", "httpx>=0.27"],
        "gateway": pip_lines(REPO / "generation/gateway/requirements.txt"),
        "spotting": pip_lines(REPO / "spotting/api/requirements.txt") + ["imageio-ffmpeg>=0.5"],
        "hybrid": pip_lines(REPO / "hybrid/api/requirements.txt"),
    }


def fetch_mmaudio_source() -> Path:
    ref = dockerfile_arg(REPO / "generation/mmaudio/Dockerfile", "MMAUDIO_REF")
    repo = dockerfile_arg(REPO / "generation/mmaudio/Dockerfile", "MMAUDIO_REPO")
    source = RUNTIME / "src/MMAudio"
    marker = source / ".fetched-ref"
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == ref:
        return source
    say(f"fetching the MMAudio source ({ref[:7]})")
    status_line("Downloading the MMAudio source")
    path = repo.replace("https://github.com/", "").removesuffix(".git")
    archive = RUNTIME / "downloads/mmaudio.zip"
    download(f"https://codeload.github.com/{path}/zip/{ref}", archive, "MMAudio source")
    if source.exists():
        shutil.rmtree(source)
    with zipfile.ZipFile(archive) as z:
        top = z.namelist()[0].split("/")[0]
        z.extractall(RUNTIME / "src")
    (RUNTIME / "src" / top).rename(source)
    archive.unlink(missing_ok=True)
    marker.write_text(ref + "\n", encoding="utf-8")
    return source


def prepare_venvs() -> None:
    source = fetch_mmaudio_source()
    ref = (source / ".fetched-ref").read_text(encoding="utf-8").strip()
    for name, packages in venv_packages().items():
        if name == "mmaudio":
            packages = packages + mmaudio_packages(source)
        venv = VENVS / name
        stamp = venv / ".installed"
        # The torch pins hold for every later install into the same virtualenv: MMAudio
        # asks only for torch>=2.5.1, and pip once resolved its dependencies to torch 2.15,
        # whose libraries torchaudio 2.5.1 can no longer load
        pins = [p for p in packages if re.match(r"(torch|torchvision|torchaudio)==", p)]
        wanted = hashlib.sha256(json.dumps([packages, ref if name == "mmaudio" else "", "cu124"]).encode()).hexdigest()
        if stamp.exists() and stamp.read_text(encoding="utf-8") == wanted:
            say(f"virtualenv {name}: up to date")
            continue
        say(f"virtualenv {name}: installing" + (" (the first time downloads PyTorch for CUDA, about 2.5 GB)" if pins else ""))
        status_line(f"Python environment {name}" + (": downloading PyTorch for CUDA" if pins else ""))
        if not venv_python(name).exists():
            run([PYTHON, "-m", "venv", venv])
        pip = [venv_python(name), "-m", "pip", "install", "-q", "--disable-pip-version-check"]
        run(pip + ["--upgrade", "pip"])
        if pins:
            constraints = venv / "constraints.txt"
            constraints.write_text("\n".join(pins) + "\n", encoding="utf-8")
            # torch and its two companions from the CUDA wheels; PyPI's are CPU-only on Windows
            run(pip + ["--index-url", TORCH_INDEX] + pins)
            pip += ["-c", constraints]
        rest = [p for p in packages if p not in pins]
        if rest:
            run(pip + rest)
        if name == "mmaudio":
            run(pip + ["--no-deps", "-e", source])
        if any(p.startswith("torchaudio==") for p in pins):
            # Fails here, not later in the service, if torch and torchaudio do not fit
            run([venv_python(name), "-c", "import torch, torchaudio"])
        stamp.write_text(wanted, encoding="utf-8")


def prepare_webui(cfg: Config) -> None:
    """The web UI: its source at WEBUI_REF (fetched from GitHub when missing or when GitHub
    has a newer commit), the frontend built with Node (only when the source changed), a
    virtualenv for its session server."""
    tools = Tools()
    if not tools.npm or not tools.node:
        raise SystemExit("Node.js is missing (install_windows.ps1 installs it); the web UI cannot be built.")
    record_path = WEBUI_SRC / ".windows/installed-from"
    record = {}
    if record_path.exists():
        for line in record_path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                record[k.strip()] = v.strip()
    have = record.get("commit", "")
    latest = http_json(f"https://api.github.com/repos/{cfg.webui_repo}/commits/{cfg.webui_ref}")
    sha = latest.get("sha", "") if latest else ""
    if not (WEBUI_SRC / "package.json").exists() or (sha and sha != have and os.environ.get("AISD_WEBUI_UPDATE", "1") == "1"):
        if not sha:
            raise SystemExit("GitHub could not be reached to fetch the web UI. Is this PC online?")
        say(f"web UI: fetching {cfg.webui_repo} ({sha[:7]})")
        status_line("Downloading the web UI from GitHub")
        archive = RUNTIME / "downloads/webui.zip"
        download(f"https://codeload.github.com/{cfg.webui_repo}/zip/{sha}", archive, "web UI source")
        staging = RUNTIME / "downloads/webui-src"
        if staging.exists():
            shutil.rmtree(staging)
        with zipfile.ZipFile(archive) as z:
            z.extractall(staging)
        top = next(p for p in staging.iterdir() if p.is_dir())
        WEBUI_SRC.mkdir(parents=True, exist_ok=True)
        for path in top.rglob("*"):
            if path.is_dir():
                continue
            rel = path.relative_to(top)
            if rel.parts[0] in (".git",) or rel.name == ".env":
                continue
            copy_file(path, WEBUI_SRC / rel)
        shutil.rmtree(staging, ignore_errors=True)
        archive.unlink(missing_ok=True)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(f"origin={cfg.webui_repo}\nref={cfg.webui_ref}\ncommit={sha}\nmodified=0\n"
                               f"installed={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\nmode=local\n", encoding="utf-8")
        have = sha
    say(f"web UI: {have[:7] or 'local copy'} ({cfg.webui_ref})")

    built = WEBUI_SRC / "dist/.built-from"
    if not (built.exists() and built.read_text(encoding="utf-8") == have and (WEBUI_SRC / "dist/index.html").exists()):
        say("web UI: building the frontend (npm ci, npm run build)")
        status_line("Building the web UI")
        env = {**os.environ, "PATH": ";".join([str(tools.node.parent), registry_path()])}
        run([tools.npm, "ci", "--no-audit", "--no-fund", "--loglevel=error"], cwd=WEBUI_SRC, env=env)
        run([tools.npm, "run", "build"], cwd=WEBUI_SRC, env=env, stdout=subprocess.DEVNULL)
        built.parent.mkdir(parents=True, exist_ok=True)
        built.write_text(have, encoding="utf-8")

    requirements = WEBUI_SRC / "server/requirements.txt"
    venv = VENVS / "webui"
    stamp = venv / ".installed"
    wanted = hashlib.sha256(requirements.read_bytes()).hexdigest()
    if not (stamp.exists() and stamp.read_text(encoding="utf-8") == wanted):
        say("virtualenv webui: installing")
        if not venv_python("webui").exists():
            run([PYTHON, "-m", "venv", venv])
        pip = [venv_python("webui"), "-m", "pip", "install", "-q", "--disable-pip-version-check"]
        run(pip + ["--upgrade", "pip"])
        run(pip + ["-r", requirements])
        stamp.write_text(wanted, encoding="utf-8")
    else:
        say("virtualenv webui: up to date")
    for path in cfg.webui_sessions:
        if path == cfg.data / "sessions":
            path.mkdir(parents=True, exist_ok=True)
        elif not path.is_dir():
            say(f"web UI: the sessions folder {path} does not exist (WEBUI_SESSIONS_DIRS in .env)")


def prepare_layout(cfg: Config) -> None:
    """The services' folders laid out as in their images, as copies of this repository's
    files (refreshed at every start, so an edit in the repository is what runs next)."""
    say("laying out the services' folders")
    mm = APPS / "mmaudio"
    copy_tree(REPO / "shared", mm / "shared")
    for script in ("main.py", "prefetch.py", "mps_compat.py"):
        copy_file(REPO / "generation/mmaudio" / script, mm / "api" / script)
    # MMAudio resolves its checkpoints relative to its working directory (as entrypoint.sh links them)
    for sub in ("weights", "ext_weights"):
        (cfg.models / "mmaudio" / sub).mkdir(parents=True, exist_ok=True)
        junction(cfg.models / "mmaudio" / sub, mm / "api" / sub)

    search = APPS / "search"
    copy_file(REPO / "search/api/main.py", search / "main.py")
    copy_tree(REPO / "search/api/utils", search / "utils")
    copy_tree(REPO / "search/common", search / "common")
    copy_file(REPO / "search/schema.sql", search / "common/schema.sql")

    setup = APPS / "setup"
    for name in ("main.py", "page.html"):
        copy_file(REPO / "setup" / name, setup / name)
    for script in (REPO / "search/indexer").glob("*.py"):
        copy_file(script, setup / script.name)
    copy_tree(REPO / "search/sources", setup / "sources")
    copy_tree(REPO / "search/common", setup / "common")
    copy_file(REPO / "search/schema.sql", setup / "common/schema.sql")

    for path in (cfg.audio, cfg.libraries, cfg.archive, cfg.own_sounds, cfg.hybrid_output, cfg.webui_cache,
                 cfg.models / "huggingface", cfg.models / "torch", LOGS / "setup", RUNTIME / "work/gateway", PIDS):
        path.mkdir(parents=True, exist_ok=True)


# ── the supervisor: the services as background processes, started at logon ──

def pid_file(name: str) -> Path:
    return PIDS / f"{name}.pid"


def wanted_file() -> Path:
    return RUNTIME / "services.json"


def read_wanted() -> list[str]:
    try:
        return [n for n in json.loads(wanted_file().read_text(encoding="utf-8")) if n in ORDER]
    except (OSError, ValueError):
        return []


def write_wanted(names: list[str]) -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    wanted_file().write_text(json.dumps([n for n in ORDER if n in names]), encoding="utf-8")


def pid_alive(pid: int) -> bool:
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    code = ctypes.c_ulong()
    alive = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code)) and code.value == 259
    ctypes.windll.kernel32.CloseHandle(handle)
    return bool(alive)


def pid_of(name: str) -> int | None:
    try:
        pid = int(pid_file(name).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return pid if pid_alive(pid) else None


def kill_tree(pid: int) -> None:
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, creationflags=CREATE_NO_WINDOW)


def stop_service(name: str, cfg: Config | None = None) -> bool:
    pid = pid_of(name)
    if pid is None:
        return False
    if name == "postgres" and cfg is not None:
        subprocess.run([str(PG_BIN / "pg_ctl.exe"), "stop", "-D", str(cfg.pgdata), "-m", "fast", "-w", "-t", "30"],
                       capture_output=True, creationflags=CREATE_NO_WINDOW)
    if pid_alive(pid):
        kill_tree(pid)
    pid_file(name).unlink(missing_ok=True)
    return True


def spawn_service(name: str) -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    PIDS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{name}.log"
    if log.exists() and log.stat().st_size > 20 << 20:
        log.rename(log.with_suffix(".old.log"))
    with log.open("ab") as out:
        out.write(f"\n== {time.strftime('%Y-%m-%d %H:%M:%S')} started\n".encode())
        # python.exe, not pythonw.exe: the service it runs is a console program and would get a
        # visible console of its own from a parent without one
        process = subprocess.Popen([str(PYTHON), str(Path(__file__).resolve()), "run", name], cwd=str(REPO),
                                   stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                   creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP)
    pid_file(name).write_text(str(process.pid), encoding="utf-8")
    return process.pid


def supervise() -> None:
    """Keeps the services in services.json running: starts the missing ones, restarts the
    ones that stopped (15 s later), stops the ones no longer wanted. Ends when the stop
    file appears. One instance: its pid is in pids\\supervisor.pid."""
    PIDS.mkdir(parents=True, exist_ok=True)
    stop_file = RUNTIME / "stop"
    stop_file.unlink(missing_ok=True)
    own = pid_file("supervisor")
    own.write_text(str(os.getpid()), encoding="utf-8")
    cfg = Config()
    started: dict[str, float] = {}
    try:
        while not stop_file.exists():
            cfg = Config() if not started else cfg
            wanted = read_wanted()
            for name in ORDER:
                running = pid_of(name) is not None
                if name in wanted and not running:
                    if time.time() - started.get(name, 0) < 15:
                        continue
                    port = cfg.ports.get(name)
                    if name not in started and port and port_busy(port):
                        started[name] = time.time()
                        continue    # another program (a Docker stack?) holds the port: tried again later
                    spawn_service(name)
                    started[name] = time.time()
                    if name == "postgres":
                        try:
                            ensure_database(cfg)
                        except Exception:  # noqa: BLE001
                            pass
                elif name not in wanted and running:
                    stop_service(name, cfg)
            time.sleep(5)
    finally:
        for name in reversed(ORDER):
            stop_service(name, cfg)
        own.unlink(missing_ok=True)
        stop_file.unlink(missing_ok=True)


def supervisor_running() -> bool:
    return pid_of("supervisor") is not None


def start_supervisor() -> None:
    if supervisor_running():
        return
    (RUNTIME / "stop").unlink(missing_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / "supervisor.log").open("ab") as out:
        subprocess.Popen([str(PYTHONW if PYTHONW.exists() else PYTHON), str(Path(__file__).resolve()), "supervise"],
                         cwd=str(REPO), stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, close_fds=True,
                         creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW)
    for _ in range(20):
        if supervisor_running():
            return
        time.sleep(0.5)
    raise SystemExit(f"the supervisor did not start; see {LOGS / 'supervisor.log'}")


def powershell(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script],
                          capture_output=True, text=True, creationflags=CREATE_NO_WINDOW)


def write_logon_task() -> None:
    """A task of this Windows account that starts the supervisor at every logon. Through the
    ScheduledTasks cmdlets: schtasks.exe refuses /RU for an account without elevation. No
    time limit (the default would end the supervisor after three days), also on battery."""
    python = PYTHONW if PYTHONW.exists() else PYTHON
    backend = Path(__file__).resolve()
    QUOTE = chr(34)  # a double quote inside the single-quoted PowerShell string
    script = "; ".join([
        f"$action = New-ScheduledTaskAction -Execute '{python}' -Argument '{QUOTE}{backend}{QUOTE} supervise'",
        "$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME",
        "$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited",
        "$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries "
        "-ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew",
        f"Register-ScheduledTask -TaskName '{TASK_NAME}' -Action $action -Trigger $trigger -Principal $principal "
        "-Settings $settings -Force | Out-Null"])
    result = powershell(script)
    if result.returncode != 0:
        say(f"the logon task could not be created ({result.stderr.strip() or result.stdout.strip()}); "
            f"start the backend with 'Start backend' after a restart")


def remove_logon_task() -> None:
    powershell(f"Unregister-ScheduledTask -TaskName '{TASK_NAME}' -Confirm:$false -ErrorAction SilentlyContinue")


def start(names: list[str], cfg: Config) -> None:
    wanted = read_wanted()
    for name in names:
        if name not in wanted:
            wanted.append(name)
    write_wanted(wanted)
    start_supervisor()
    for name in names:
        port = cfg.ports.get(name)
        # The supervisor starts it within moments; a port that is busy before it got to it is
        # the service itself coming up, not another program
        for _ in range(10):
            if pid_of(name) or not (port and port_busy(port)):
                break
            time.sleep(0.5)
        if pid_of(name):
            say(f"{name}: already running")
        elif port and port_busy(port):
            say(f"{name}: port {port} is taken by another program (a Docker stack?); not started")
        else:
            say(f"{name}: starting (log: {LOGS / (name + '.log')})")
    if "postgres" in names:
        for _ in range(60):
            if probe(None, cfg):
                break
            time.sleep(1)
        ensure_database(cfg)


def stop(names: list[str], cfg: Config, keep_supervisor: bool = True) -> None:
    wanted = [n for n in read_wanted() if n not in names]
    write_wanted(wanted)
    for name in reversed(names):
        if stop_service(name, cfg):
            say(f"{name}: stopped")
    if not wanted or not keep_supervisor:
        (RUNTIME / "stop").touch()
        for _ in range(20):
            if not supervisor_running():
                break
            time.sleep(0.5)
        pid = pid_of("supervisor")
        if pid:
            kill_tree(pid)
            pid_file("supervisor").unlink(missing_ok=True)


def status(cfg: Config) -> int:
    print(f"{'service':<10} {'pid':>7}  {'answers':<8} where")
    bad = 0
    for name in wanted_services(cfg):
        ok = probe(health_url(name, cfg), cfg)
        bad += 0 if ok else 1
        print(f"{name:<10} {pid_of(name) or '-':>7}  {'yes' if ok else 'no':<8} localhost:{cfg.ports[name]}")
    print(f"supervisor {pid_of('supervisor') or '-':>7}  {'':<8} {'logon task: ' + TASK_NAME}")
    print(f"(spotting uses the vision-language model at {cfg.vlm_url})")
    print(f"\nsetup page: http://localhost:{cfg.ports['setup']}/    logs: {LOGS}")
    return 1 if bad else 0


def run_service(name: str) -> None:
    cfg = Config()
    argv, cwd, env = service_spec(name, cfg)
    # The service writes into this wrapper's log file, and gets no console of its own: handles
    # not passed on explicitly would leave it talking to the hidden console, which Windows
    # Terminal then shows as a window
    quiet = dict(stdin=subprocess.DEVNULL, stdout=sys.stdout, stderr=subprocess.STDOUT, creationflags=CREATE_NO_WINDOW)
    if name == "mmaudio" and env.get("MMAUDIO_PREFETCH") == "1":
        # As the image's entrypoint: the weights are on disk before the server starts,
        # or the service does not start at all (the supervisor tries again)
        result = subprocess.run([argv[0], "prefetch.py"], cwd=cwd, env=env, **quiet)
        if result.returncode != 0:
            sys.exit(result.returncode)
    sys.exit(subprocess.run(argv, cwd=cwd, env=env, **quiet).returncode)


# ── install: configuration, models, first start ──────────────────────────────

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
    the first run, the folders given as parameters or asked for."""
    env_path = REPO / ".env"
    fresh = not env_path.exists()
    if fresh:
        shutil.copy(REPO / ".env.example", env_path)
        set_env_value("POSTGRES_PASSWORD", hashlib.sha256(os.urandom(32)).hexdigest()[:24])
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
        say("The web UI opens Pro Tools sessions (.ptx) from a folder you choose, read-only.")
        options["sessions"] = ask_folder("   Folder with your Pro Tools sessions, Enter for <data folder>\\sessions: ",
                                         must_exist=True)
    if options.get("webui") in ("on", "off"):
        set_env_value("WINDOWS_WEBUI", "1" if options["webui"] == "on" else "0")
    if options.get("sessions"):
        sessions = Path(options["sessions"]).expanduser().resolve()
        if not sessions.is_dir():
            raise SystemExit(f"The sessions folder does not exist: {sessions}")
        set_env_value("WEBUI_SESSIONS_DIRS", str(sessions))
        say(f"web UI sessions: {sessions}")
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
    # The vision-language model: the Ollama installed on Windows, not a container
    env = read_env()
    if re.match(r"^https?://ollama(:\d+)?/?$", env.get("VLM_URL", "")) or "host.docker.internal" in env.get("VLM_URL", ""):
        set_env_value("VLM_URL", "http://127.0.0.1:11434")
    set_env_value("BUNDLED_OLLAMA", "0")


def ollama_store_setting(cfg: Config) -> bool:
    """OLLAMA_MODELS_STORE from .env as OLLAMA_MODELS for this Windows account: the Ollama
    server reads it when it starts (the pull runs in the server, so an environment variable
    of the pull command alone would not move the model). True when it changed; a store the
    user set for themselves is left alone when .env names none."""
    if not cfg.ollama_models_custom:
        return False
    wanted = str(cfg.ollama_models)
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_READ | winreg.KEY_SET_VALUE) as key:
        try:
            current = str(winreg.QueryValueEx(key, "OLLAMA_MODELS")[0])
        except FileNotFoundError:
            current = ""
        if current.rstrip("\\").lower() == wanted.rstrip("\\").lower():
            return False
        winreg.SetValueEx(key, "OLLAMA_MODELS", 0, winreg.REG_SZ, wanted)
    # Programs started from now on (Explorer, the Start menu) see the new value
    ctypes.windll.user32.SendMessageTimeoutW(0xFFFF, 0x001A, 0, "Environment", 0x0002, 5000, None)
    say(f"OLLAMA_MODELS={wanted} set for this Windows account")
    return True


def ensure_ollama(cfg: Config) -> None:
    """The Ollama installed on Windows answers, and holds the model."""
    tools = Tools()
    if not cfg.ollama_local:
        say(f"the vision-language model comes from {cfg.vlm_url} (VLM_URL)")
        return
    if not tools.ollama:
        raise SystemExit("Ollama is not installed (install_windows.ps1 installs it): https://ollama.com/download/windows")
    if ollama_store_setting(cfg) and probe(cfg.vlm_url + "/api/version", cfg):
        say("restarting Ollama so it uses that store")
        subprocess.run(["taskkill", "/IM", "ollama app.exe", "/IM", "ollama.exe", "/F"], capture_output=True, creationflags=CREATE_NO_WINDOW)
        time.sleep(2)
    if not probe(cfg.vlm_url + "/api/version", cfg):
        app = tools.ollama.with_name("ollama app.exe")
        if app.exists():
            subprocess.Popen([str(app)], creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP, close_fds=True)
        else:
            subprocess.Popen([str(tools.ollama), "serve"], creationflags=DETACHED_PROCESS | CREATE_NO_WINDOW, close_fds=True)
        for _ in range(30):
            if probe(cfg.vlm_url + "/api/version", cfg):
                break
            time.sleep(2)
        if not probe(cfg.vlm_url + "/api/version", cfg):
            raise SystemExit(f"Ollama does not answer at {cfg.vlm_url}; start it from the Start menu and run this again.")
    say(f"Ollama answers at {cfg.vlm_url}")
    status_line(f"Fetching the vision-language model {cfg.vlm_model}")
    run([tools.ollama, "pull", cfg.vlm_model])


def prefetch_mmaudio(cfg: Config) -> None:
    """MMAudio's weights (about 10 GB) with the download shown here rather than in a log."""
    argv, cwd, env = service_spec("mmaudio", cfg)
    status_line("Fetching the MMAudio weights")
    run([argv[0], "prefetch.py"], cwd=cwd, env=env)


def write_shortcuts(cfg: Config) -> Path:
    """Next to the installation and in the Start menu: Start backend, Stop backend,
    Backend status, Setup page, Web UI."""
    home = REPO.parent
    python = f'"{PYTHON}" "{Path(__file__).resolve()}"'
    files = {
        "Start backend.cmd": f"@echo off\r\n{python} start\r\npause\r\n",
        "Stop backend.cmd": f"@echo off\r\n{python} stop\r\npause\r\n",
        "Backend status.cmd": f"@echo off\r\n{python} status\r\npause\r\n",
        "Setup page.url": f"[InternetShortcut]\r\nURL=http://localhost:{cfg.ports['setup']}/\r\n",
    }
    if cfg.webui:
        files["Web UI.url"] = f"[InternetShortcut]\r\nURL=http://localhost:{cfg.ports['webui']}/\r\n"
    menu = Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs/AI Sound Design"
    for target in (home, menu):
        try:
            target.mkdir(parents=True, exist_ok=True)
            for name, content in files.items():
                (target / name).write_text(content, encoding="utf-8")
            if not cfg.webui:
                (target / "Web UI.url").unlink(missing_ok=True)
        except OSError:
            pass
    return home


def install(options: dict[str, str], interactive: bool, only: str = "") -> int:
    def step(text: str) -> None:
        print(f"\n== {text}", flush=True)

    step("Configuration")
    configure(options, interactive)
    cfg = Config()
    if only == "webui":
        step("Web UI (the four modes in the browser)")
        cfg = Config()
        if not cfg.webui:
            set_env_value("WINDOWS_WEBUI", "1")
            cfg = Config()
        prepare_webui(cfg)
        (cfg.webui_cache).mkdir(parents=True, exist_ok=True)
        PIDS.mkdir(parents=True, exist_ok=True)
        if pid_of("webui"):
            stop(["webui"], cfg)
            time.sleep(1)
        start(["webui"], cfg)
        write_logon_task()
        write_shortcuts(cfg)
        late = wait_for(["webui"], cfg, minutes=2)
        say(f"web UI: http://localhost:{cfg.ports['webui']}/" + (" (not answering yet; see the log)" if late else ""))
        return 0

    free = shutil.disk_usage(cfg.data if cfg.data.exists() else REPO).free // 2**30
    say(f"{free} GB free on the data folder's drive" + ("" if free >= 90 else
        ": the models and the index take about 40 GB, a downloaded BBC archive another 60 GB"))

    step("PostgreSQL 17 with pgvector")
    prepare_postgres()

    step("Python environments, MMAudio source, folders (the first time several GB of PyTorch)")
    prepare_venvs()
    prepare_layout(cfg)
    prepare_database(cfg)

    if cfg.webui:
        step("Web UI (the four modes in the browser)")
        prepare_webui(cfg)
    elif "webui" in read_wanted():
        stop(["webui"], cfg)
        say("web UI: switched off (WINDOWS_WEBUI=0)")

    step("Database")
    start(["postgres"], cfg)

    step(f"Vision-language model for spotting ({cfg.vlm_model}, about 6 GB the first time)")
    ensure_ollama(cfg)

    step("MMAudio weights (about 10 GB the first time)")
    prefetch_mmaudio(cfg)

    step("Starting the services")
    # A service that already runs is restarted: after an update it has to load the new code
    running = [n for n in APP_SERVICES if n in wanted_services(cfg) and pid_of(n)]
    if running:
        stop(running, cfg)
        time.sleep(2)
    start(wanted_services(cfg), cfg)
    write_logon_task()
    say("waiting for them to answer (MMAudio loads its model onto the GPU first: a minute or two)")
    late = wait_for(wanted_services(cfg), cfg, minutes=6)
    for name in late:
        say(f"{name} does not answer yet; see: python windows\\backend.py logs {name}")

    step("Shortcuts")
    home = write_shortcuts(cfg)
    say(f"in {home} and the Start menu: " + ("Web UI, " if cfg.webui else "") +
        "Start backend, Stop backend, Backend status, Setup page")

    step("Done")
    say(f"setup page: http://localhost:{cfg.ports['setup']}/ (the services' state, the sound index)")
    if cfg.webui:
        webui = f"http://localhost:{cfg.ports['webui']}/"
        say(f"web UI:     {webui}")
        if "webui" not in late and not GUI and not options.get("no_browser"):
            os.startfile(webui)
    say("The services start at every logon; 'Stop backend' stops them until the next logon or 'Start backend'.")
    return 0


def database(cfg: Config, audio: bool) -> int:
    """Fills the sound index as "Set everything up" on the setup page does; the setup
    service does the work and this shows its progress (see the Mac's backend.py)."""
    late = wait_for(["postgres", "search", "setup"], cfg, minutes=5)
    if late:
        say(f"{', '.join(late)} does not answer; see: python windows\\backend.py status")
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
    wanted = ["bbc_metadata", "text_embeddings"] + (["bbc_archive"] if archive else ["bbc_audio"] if audio else []) \
        + ["audio_index"]
    if audio and not archive and steps.get("bbc_audio", {}).get("status") != "complete":
        free = shutil.disk_usage(cfg.audio if cfg.audio.exists() else cfg.data).free // 2**30
        if free < 70:
            say(f"The BBC audio takes about 60 GB, and {cfg.audio.parent} has {free} GB free: not started.")
            if GUI:
                print(f"@@error The BBC audio needs about 60 GB; {free} GB are free. Free some room and try again.", flush=True)
            return 1
    say("BBC Sound Effects: free for personal, educational and research use under the BBC's terms")
    say("(https://sound-effects.bbcrewind.co.uk/licensing).")
    if GUI:
        print("@@status Starting the sound database", flush=True)
    call("/pipeline/start", {"download": audio})
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
                    print(f"   + {s['title']}: {s['detail']}", flush=True)
                    line = ""
            running = [n for n in wanted if n in state["running"]]
            if running:
                s = steps[running[0]]
                share = f"{100 * s['done'] / s['total']:.0f}%  " if s["total"] else ""
                text = f"   ... {s['title']}: {share}{s['detail']}"[:150]
                if text != line:
                    if GUI:
                        print(f"@@status {s['title']}: {s['detail']}", flush=True)
                        print(f"@@progress {100 * s['done'] / s['total']:.1f}" if s["total"] else "@@progress -", flush=True)
                    else:
                        print(f"\r{text:<150}", end="", flush=True)
                    line = text
                continue
            if state["pipeline"].get("active"):
                continue
            failed = [n for n in wanted if steps.get(n, {}).get("status") == "error"]
            retry = [n for n in failed if n not in retried]
            if retry:
                print(f"\n   {steps[retry[0]]['title']} had failed before; trying it again", flush=True)
                retried.add(retry[0])
                call(f"/jobs/{retry[0]}/start", {})
                call("/pipeline/start", {"download": audio})
                continue
            if line and not GUI:
                print()
            if failed:
                for name in failed:
                    s = steps[name]
                    say(f"{s['title']} failed: {s.get('error') or ''}")
                    if GUI:
                        print(f"@@error {s['title']} failed; Try Again carries on where it stopped.", flush=True)
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
    if command == "supervise":
        supervise()
        return 0
    if command in ("install", "webui"):
        options, flags = {}, {"--data-dir": "data_dir", "--bbc-archive": "bbc_archive",
                              "--own-sounds": "own_sounds", "--models-dir": "models_dir",
                              "--sessions": "sessions", "--webui": "webui"}
        i = 0
        while i < len(rest):
            if rest[i] in flags and i + 1 < len(rest):
                options[flags[rest[i]]] = rest[i + 1]
                i += 2
            elif rest[i] in ("--no-prompt", "--no-browser"):
                options[rest[i].strip("-").replace("-", "_")] = "1"
                i += 1
            else:
                raise SystemExit(f"unknown option: {rest[i]}")
        if options.get("webui", "on") not in ("on", "off"):
            raise SystemExit("--webui on|off")
        interactive = "--no-prompt" not in rest and sys.stdin.isatty()
        return install(options, interactive, only="webui" if command == "webui" else "")
    cfg = Config()
    if command == "database":
        if [a for a in rest if a != "--audio"]:
            raise SystemExit("database [--audio]")
        return database(cfg, audio="--audio" in rest)
    unknown = [n for n in rest if n not in ORDER and not n.isdigit()]
    if unknown and command != "logs":
        raise SystemExit(f"unknown service(s): {', '.join(unknown)}; known: {', '.join(ORDER)}")
    names = [n for n in ORDER if n in rest] or wanted_services(cfg)
    if command == "prepare":
        prepare_postgres()
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
        stop(names, cfg, keep_supervisor=bool([n for n in rest if n in ORDER]))
        return 0
    if command == "restart":
        stop(names, cfg)
        time.sleep(2)
        prepare_layout(cfg)
        start(names, cfg)
        return 0
    if command == "status":
        return status(cfg)
    if command == "logs":
        if not rest:
            raise SystemExit("logs <service> [lines]")
        lines = int(rest[1]) if len(rest) > 1 else 60
        log = LOGS / f"{rest[0]}.log"
        if log.exists():
            print("".join(log.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)[-lines:]))
        return 0
    if command == "uninstall":
        stop(ORDER, cfg, keep_supervisor=False)
        remove_logon_task()
        say("services stopped, logon task removed; the data folder and .windows\\ are left in place")
        return 0
    raise SystemExit(f"unknown command: {command}\n{__doc__}")


if __name__ == "__main__":
    sys.exit(main())
