# ai-services

An example backend for the [AI Sound Design Pro Tools plugin](https://github.com/ai-sound-design/ai-protools-aax).

The plugin adds four AI workflows to Pro Tools: generating sound for a video
clip, recommending sounds from a searchable library, spotting sound events as
memory locations, and a hybrid of the two that returns one generated sound per
sound event. It does none of that itself; it sends the work to HTTP services. This repository is one complete set of such services, packaged
so that a single `docker compose up` brings it up on a machine with an NVIDIA
GPU, and `install_mac.sh` runs the same services natively on an Apple Silicon
Mac. It is also the reference for writing your own: every service folder
documents the contract the plugin relies on, and a service that fulfils it can
replace the one here without a change to the plugin.

All workflows are fully functional here. This differs from the prototype
used in the user study, where the spotting was operated Wizard-of-Oz style by
the experimenters; in this backend the spotting service runs a vision-language
model.

## Demo

The plugin's four modes running on this backend, recorded in Pro Tools:
Spotting, Sound Recommendation, Sound Generation and Hybrid. All four videos,
with what each mode does, are in the
[plugin's README](https://github.com/ai-sound-design/ai-protools-aax#modes).
The Hybrid run uses all four services together:

https://github.com/user-attachments/assets/0d8f86b1-ce94-4fe8-9c39-b86115dfc923

## How it works

```
Pro Tools ── AAX plugin ── bundled Python scripts ──HTTP──▶ generation service   (:8000)
                                                   ──HTTP──▶ search service       (:8002)
                                                   ──HTTP──▶ spotting service     (:8003)
                                                   ──HTTP──▶ hybrid service       (:8004)
```

1. The user marks a time range on any track. The plugin finds the video clips
   beneath it and cuts each clip's part to a small video file.
2. Depending on the mode, the plugin sends that video (and an optional prompt)
   to one of the services and waits for the answer: an audio file, a list of
   matching sounds, a list of sound events with timecodes, or one sound per
   event.
3. The plugin puts the result back into the session: the audio on its own
   track at the clip's position, the chosen sound from the list, one memory
   location per event, or one new track per generated sound, with the
   library recordings found for it on a track underneath when asked.

While a spotting or hybrid request runs, the plugin polls the service's
progress endpoint (`/spot/progress/{job_id}`, `/hybrid/progress/{job_id}`)
and shows what it reports; the endpoints are optional, a service without
them gets a busy bar.

Which service the plugin talks to is decided by **adapter profiles**: one
small JSON file per backend in the plugin's adapters folder
(`%APPDATA%\AI Sound Design\adapters\` on Windows, `~/Library/AI Sound Design/adapters/`
on macOS). A profile names the address, the health endpoint and, for
generation, how the request is built. The profiles the plugin ships with point
at the ports above on `localhost`, so this backend on the same machine works
without any configuration. Any other address, machine or model is a new
profile, not a new plugin build.

The four kinds of service, one folder each:

| Folder | Service | Port | What it does |
|--------|---------|------|--------------|
| [`generation/`](generation/README.md) | `mmaudio-api` | 8000 | Video (or prompt) in, one window of audio out (4-12 s). [MMAudio](https://github.com/hkchengrex/MMAudio) as the reference model; `template/` for wrapping another one. |
| [`generation/gateway/`](generation/gateway/README.md) | `generation-gateway` | 8010 | The same contract without a length limit: long requests in windows with crossfades, short ones padded, in front of the model service. The plugin and the hybrid service talk to this. |
| [`search/`](search/README.md) | `sound-search-api` | 8002 | Video or prompt in, matching sounds out. X-CLIP embeddings over any indexed library: your own folders of sounds, the BBC Sound Effects Archive, or both. Also sound in, matching recordings out, located to the second: CLAP embeddings of the files on disk, 2 s windows merged into segments where they sound alike. |
| [`spotting/`](spotting/README.md) | `spotting-api` | 8003 | Video range in, sound events with timecodes out. A vision-language model describes the sounds a silent picture implies. |
| [`hybrid/`](hybrid/README.md) | `hybrid-api` | 8004 | Video range in, one generated sound per sound event out, each with its own start and end; on request also library recordings for each one: the event's description proposes candidates, the generated sound picks among them and the stretch inside each, stitched from pieces. The reference implementation composes the services above. |

Behind them: `postgres` (the sound index, pgvector), `ollama` (hosts the
vision-language model; optional, see below) and `indexer` (one-shot jobs that
fill the search index).

Everything that grows lives outside the images: model weights in named Docker
volumes, the sound index in a volume, fetched audio and your own libraries in
plain folders under `./data/`, so rebuilding an image never downloads a model
twice.

## Installation

### Requirements

- Docker with Compose v2: Docker Engine on Linux, Docker Desktop on Windows.
  (On a Mac, Docker cannot reach the GPU; see [macOS](#macos-install_macsh)
  below, which runs the services without Docker. On Windows the same is
  possible without Docker Desktop, see
  [Windows without Docker](#windows-without-docker-install_windowsps1).)
- An NVIDIA GPU with current drivers and the NVIDIA Container Toolkit. On
  Windows this comes with Docker Desktop and WSL2; on Linux install the toolkit
  from NVIDIA and restart Docker. Check that a container sees the GPU:

  ```bash
  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
  ```

- Roughly 80 GB of free disk space: the images (about 47 GB, most of it the
  PyTorch base they share), the model weights (MMAudio 6 GB, the search models
  8 GB, the vision-language model 6 GB) and the sound index (10 GB with the BBC
  archive). A downloaded BBC archive takes another 60 GB as mp3.
- On Windows, enough memory for the Docker VM: MMAudio needs about 10 GB while
  loading. Put `memory=24GB` under `[wsl2]` in `%UserProfile%\.wslconfig` and
  run `wsl --shutdown` once.

### Windows: install.ps1

On Windows one script does the first start and opens the setup page:

```powershell
git clone https://github.com/ai-sound-design/ai-services.git
cd ai-services
powershell -ExecutionPolicy Bypass -File install.ps1
```

It checks the requirements above (the NVIDIA driver, Docker Desktop with the WSL 2
engine, the GPU inside a container, the VM's memory in `.wslconfig`, free disk
space), creates `.env`, builds the images, starts the services and opens
`http://localhost:8020`. It installs neither Docker Desktop nor the driver: when
one is missing it stops with the link to its installer, and is simply run again
afterwards. Every step only does what is still missing, so it can be run any time.

On the first run it creates `.env` from `.env.example` with a random database
password and asks for the data folder (fetched audio and your libraries; default
`.\data`, worth a bigger drive for the 60 GB of the BBC archive). It also asks
for three optional folders: an existing copy of the BBC Sound Effects archive (a
folder of `<id>.wav` or `.mp3` files), which is then registered instead of
downloaded; a folder of your own sounds, which becomes a library of the search
(both only read); and a folder for the models, if they should be visible as
files rather than live in Docker volumes (see below). All of them can be given
as parameters (`-DataDir "D:\AI Sound Design" -BbcArchive "D:\BBC" -OwnSounds
"D:\Sounds" -ModelsDir "D:\Models"`, or `-NoPrompt` for the defaults) and end up
as the `.env` entries `AUDIO_HOST_DIR`, `LIBRARIES_HOST_DIR`,
`BBC_ARCHIVE_HOST_DIR`, `OWN_SOUNDS_HOST_DIR`, `OWN_SOUNDS_NAME`, `MODELS_STORE`
and `OLLAMA_MODELS_STORE`. The script never edits `docker-compose.yml`.

`-Ollama docker|host` says where the vision-language model for the spotting
runs: in the stack's own Ollama container (the default), or in an Ollama
installed on Windows (`host`: installed with winget when missing,
`OLLAMA_HOST=0.0.0.0` set for the account so the containers reach it as
`host.docker.internal`, the model pulled; `.env` then has `BUNDLED_OLLAMA=0`
and `VLM_URL=http://host.docker.internal:11434`). `-NoBrowser` leaves the setup
page closed at the end.

**Without a terminal: the installer app.** Download
`AI-Sound-Design-Installer-Windows.zip` from the newest release of
[installer](https://github.com/ai-sound-design/installer/releases/latest),
unpack it and double-click **AI Sound Design Installer.exe**. Its window offers
the backend services (in Docker Desktop, or on the PC directly as below), the
web UI, the Pro Tools plugin and the sound database, each with a checkbox, the
Ollama choice above, the folders to use, and then the progress. It fetches this
repository's `main` itself, runs `install.ps1` and remembers the commit it
installed (`.windows\installed-from`); opened again, it offers what has an
update. The app is not signed: the first time, Windows SmartScreen says it
protected the PC; *More info* > *Run anyway*.

### Windows without Docker: install_windows.ps1

For a Windows PC without Docker Desktop (or without the virtualization it
needs), the services run natively, as on a Mac: each in a Python virtualenv of
its own with the versions its Dockerfile installs, from the same source files,
on the same ports, configured by the same `.env`. PyTorch comes from its CUDA
12.4 wheels, so MMAudio, the search and the indexer run on the NVIDIA GPU; the
vision-language model runs in the Ollama for Windows; the sound index in
PostgreSQL 17 (EDB's binaries, the server parts only) with pgvector built for
it (pgvector publishes no Windows binaries; the installer's releases carry one
built by GitHub). The plugin and the setup page see no difference. The web UI
is set up as well, on http://localhost:8030.

The installer app above offers this as *On this PC directly*. In a terminal:

```powershell
git clone https://github.com/ai-sound-design/ai-services.git "$env:UserProfile\AI Sound Design\ai-services"
cd "$env:UserProfile\AI Sound Design\ai-services"
powershell -ExecutionPolicy Bypass -File install_windows.ps1
```

What is missing is installed with winget: Python 3.12 (for this account),
ffmpeg (Gyan's full build, with the drawtext filter the spotting burns frame
times with), Node.js (which only builds the web UI; its installer asks for
administrator rights) and Ollama (not when `.env` already points `VLM_URL` at
another server, such as LM Studio or an Ollama elsewhere with `VLM_API=openai`).
It creates `.env` as `install.ps1` does (the
same parameters, plus `-Sessions` for the web UI's Pro Tools sessions,
`-WebUI on|off` and `-Only webui`, which sets up only the web UI beside a Docker
stack), fetches PostgreSQL and pgvector into `.windows\pgsql`, builds the
virtualenvs and MMAudio's pinned source under `.windows\`, fetches the models
(MMAudio 10 GB, the vision-language model 6 GB), starts every service and
makes the shortcuts. Like `install.ps1`, it only does what is missing and can be
run again at any time.

The services run as background processes under a small supervisor that
restarts one when it exits, started by a task of the account at every logon
(*AI Sound Design backend* in the Task Scheduler). `windows\backend.py` manages
them, with the same commands as the Mac's `mac/backend.py`:

```powershell
python windows\backend.py status                 # which answer, on which port
python windows\backend.py logs mmaudio 100       # a service's log (.windows\logs\)
python windows\backend.py restart search         # after an edit or a change in .env
python windows\backend.py stop                   # all of them, until the next start or logon
python windows\backend.py start
python windows\backend.py uninstall              # stop everything, remove the logon task; data and .windows\ stay
python windows\backend.py database [--audio]     # fill the sound index (the setup page's "Set everything up"; --build-index builds it here)
```

Next to the installation and in the Start menu: **Web UI**, **Setup page**,
**Start backend**, **Stop backend** and **Backend status**. Where things are:
the database in `<data folder>\postgres`, the models in `<data folder>\models`
(or `MODELS_STORE`), the vision-language model in Ollama's
`%UserProfile%\.ollama\models`, everything installed in `.windows\`. Settings
that only matter here: `WINDOWS_BIND_HOST` (default `127.0.0.1`, this PC only;
`0.0.0.0` makes the services reachable from the network, as Docker does),
`WINDOWS_WEBUI=0` to leave the web UI out, `WEBUI_SESSIONS_DIRS`, `WEBUI_PORT`
and `WEBUI_REF` as on a Mac. The Docker stack and the native services cannot
run at the same time (the same ports): `docker compose stop` first.

### macOS: install_mac.sh

On an Apple Silicon Mac (M1 or later, macOS 14 or newer, 32 GB of memory
recommended) the services run natively instead of in Docker: Docker on macOS
cannot reach the Apple GPU, so the models would run on the CPU inside its VM.
Each service runs in a Python virtualenv of its own with the versions its
Dockerfile installs, from the same source files, on the same ports, configured
by the same `.env`; MMAudio, the search and the indexer run on the Apple GPU
(MPS), the vision-language model in a native Ollama, the sound index in a
Homebrew PostgreSQL 17 with pgvector. The plugin and the setup page see no
difference. The installer also sets up the
[web UI](https://github.com/ai-sound-design/ai-sound-webui), the plugin's four
modes in the browser without Pro Tools, on http://localhost:8030.

**Without Terminal: the installer app.** Download
`AI-Sound-Design-Installer-macOS.zip` from the newest release of
[installer](https://github.com/ai-sound-design/installer/releases/latest),
open it and double-click **AI Sound Design Installer**. Its window offers the
backend services, the web UI, the Pro Tools plugin (the newest release of
[ai-protools-aax](https://github.com/ai-sound-design/ai-protools-aax/releases)) and
the sound database (the BBC index, optionally with all of its audio), each with
a checkbox, the folders to use, and then the progress; macOS asks for the Mac's
password when Homebrew or the plugin is installed. The app is not signed by
Apple, so the first double-click ends in a message that macOS could not verify
it: click *Done*, open *System Settings > Privacy & Security*, click **Open
Anyway** and confirm with the Mac's password.

The installer fetches this repository's `main` itself and remembers the commit
it installed (`.mac/installed-from`); opened again, it compares what is installed
with GitHub (this repository, the web UI, the plugin's releases, its own releases)
and ticks what has an update. The sound database loads in the background: the
window can be closed, the loading carries on after a restart, and the installer
shows where it is.

Afterwards the app **AI Sound Design** (in Applications, so in Launchpad and
Spotlight; drag it into the Dock) opens the web UI, starting the services first
when they were stopped. `~/AI Sound Design` holds the installation, the
installer for later updates and the files **Web UI**, **Start backend**, **Stop
backend**, **Backend status** and **Setup page** to double-click.

**With Terminal:**

```bash
git clone https://github.com/ai-sound-design/ai-services.git "$HOME/AI Sound Design/ai-services"
cd "$HOME/AI Sound Design/ai-services"
bash install_mac.sh
```

The services must not run from Desktop, Documents, Downloads or iCloud Drive:
macOS keeps background services out of them, and a service there hangs at its
first file access. A copy in one of them (the downloaded ZIP) copies itself to
`~/AI Sound Design/ai-services` and installs from there; an installation already
there keeps its `.env`, its data and its environments. A checkout anywhere else
runs in place.

What is missing is installed: [Homebrew](https://brew.sh) with Apple's command
line tools (about 1 GB, no Xcode; this needs an administrator account and asks
for its password once), the Xcode licence accepted where a full Xcode was never
opened, then ffmpeg, PostgreSQL 17, pgvector, Python 3.12, Ollama and Node.js
(which only builds the web UI) with Homebrew. It creates `.env` as `install.ps1` does (a
random database password; data folder, BBC archive, own sounds and models folder
asked for or given as `--data-dir`, `--bbc-archive`, `--own-sounds`,
`--models-dir`, `--sessions` for the web UI's Pro Tools sessions, `--no-prompt`), builds the virtualenvs and MMAudio's pinned
source under `.mac/`, fetches the models (MMAudio 10 GB, the vision-language
model 6 GB), starts every service, makes the app **AI Sound Design** that opens
the web UI, and opens the web UI. Like `install.ps1`, it only does what is missing
and can be run again at any time.

The services run as launchd agents: they start at every login and are restarted
when one exits. `mac/backend.py` manages them:

```bash
python3 mac/backend.py status                 # which answer, on which port
python3 mac/backend.py logs mmaudio 100       # a service's log (.mac/logs/)
python3 mac/backend.py restart search         # after an edit or a change in .env
python3 mac/backend.py stop                   # all of them, until the next start or login
python3 mac/backend.py start
python3 mac/backend.py uninstall              # remove the agents and the app; data and .mac/ stay
python3 mac/backend.py database [--audio]     # fill the sound index (the setup page's "Set everything up"; --build-index builds it here)
```

Where things are: the database in `<data folder>/postgres`, the models in
`<data folder>/models` (or `MODELS_STORE`), the vision-language model in
Ollama's `~/.ollama/models`, everything installed in `.mac/`. Some settings only
matter on a Mac: `MAC_BIND_HOST` (default `127.0.0.1`, this Mac only; `0.0.0.0`
makes the services reachable from the network, as Docker does) and
`OLLAMA_PORT` (the Ollama this backend runs, default 11435, beside an Ollama of
your own on 11434). The web UI: `WEBUI_SESSIONS_DIRS` (the folders with Pro
Tools sessions it shows, several separated by `;`; not in Documents, Desktop or
Downloads, which macOS keeps background services out of), `WEBUI_PORT` (8030),
`MAC_WEBUI=0` to leave it out, `WEBUI_REF` for another branch or tag than
`main`. MMAudio runs in `float32` there (`FORCE_DTYPE`, which the
installer sets): the M1 and M2 have no native `bfloat16`, which made a sound
about a quarter slower. On an M1 Ultra an 8 s sound from a prompt takes about
9 s, a 20 s sound from a video about 80 s, spotting a 14 s clip about 100 s.

### The setup page

`http://localhost:8020` (service `setup`) shows the state of the stack: every
service and whether its model is loaded, the download of the MMAudio weights
and the vision-language model with progress, and the steps that fill the sound
index: the prebuilt BBC index from Hugging Face
([ai-sound-design/bbc-sound-index](https://huggingface.co/datasets/ai-sound-design/bbc-sound-index),
about 2.5 GB: the metadata, the text embeddings and the audio index of the whole
archive, no audio; a recording is fetched when it is used), or instead the BBC
metadata and the text embeddings made here, then the BBC audio files (registered
from an existing archive folder, or downloaded), the audio index over the files
on disk, and every folder under `data/libraries`. *Set everything up* runs the
steps in order; each can be paused and resumed and carries on where it stopped,
also after a restart of the machine, since the steps only ever process what is
still missing. The steps are the indexer's scripts below, so the command line
does the same. The index is free for personal, educational and research use
under the BBC's terms.

### Where things are stored

- **The data folder** (`.\data` unless `install.ps1` or `.env` say otherwise):
  fetched or downloaded audio (`AUDIO_HOST_DIR`) and your libraries
  (`LIBRARIES_HOST_DIR`). Plain folders, visible in Explorer, on whichever drive
  has the room.
- **The models** live in two named Docker volumes: `models` (MMAudio's weights
  and the search models, about 15 GB) and `ollama-models` (the vision-language
  model, about 6 GB). Docker Desktop lists them under *Volumes* with their
  sizes, and the setup page shows them too. A volume sits in Docker's own disk,
  where a 6 GB model loads in seconds; a Windows folder would be read through
  Docker Desktop's file sharing and take minutes per load. Anyone who wants the
  files visible anyway sets `MODELS_STORE` and `OLLAMA_MODELS_STORE` in `.env`
  to folders (or runs `install.ps1 -ModelsDir`), and the installer copies what
  is already there into the folders, or back into the volumes when the entries
  are removed again.
- **The database** (the sound index, 10 GB with the BBC archive), the setup
  page's logs and the hybrid service's output are named volumes as well
  (`postgres-data`, `setup-logs`, `hybrid-output`): PostgreSQL refuses a Windows
  folder as its data directory and would be far slower on one. On Windows all
  of Docker's data, volumes and images, lives in one virtual disk, by default
  under `%LocalAppData%\Docker\wsl`; to move it, use Docker Desktop >
  Settings > Resources > Advanced > Disk image location. `install.ps1` prints
  where it is and how much room that drive has.

### First start by hand

```bash
git clone https://github.com/ai-sound-design/ai-services.git
cd ai-services
docker compose up -d
docker compose logs -f
```

Every setting has a working default. To change one, copy `.env.example` to
`.env` and edit it; the file is not committed.

**The first start takes long.** It builds the images (which pulls the PyTorch
base image and installs the models' dependencies) and then downloads what the
models need: the MMAudio weights and encoders (roughly 10 GB) and the vision-language
model (about 6 GB) into the data folder.
With a fast connection that is half an hour; with a slow one, considerably
longer. Nothing is wrong while `docker compose ps` reports the services as
`health: starting` and `docker compose logs -f` shows downloads progressing.
Later starts take seconds, and the first request after a start takes a few
minutes while a model is loaded into GPU memory.

### Check

```bash
curl http://localhost:8000/health     # generation
curl http://localhost:8002/health     # search
curl http://localhost:8003/health     # spotting
curl http://localhost:8004/health     # hybrid
```

In the plugin, open *Settings*: the rows for MMAudio, sound search, spotting
and hybrid have a *Test* button each, which calls the same endpoints.

### Fill the sound index

The quick way is the ready-made index of the BBC archive on Hugging Face,
[ai-sound-design/bbc-sound-index](https://huggingface.co/datasets/ai-sound-design/bbc-sound-index)
(about 2.5 GB, no audio): the setup page's first step loads it, or
`docker compose run --rm indexer python load_index.py`. What follows builds the
index here instead, or adds your own libraries to it. An index loaded or built
before the int8 format of the 2 s windows holds float16 ones, which the search
still reads; `docker compose run --rm indexer python quantize_fine.py --vacuum`
converts them in place and halves the table. The dataset also holds the windows
as float16 (`BBC_INDEX_VARIANT=fp16` in `.env`, or `load_index.py --variant fp16`).

The search starts empty. Index a folder of your own sounds, the BBC Sound
Effects Archive, or both; details in [`search/README.md`](search/README.md).

```bash
# Your own sounds: ./data/libraries/<name>/**/*.wav (or aif, flac, mp3, ...)
docker compose run --rm indexer python index_folder.py <name>

# The BBC archive (free for research use under the BBC's terms; nothing of it
# is in this repository). Two commands, a few minutes, no bulk download: the
# search fetches a sound from the BBC the first time it is previewed or imported.
docker compose run --rm indexer python sources/bbc/harvest.py
docker compose run --rm indexer python embed.py --library bbc
```

For a machine without internet access, download the archive once
(`sources/bbc/download.py`) and set `AUDIO_FETCH=local_only` in `.env`. An
existing copy of the archive is registered instead, read-only:
`BBC_ARCHIVE_HOST_DIR` in `.env` names the folder and
`docker compose run --rm indexer python sources/bbc/register.py` points the
rows at its files.

The search **by sound**, which the plugin's *Use database sounds* relies on,
needs the audio index. For the BBC archive it comes ready-made: `load_index.py`
fetches it from Hugging Face and merges it in (`export_index.py` writes it from a
finished index). Built here instead, it needs the audio on disk: after
`download.py` (your own libraries are on disk anyway) run `embed_segments.py`.
This takes hours for the whole BBC archive (about four on a laptop GPU), can be
interrupted and resumed, and the partial index is usable right away;
`index_folder.py` does it for a folder library on its own.

```bash
docker compose run --rm indexer python embed_segments.py
```

### Stop, update, remove

```bash
docker compose down                     # stop; volumes and ./data stay
git pull && docker compose up -d --build   # update to a newer version
docker compose down -v                  # remove the containers and the volume of the database (the index)
```

`./data/` (fetched audio, your libraries, the models) is never touched by
Compose; delete it yourself if you want it gone. On Linux the files Docker
writes there belong to root, so that takes `sudo`.

## The vision-language model: bundled or your own

Spotting needs a vision-language model with a vision projector. There are two
ways to provide it:

1. **Bundled (default).** The `ollama` container is part of the stack and pulls
   `gemma4:e4b-it-qat` on first start. If Ollama is already installed on the
   host, point `OLLAMA_MODELS_STORE` at its store (`C:/Users/<you>/.ollama`,
   `~/.ollama`) and nothing is downloaded twice.
2. **Your own server.** Set `BUNDLED_OLLAMA=0` in `.env`; the container is then
   left out and `spotting-api` talks to `VLM_URL` instead:
   - an Ollama on the host: `VLM_URL=http://host.docker.internal:11434`, and start
     that Ollama with `OLLAMA_HOST=0.0.0.0` (by default it listens on localhost
     only, which a container cannot reach);
   - LM Studio, vLLM or any other OpenAI-compatible server: `VLM_API=openai` and
     for example `VLM_URL=http://host.docker.internal:1234`;
   - OpenAI itself, with no local model and no GPU for the spotting (the frames
     leave the machine): `VLM_API=openai`, `VLM_URL=https://api.openai.com`, a
     vision model in `VLM_MODEL` and the key in `VLM_API_KEY`.

`VLM_MODEL` selects the model. It must carry a vision projector: plain
`gemma4:e4b` silently ignores the images and answers from the text alone.

## Using another generation model

The plugin is not tied to MMAudio: a generation service only has to answer a
health check and accept a multipart request, and an adapter profile in the
plugin's adapters folder tells the plugin how. `generation/template/` is a
skeleton to fill in; [`generation/README.md`](generation/README.md) has the
contract and the profile format.

## Configuration

All settings are environment variables, listed with comments in
`.env.example`. The ones that matter most:

| Variable | Default | Meaning |
|----------|---------|---------|
| `MMAUDIO_PORT`, `GENERATION_PORT`, `SOUND_SEARCH_PORT`, `SPOTTING_PORT`, `HYBRID_PORT` | `8000`, `8010`, `8002`, `8003`, `8004` | Host ports; the plugin's default profiles expect the gateway (8010), search, spotting and hybrid |
| `BUNDLED_OLLAMA` | `1` | `0` leaves the Ollama container out |
| `VLM_URL`, `VLM_MODEL`, `VLM_API` | `http://ollama:11434`, `gemma4:e4b-it-qat`, `ollama` | Where and which the spotting model is |
| `AUDIO_FETCH` | `on_demand` | `local_only` restricts the search to sounds already on disk |
| `AUDIO_HOST_DIR` | `./data/audio` | Fetched and downloaded audio, one subfolder per library |
| `LIBRARIES_HOST_DIR` | `./data/libraries` | Your own sound libraries, one subfolder each |
| `BBC_ARCHIVE_HOST_DIR` | `./data/archive` | An existing copy of the BBC archive, mounted read-only; `register.py` or the setup page point the rows at its files |
| `OWN_SOUNDS_HOST_DIR`, `OWN_SOUNDS_NAME` | none, `own` | A folder of your own sounds, mounted read-only as the library `OWN_SOUNDS_NAME` |
| `MMAUDIO_MODEL` | `large_44k_v2` | Which MMAudio variant to serve |
| `CLAP_MODEL` | `laion/clap-htsat-fused` | Audio encoder for the search by sound; `embed_audio.py` and the API must agree |
| `GATEWAY_MAX_SECONDS`, `GATEWAY_OVERLAP_SECONDS` | `600`, `1.0` | The longest sound the generation gateway makes, and the overlap between its crossfaded windows |
| `FORCE_DEVICE` | `auto` | Leave at `auto`: CUDA with an NVIDIA GPU, the Apple GPU (`mps`) on a Mac; `cpu` exists in the code but is far too slow to be useful |

## Connecting the plugin

The plugin's default adapter profiles already point at the ports above on
`localhost`. To reach this backend from another machine, edit the address in
the plugin's Settings dialog (it is written into the profile file) or add a
profile of your own; the format is described in the plugin repository and in
the README of each service folder here. A search profile may name one of the
indexed libraries, which puts that library into the plugin's Backend list.

## Status

Research prototype, not a production service. The services were hosted on a
university server for the duration of the research project; this repository
is the portable, self-contained replacement.

## License

The code in this repository is licensed under the [MIT License](LICENSE).
Third-party components are **not** covered by it and keep their own terms:

| Component | Terms | Included here |
|-----------|-------|---------------|
| [MMAudio](https://github.com/hkchengrex/MMAudio) | MIT | No, fetched when the image is built |
| [X-CLIP](https://huggingface.co/microsoft/xclip-base-patch32) | MIT | No, downloaded at runtime |
| [CLAP](https://huggingface.co/laion/clap-htsat-fused) | Apache-2.0 | No, downloaded at runtime |
| Gemma 4 via [Ollama](https://ollama.com) | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) | No, pulled on first start |
| [pgvector](https://github.com/pgvector/pgvector) / PostgreSQL | PostgreSQL License | No, Docker image |
| BBC Sound Effects Archive | [BBC licensing](https://sound-effects.bbcrewind.co.uk/licensing), research use | No, indexed by you |

## Author

[Christoph J. Weber](https://github.com/chris-j-weber)
