# ai-services

An example backend for the [AI Sound Design Pro Tools plugin](https://github.com/ai-sound-design/ai-protools-aax).

The plugin adds four AI workflows to Pro Tools: generating sound for a video
clip, recommending sounds from a searchable library, spotting sound events as
memory locations, and a hybrid of the two that returns one generated sound per
sound event. It does none of that itself; it sends the work to HTTP services. This repository is one complete set of such services, packaged
so that a single `docker compose up` brings it up on a machine with an NVIDIA
GPU. It is also the reference for writing your own: every service folder
documents the contract the plugin relies on, and a service that fulfils it can
replace the one here without a change to the plugin.

All workflows are fully functional here. This differs from the prototype
used in the user study, where the spotting was operated Wizard-of-Oz style by
the experimenters; in this backend the spotting service runs a vision-language
model.

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
   library recordings that sound like it on a track underneath when asked.

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
| [`search/`](search/README.md) | `sound-search-api` | 8002 | Video or prompt in, matching sounds out. X-CLIP embeddings over any indexed library: your own folders of sounds, the BBC Sound Effects Archive, or both. Also sound in, matching recordings out, located to the second: CLAP embeddings of 10 s windows of the files on disk. |
| [`spotting/`](spotting/README.md) | `spotting-api` | 8003 | Video range in, sound events with timecodes out. A vision-language model describes the sounds a silent picture implies. |
| [`hybrid/`](hybrid/README.md) | `hybrid-api` | 8004 | Video range in, one generated sound per sound event out, each with its own start and end; on request also library recordings that sound like each one, stitched from pieces. The reference implementation composes the services above. |

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
  (macOS has no NVIDIA GPU passthrough, so the stack cannot run there.)
- An NVIDIA GPU with current drivers and the NVIDIA Container Toolkit. On
  Windows this comes with Docker Desktop and WSL2; on Linux install the toolkit
  from NVIDIA and restart Docker. Check that a container sees the GPU:

  ```bash
  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
  ```

- Roughly 30 GB of free disk space for images and model weights, more if you
  download the BBC archive (about 10 GB as mp3).
- On Windows, enough memory for the Docker VM: MMAudio needs about 10 GB while
  loading. Put `memory=24GB` under `[wsl2]` in `%UserProfile%\.wslconfig` and
  run `wsl --shutdown` once.

### First start

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
models need: the MMAudio weights and encoders (roughly 10 GB) into named
volumes, the vision-language model (about 6 GB) into the Ollama container.
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
(`sources/bbc/download.py`) and set `AUDIO_FETCH=local_only` in `.env`.

The search **by sound**, which the plugin's *Use database sounds* relies on,
needs the audio itself: after `download.py` (your own libraries are on disk
anyway) build the audio windows. This takes hours for the whole BBC archive,
can be interrupted and resumed, and the partial index is usable right away;
`index_folder.py` does it for a folder library on its own.

```bash
docker compose run --rm indexer python embed_audio.py
```

### Stop, update, remove

```bash
docker compose down                     # stop; volumes and ./data stay
git pull && docker compose up -d --build   # update to a newer version
docker compose down -v                  # remove the containers and the volumes (weights, index)
```

`./data/` (fetched audio, your libraries, the Ollama store) is never touched
by Compose; delete it yourself if you want it gone. On Linux the files Docker
writes there belong to root, so that takes `sudo`.

## The vision-language model: bundled or your own

Spotting needs a vision-language model with a vision projector. There are two
ways to provide it:

1. **Bundled (default).** The `ollama` container is part of the stack and pulls
   `gemma4:e4b-it-qat` on first start. If Ollama is already installed on the
   host, point `OLLAMA_MODELS_HOST_DIR` at its store (`C:/Users/<you>/.ollama`,
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
| `MMAUDIO_MODEL` | `large_44k_v2` | Which MMAudio variant to serve |
| `CLAP_MODEL` | `laion/clap-htsat-fused` | Audio encoder for the search by sound; `embed_audio.py` and the API must agree |
| `GATEWAY_MAX_SECONDS`, `GATEWAY_OVERLAP_SECONDS` | `600`, `1.0` | The longest sound the generation gateway makes, and the overlap between its crossfaded windows |
| `FORCE_DEVICE` | `auto` | Leave at `auto`. The stack needs an NVIDIA GPU; `cpu` exists in the code but is far too slow to be useful |

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

Anonymized for peer review.
