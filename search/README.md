# Sound search

Video or prompt in, matching sounds from an indexed library out. `api/` is the
service the plugin talks to, `indexer/` fills the index, `sources/` holds one
folder per external archive (currently the BBC Sound Effects Archive).

## How it works

The index is one PostgreSQL table, `sounds` (`schema.sql`), with one row per
sound from any library: a description, a category, a duration, where the audio
is (on disk, at a media URL, or both) and the X-CLIP text embedding of the
description. A query, video frames or a prompt, is embedded with the same model
and compared by cosine similarity (pgvector). For that search the audio itself
is never analysed: X-CLIP is a video-text model, so what makes a library
searchable by video or prompt is the quality of its descriptions.

A second table, `sound_windows`, makes a library searchable **by sound**. Every
file on disk is cut into overlapping 10 s windows (5 s hop, at most 120 per
file) and each window gets a CLAP audio embedding; a query sound is embedded
the same way and the nearest windows come back with the file *and the second
it matches at*, which is what a two-second event inside a three-minute
recording needs. CLAP shares one space for audio and text, so the event's
description can be blended into the query (`text_weight`). The plugin's Hybrid
mode uses this to replace generated sounds with library recordings (*Use
database sounds*). The audio has to be read for this, so only files on disk
are indexed; the index is built with `embed_audio.py` and grows while it runs
(HNSW), so a partial index is usable at once.

## Libraries

Every row belongs to a library, named in the `library` column. Two kinds:

**A folder of your own sounds.** Put it under `LIBRARIES_HOST_DIR`
(default `./data/libraries`), one subfolder per library, and index it:

```bash
docker compose run --rm indexer python index_folder.py mine      # ./data/libraries/mine
```

The description of each file comes from, in order of preference, a sidecar
`index.csv` in the library root (columns `file`, `description`, optionally
`category`), the file's own title or comment tag, or the file name and its
folders split into words (`Doors/DOOR_Wooden_Creak_03.wav` becomes "doors door
wooden creak"). Re-running only touches what changed; `--prune` removes rows
whose file is gone. Embeddings are computed at the end of the run.

**The BBC Sound Effects Archive.** Free for personal, educational and research
use under the [BBC's terms](https://sound-effects.bbcrewind.co.uk/licensing).
No BBC audio or metadata are in this repository; build the index yourself:

```bash
docker compose run --rm indexer python sources/bbc/harvest.py     # metadata from the BBC API, minutes
docker compose run --rm indexer python embed.py --library bbc     # embeddings, minutes on a GPU
```

That is enough to search: with `AUDIO_FETCH=on_demand` (the default) the API
fetches a sound from the BBC the first time it is previewed or imported and
keeps it under `AUDIO_HOST_DIR/bbc/`. For a machine without internet access,
download the archive once (about 10 GB as mp3) and set `AUDIO_FETCH=local_only`:

```bash
docker compose run --rm indexer python sources/bbc/download.py
```

Add `--limit 2000` to harvest or download for a quick test with a subset.

**Search by sound** needs the audio, so after `download.py` (or for your own
libraries, which are on disk anyway) build the audio windows. For the whole
BBC archive (1,060 hours) this takes about five hours on a laptop GPU; decoding
the files is the larger part. It can be interrupted and resumed, and
`index_folder.py` runs it for a folder library on its own.

```bash
docker compose run --rm indexer python embed_audio.py                  # every local file
docker compose run --rm indexer python embed_audio.py --limit 500      # a first look
```

**Another archive.** Copy `sources/bbc/` as a pattern: a script that turns the
archive's records into rows with `library`, `external_id`, `description`,
`category`, `duration_seconds` and a `media_url` (or `file_path`), passed to
`common.db.upsert_sounds()`; then `embed.py --library <name>`.

## The contract the plugin uses

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | 2xx when ready; also lists the libraries with their sizes. |
| `GET /libraries` | The indexed libraries: sounds, embedded, on disk, hours. |
| `POST /search/sounds` | Multipart form: `video` (file, optional), `text` (optional), `limit`, `library` (optional: one name, or several separated by commas), `text_weight`, `num_frames`, `threshold`. Answers `{"results": [{"id", "library", "description", "category", "collection", "duration_seconds", "similarity"}, ...]}`. |
| `GET /sounds/{id}` | Metadata of one sound. |
| `GET /sounds/{id}/preview` | The audio, for listening before importing. |
| `GET /sounds/{id}/download` | The audio, for importing into the session. |
| `POST /search/by_audio` | Multipart form: `audio` (file), optional `text` and `text_weight` (share of the text in the query, default 0), `limit`, `library`, `category`, `refine` (default true: locate the best stretch of the query's length inside each matched window), `envelope_weight` (0..1, default 0.3: share of the loudness-envelope similarity in the refined score, so the match also moves like the query), `exclude` (sound ids). Answers `{"results": [{"id", "library", "description", "category", "duration_seconds", "offset_seconds", "length_seconds", "similarity"}, ...]}`; `offset_seconds` is where in the file the match lies. Needs the audio windows of `embed_audio.py`; `GET /health` reports `audio_search.available` and a `reason` when it is not. |
| `GET /sounds/{id}/snippet?start=&length=&channels=` | `length` seconds of the sound from `start`, as 48 kHz WAV with short fades: what a match points at, ready to place. `channels` 1 or 2 forces a channel count (pieces of different recordings on one track need the same). |

The plugin's search profile can name the library to search, so one profile
per library gives a switch in the plugin's Backend list:

```json
{
  "name": "My recordings",
  "kind": "search",
  "base_url": "http://localhost:8002",
  "health": "/health",
  "protocol": "ai-sound-design-search-v1",
  "request": { "fields": { "library": "mine" } }
}
```

Without a `library` field the search covers every library.

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `AUDIO_FETCH` | `on_demand` | `on_demand` fetches audio on first use; `local_only` restricts results to sounds on disk |
| `AUDIO_HOST_DIR` | `./data/audio` | Fetched and downloaded audio, one subfolder per library |
| `LIBRARIES_HOST_DIR` | `./data/libraries` | Your own libraries, one subfolder each, mounted read-only |
| `XCLIP_MODEL` | `microsoft/xclip-base-patch32-16-frames` | Query encoder; must match the embedding column (base 512, large 768) |
| `CLAP_MODEL` | `laion/clap-htsat-fused` | Audio encoder for the search by sound; the API and `embed_audio.py` must use the same |
| `BBC_AUDIO_FORMAT` | `mp3` | Format `sources/bbc/download.py` fetches |
