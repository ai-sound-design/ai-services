# Hybrid

A video range in, one generated sound per sound event out, each with its own
start and end inside the range. The plugin's Hybrid mode places every sound on
a track of its own, named after the event, at the event's position.

`api/` is the reference implementation. It has no model of its own: the
events come with the request (the memory locations inside the range, sent by
the plugin when *Use existing memory locations* is ticked) or, when none are
sent, from the spotting service;for every event the matching part of the video goes to the
generation service with the event's description as the prompt, and the answer
is trimmed to the event's length. A model that does all of this in one pass
replaces this service without a change to the plugin: it only has to fulfil
the contract below.

## The contract the plugin uses

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | 2xx when ready. |
| `POST /hybrid` | Multipart form: `video` (the cut range), `start_timecode` (`HH:MM:SS:FF`, where the range starts on the timeline), `fps`, `prompt` (optional, applies to every event), `negative_prompt`, `seed`, `events` (optional JSON, see below), and for library matches `match` (true/false) with `pieces_per_10s`, `min_piece_seconds`, `layers`, `text_weight`, `library`, `category_filter`, `min_similarity`, `ambience_handle_seconds`. |
| `POST /hybrid/scenes` | Multipart form: `videos` (the clips of the range, one file each, in timeline order), optional `names` (JSON list of clip names), optional `job_id`. Answers which consecutive clips form one scene, see below. Optional: the plugin's *Detect scenes* switch needs it. |
| `GET /hybrid/files/{name}` | One generated sound, as WAV. |
| `GET /hybrid/progress/{job_id}` | While a `/hybrid` request sent with an optional `job_id` form field runs: `stage` (spotting, generating, done), `fraction` (0..1 of the request) and `detail` ("sound 3 of 7: cat footsteps"). Optional; the plugin shows a busy bar without it. |

`events`, when sent, is a JSON list of the sound events the backend should
produce, relative to the start of the sent video:

```json
[{"label": "cat meowing", "description": "sfx: a cat meows twice", "category": "sfx",
  "start_seconds": 1.2, "end_seconds": 3.0}]
```

Without it the backend decides what the events are.

`/hybrid` answers:

```json
{
  "sounds": [
    {"id": "…", "label": "cat meowing", "category": "sfx", "description": "…",
     "start_seconds": 1.2, "end_seconds": 3.0, "audio_url": "/hybrid/files/….wav"}
  ],
  "events_source": "memory_locations",
  "video_duration_seconds": 8.6,
  "model": "hybrid: spotting + generation",
  "seconds_taken": 41.0
}
```

`audio_url` may be relative to the service's base address or absolute. Sounds
may overlap and may be shorter than the range; the plugin places each at
`start_timecode + start_seconds`.

With `match=true` every sound also carries the library recordings that sound
like it, as pieces in time and in `layers` (further, different recordings to
stack underneath). The generated sound is searched whole first and only cut
where its spectrum changes and the cut improves the match, up to
`pieces_per_10s` pieces per ten seconds, none shorter than
`min_piece_seconds`; a recording that fits the whole sound stays whole. Pieces
under `min_similarity` are left out, so a sound without a convincing match
keeps only its generated version. `start_seconds` of a piece is relative to
the generated sound:

```json
"match": {
  "pieces": [
    {"layer": 1, "start_seconds": 0.0, "length_seconds": 2.4, "sound_id": 8123,
     "library": "bbc", "description": "Wave breaking on shingle", "offset_seconds": 141.5,
     "handle_before_seconds": 0.0, "handle_after_seconds": 0.0,
     "similarity": 0.71, "audio_url": "/hybrid/files/….wav"},
    {"layer": 1, "start_seconds": 2.4, "length_seconds": 3.1, "sound_id": 2210, "…": "…"}
  ],
  "pieces_per_10s": 3, "layers": 1, "handle_seconds": 0.0
}
```

An event of category `ambience` gets `ambience_handle_seconds` of its recording
before and after every piece, as far as the recording reaches, so the plugin
can fade it in and out: the piece's file then starts `handle_before_seconds`
before `start_seconds` and the plugin places it that much earlier. An ambience
event longer than the generated sound (the generation model has a maximum
length) whose recording matched the sound as a whole is cut in the length of
the event, as far as the recording reaches, so a two-minute room tone gets two
minutes of it; `length_seconds` says how much.

The plugin's *Use database sounds* switch places the pieces instead of the
generated sound (which stays only where nothing matched); *Keep generated
sounds* puts it on a track above them. The reference
implementation asks the search service's `/search/by_audio` (CLAP over 10 s
windows of the files on disk, see search/README.md) per piece and keeps the
sequence in one recording where the scores allow, so a wave does not become a
patchwork of ten waves. `GET /health` reports `database_match.available` and a
`reason` when it is not; the plugin greys the switch out accordingly.

The plugin's profile for this kind:

```json
{
  "name": "Hybrid (local)",
  "kind": "hybrid",
  "base_url": "http://localhost:8004",
  "health": "/health",
  "protocol": "ai-sound-design-hybrid-v1",
  "request": { "endpoint": "/hybrid", "video_field": "video",
               "fields": { "prompt": "{prompt?}", "negative_prompt": "{negative_prompt?}", "seed": "{seed}" } },
  "response": { "kind": "json", "sounds_field": "sounds", "audio_url_field": "audio_url" },
  "supports": ["negative_prompt", "seed", "memory_locations"],
  "timeout_seconds": 1800
}
```

`supports` lists `memory_locations` when the backend accepts the `events`
field; the plugin then enables the *Use existing memory locations* switch.

### Scenes

`/hybrid/scenes` groups the clips of a range into scenes (one place, one
continuous stretch of time), so that sounds can be organised per scene rather
than per clip. The reference implementation forwards to the spotting service's
`/scenes` (see spotting/README.md). The answer:

```json
{
  "scenes": [
    {"index": 1, "name": "Urban street, night", "description": "A taxi crosses a busy junction.",
     "first_clip": 0, "last_clip": 1},
    {"index": 2, "name": "Living room, day", "description": "…", "first_clip": 2, "last_clip": 3}
  ],
  "clip_scenes": [1, 1, 2, 2],
  "model": "gemma4:e4b-it-qat",
  "seconds_taken": 61.0
}
```

The plugin writes one memory location per scene, spanning its clips, and
remembers the scene of every sound. `GET /health` reports `scenes_available`.
Progress for a `job_id` sent along is answered by `/hybrid/progress/{job_id}`.

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `GENERATION_URL` | `http://mmaudio-api:8000` | Where the generation service runs |
| `SPOTTING_URL` | `http://spotting-api:8003` | Where the spotting service runs, used when no events are sent |
| `SEARCH_URL` | `http://sound-search-api:8002` | Where the search service runs, used for library matches |
| `GEN_MIN_SECONDS`, `GEN_MAX_SECONDS` | `4`, `12` | The generation model's length window: shorter events are padded and trimmed afterwards, longer ones get a sound of the maximum length |
| `OUTPUT_DIR`, `KEEP_HOURS` | `/data/hybrid`, `24` | Where the sounds wait for download, and for how long |

Events are generated one after another, so a range with many events takes
about the generation time per event times their number.
