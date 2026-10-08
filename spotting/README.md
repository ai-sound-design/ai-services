# Spotting

A video range in, the sound events a sound editor would need to cover out, each
with a start and end timecode. The plugin turns them into Pro Tools memory
locations.

`api/` samples frames from the video, burns each frame's time into its corner
and shows them, a few at a time, to a vision-language model with a prompt that
asks for the sounds a silent picture implies. The model is not part of the
image: it is reached over HTTP and chosen by environment variables, so the
same service works with the bundled Ollama container, an Ollama or LM Studio
on the host, or a remote endpoint (see the top-level README).

## The contract the plugin uses

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | 2xx when the service and its model are reachable. |
| `GET /capabilities` | Model name, API dialect, sampling settings. |
| `POST /spot` | Multipart form: `video` (file), `start_timecode` (`HH:MM:SS:FF`, where the sent video starts on the timeline), `fps` (timecode rate), optional `hints` (free text about the scene), optional `categories` (comma-separated kinds to keep: dialogue, foley, sfx, ambience, music; the model is told which are wanted, other events are dropped), optional `sample_fps`, optional `job_id` (any unique string; enables the progress query). |
| `GET /spot/progress/{job_id}` | While a `/spot` request with that `job_id` runs: `stage`, `fraction` (0..1, model calls done) and `detail` ("frames 16 of 40"). The plugin polls this every two seconds and shows the percentage for the clip. Optional: a backend without it just shows a busy bar. |
| `POST /cuts` | Multipart form: `video` (file). The picture cuts of the clip in seconds from its start, for a caller that brings its own events (the plugin's markers) and wants them split per shot like `/spot` does. Optional. |
| `POST /scenes` | Multipart form: `videos` (the clips of a range, one file each, in timeline order), optional `names` (JSON list, same order), optional `job_id` (progress under `/spot/progress/{job_id}`). Groups consecutive clips into scenes and names them, see below. Optional: the plugin's *Detect scenes* switch needs it. |

`/spot` answers:

```json
{
  "events": [
    {
      "label": "car passes",
      "category": "sfx",
      "description": "A car drives past from left to right on a wet street",
      "start_seconds": 1.5, "end_seconds": 4.0,
      "start_timecode": "00:00:18:15", "end_timecode": "00:00:21:00",
      "confidence": 0.8
    }
  ],
  "cuts": [2.67, 4.43, 6.63],
  "scenes": [
    {"start_seconds": 0.0, "end_seconds": 8.6, "place": "wet city street at night",
     "what_happens": "A woman in a raincoat waits at the kerb while cars pass.",
     "expected_sounds": ["car pass-bys on wet asphalt", "footsteps on wet pavement", "raincoat rustle"]}
  ],
  "video_duration_seconds": 8.6,
  "frames_analysed": 17,
  "model": "gemma4:e4b-it-qat",
  "seconds_taken": 31.2
}
```

`category` is one of `dialogue`, `foley`, `sfx`, `ambience`, `music`, and
dialogue means human speech only. Overlapping chunks of frames are analysed
separately and their near-duplicate events are folded.

`cuts` are the picture cuts inside the sent clip (seconds from its start). A foley
or sound-effect event that runs across a cut is split there, one event per shot,
because the sound made for it has to change with the picture; ambience, music and
dialogue carry on across the cuts of a scene. Pieces shorter than
`SPOTTING_CUT_MIN_PIECE_S` are dropped, so an event that starts a few frames before
a cut starts at the cut. The detector compares perceptual hashes of consecutive
frames (PySceneDetect's `HashDetector`): a cut changes the whole image, a fast
camera move or the stutter of a frame-rate conversion changes far less of the
hash. At the default threshold 0.3 it found every cut of the test clips and none
of their camera moves (those stay below 0.25); `SPOTTING_CUT_DETECTION=0` turns it
off. Scenes (`/scenes`) stay per clip: a cut within a clip is a new shot, not a new
place.

`scenes` come from a first, sparse pass over the whole clip (one frame every
`1 / SPOTTING_SCENE_FPS` seconds, eight per call): the model divides the clip into
scenes and describes each one, the place, what the people do with their hands,
bodies and feet, what moves fast, which sounds a sound editor would expect, and
the ambience of the place. The dense pass then works scene by scene, and every
call is given its scene's description and expected sounds as context, so a call
that sees four seconds of hands over a keyboard knows it is looking at someone
typing in an office. The ambience events come from the scenes, one bed per place,
instead of from four-second chunks whose wordings had to be folded afterwards.
Title cards, credits and black are not a place: they get no calls of their own,
and the bed of the scene next to them runs under them, as a sound designer lays
the first scene's room tone under the opening titles.
`SPOTTING_SCENE_CONTEXT=0`, or `scene_context=0` in the request, skips the pass.

`/scenes` answers:

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

A scene is one place and one continuous stretch of story time: a new angle of
the same place at the same moment stays in the scene, a new place or a jump in
time starts the next. The reference implementation shows the model the first
and last frame of two consecutive clips and asks whether the second continues
the scene (one call per cut), then names every scene from up to eight of its
frames. The plugin writes one memory location per scene, spanning its clips.

## Choosing the model

| Variable | Default | Meaning |
|----------|---------|---------|
| `VLM_URL` | `http://ollama:11434` | Where the model runs |
| `VLM_API` | `ollama` | `ollama` (`/api/chat`) or `openai` (`/v1/chat/completions`, e.g. LM Studio, vLLM, or OpenAI itself) |
| `VLM_API_KEY` | empty | Bearer token for a hosted OpenAI-compatible service (OpenAI: `VLM_URL=https://api.openai.com`); a local server needs none |
| `VLM_MODEL` | `gemma4:e4b-it-qat` | Must carry a vision projector; a text-only variant silently ignores the frames |
| `SPOTTING_SAMPLE_FPS` | `2` | Frames sampled per second of video |
| `SPOTTING_FRAMES_PER_CALL` | `8` | Frames shown to the model per call |
| `VLM_NUM_CTX` | `32768` | Context size; eight frames need far more than Ollama's default 8192 |
| `FRAME_STAMP` | `1` | Burn the frame time into the picture so the model can place events |
| `VLM_TEMPERATURE` | `0` | Sampling temperature |
| `SPOTTING_VLM_NUM_PREDICT` | `2048` | Longest answer of one call in tokens. A grammar-constrained generation can run away; the cap ends it, the call is retried once and then skipped, and the response lists skipped calls under `warnings` |
| `SPOTTING_CUT_DETECTION` | `1` | Split foley and sound-effect events at the picture cuts inside a clip (`0`: off) |
| `SPOTTING_CUT_THRESHOLD` | `0.3` | Hash distance between consecutive frames that counts as a cut (lower finds softer cuts, and camera moves from about 0.25 down) |
| `SPOTTING_CUT_MIN_PIECE_S` | `0.3` | Shortest piece an event is split into; shorter ones are dropped |
| `SPOTTING_SCENE_CONTEXT` | `1` | Describe the scenes first and spot each chunk with its scene's context (`0`: chunks on their own, ambience per chunk) |
| `SPOTTING_SCENE_FPS` | `0.25` | Frames sampled per second of video for the scene pass |
| `SPOTTING_SCENE_CHECKLIST` | `0` | `1`: tell each chunk call which sounds the scene pass expects in its scene. Finds more (keyboard, whooshes), but a small model then invents some of them in every chunk; off, it is told only the place and what happens |
| `SPOTTING_EVENT_EVIDENCE` | `0` | Make the model cite, for every event, the frames that show its source acting (`1`); an event it cannot ground is not listed |
| `SPOTTING_EVENT_OBSERVE` | `0` | `1`: the model writes, per frame, what is visibly happening before it lists the events |
| `SPOTTING_SCENE_HINT` | `place` | What a chunk call is told about its scene: `place` (the place only), `full` (also what happens there; a small model then lists those actions in every chunk whether the frames show them or not), `none` |
| `SPOTTING_MIN_SCENE_S` | `10` | A scene shorter than this is a shot the scene pass singled out; it joins its neighbour |
| `SPOTTING_MERGE_GAP_S` | `2.0` | Same-label events closer than this are one event: one run of steps or of typing is one marker, as an editor lays one clip (cuts still split it) |
