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
| `POST /spot` | Multipart form: `video` (file), `start_timecode` (`HH:MM:SS:FF`, where the sent video starts on the timeline), `fps` (timecode rate), optional `hints` (free text about the scene), optional `sample_fps`, optional `job_id` (any unique string; enables the progress query). |
| `GET /spot/progress/{job_id}` | While a `/spot` request with that `job_id` runs: `stage`, `fraction` (0..1, model calls done) and `detail` ("frames 16 of 40"). The plugin polls this every two seconds and shows the percentage for the clip. Optional: a backend without it just shows a busy bar. |

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
  "video_duration_seconds": 8.6,
  "frames_analysed": 17,
  "model": "gemma4:e4b-it-qat",
  "seconds_taken": 31.2
}
```

`category` is one of `dialogue`, `foley`, `sfx`, `ambience`, `music`, and
dialogue means human speech only. Overlapping chunks of frames are analysed
separately and their near-duplicate events are folded.

## Choosing the model

| Variable | Default | Meaning |
|----------|---------|---------|
| `VLM_URL` | `http://ollama:11434` | Where the model runs |
| `VLM_API` | `ollama` | `ollama` (`/api/chat`) or `openai` (`/v1/chat/completions`, e.g. LM Studio, vLLM) |
| `VLM_MODEL` | `gemma4:e4b-it-qat` | Must carry a vision projector; a text-only variant silently ignores the frames |
| `SPOTTING_SAMPLE_FPS` | `2` | Frames sampled per second of video |
| `SPOTTING_FRAMES_PER_CALL` | `8` | Frames shown to the model per call |
| `VLM_NUM_CTX` | `32768` | Context size; eight frames need far more than Ollama's default 8192 |
| `FRAME_STAMP` | `1` | Burn the frame time into the picture so the model can place events |
| `VLM_TEMPERATURE` | `0` | Sampling temperature |
