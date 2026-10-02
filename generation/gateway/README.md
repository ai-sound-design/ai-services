# Generation gateway

The `/generate` contract without a length limit, in front of a thin model service.

A model service (MMAudio today, whatever comes next tomorrow) answers one call with
one window of audio and says in its `/health` what it can do. The gateway sits in
front of it and does everything that does not depend on the model, so that a new
model is a new thin container and nothing else changes:

- **Long requests** are generated in overlapping windows of the model's maximum
  length, each from its own stretch of the video, with the same prompt and seed,
  and joined with an equal-power crossfade over the overlap (default 1 s). A 40 s
  request with a 12 s model becomes 0-12, 11-23, 22-34, 28-40.
- **Continuing models**: a model whose health says `"continuation": true` gets the
  last seconds of the previous window as audio context (field `context_audio`, or
  what its health names) and the windows are butt-joined instead of crossfaded.
- **Short requests** below the model's minimum get the video held on its last frame
  (or, text-only, the minimum length) and the audio trimmed back afterwards.
- **Progress** per window under `/generate/progress/{job_id}` when the request
  carries a `job_id`; a caller that disconnects stops the run between windows.

The plugin's generation profile and the hybrid service point at the gateway, not at
the model. The window size is the model's business; the gateway's own limit
(`GATEWAY_MAX_SECONDS`, default 600 s) is what the plugin's profile states as
`duration.max`.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | `min_seconds`, `max_seconds` (the gateway's limit), `window_seconds` (the model's), `overlap_seconds`, `continuation`, `modes`; 503 while the model service is unreachable |
| `POST /generate` | multipart `video` (optional), `prompt`, `negative_prompt`, `seed`, `duration` (seconds; required without a video, otherwise optional and at most the video's length), `output_format` (`wav`, `flac`), `job_id` (optional); `model_name`, `num_steps`, `cfg_strength`, `full_precision` are passed through. Answers the audio file with `X-Generation-Parts`, `X-Duration`, `X-Seed`, `X-Window-Seconds`, `X-Generation-Time` |
| `GET /generate/progress/{job_id}` | `stage`, `part`, `parts`, `fraction`, `detail` |

## What a model service declares

```json
"capabilities": {"min_seconds": 4, "max_seconds": 12, "modes": ["v2a", "t2a"],
                 "continuation": false, "context_field": "context_audio", "context_seconds": 2}
```

in its `/health`. Without the block the gateway assumes 4-12 s, both modes and no
continuation.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `GENERATION_MODEL_URL` | `http://mmaudio-api:8000` | the model service |
| `GATEWAY_MAX_SECONDS` | `600` | the longest sound the gateway makes |
| `GATEWAY_OVERLAP_SECONDS` | `1.0` | overlap between crossfaded windows |
| `GATEWAY_CONTEXT_SECONDS` | `2.0` | audio handed to a continuing model |
| `MODEL_TIMEOUT_S` | `1800` | one window's generation at most |
| `API_PORT` | `8010` | |

## Limits

MMAudio gets no audio from the previous window, so at a crossfade the character of a
tonal or rhythmic sound can change; an ambience hardly shows it. A longer overlap
softens the step, a continuing model removes it.
