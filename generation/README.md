# Generation services

Video (or a prompt) in, audio out. `mmaudio/` is the reference implementation
the stack starts; `template/` is a skeleton for wrapping a model of your own.

## The contract

The plugin does not know any model. What it sends and expects is described by
an **adapter profile**, a JSON file in the plugin's adapters folder
(`%APPDATA%\AI Sound Design\adapters\` on Windows,
`~/Library/AI Sound Design/adapters/` on macOS). So a generation service has to
offer only two things:

| Endpoint | Purpose |
|----------|---------|
| `GET <health>` (default `/health`) | Answers 2xx when the service is ready. The plugin's Test button and its pre-flight check call it. |
| `POST <endpoint>` (for example `/generate`) | Multipart form: the video file (absent in text-only mode) plus whatever fields the profile maps. Answers with the audio itself (WAV or FLAC, `Content-Disposition` optional) or with JSON that names where to fetch it. |

Everything else, field names, extra parameters, the response shape, is
declared in the profile:

```json
{
  "name": "My model (local)",
  "kind": "generation",
  "base_url": "http://localhost:8010",
  "base_url_tunnel": "",
  "health": "/health",
  "request": {
    "endpoint": "/generate",
    "video_field": "video",
    "fields": {
      "prompt": "{prompt}",
      "negative_prompt": "{negative_prompt}",
      "seed": "{seed}",
      "duration": "{duration?}"
    }
  },
  "response": { "kind": "audio_file" },
  "supports": ["negative_prompt", "seed", "duration", "text_only"],
  "duration": { "min": 4, "max": 12, "default": 8 },
  "timeout_seconds": 600
}
```

- `fields` values are templates. `{prompt}`, `{negative_prompt}`, `{seed}` and
  `{duration}` are filled from the plugin; a trailing `?` makes the field
  optional (left out when empty). The seed is always a concrete non-negative
  integer: the plugin's "-1 = random" is resolved before sending, so the seed
  that was used is logged and part of the file name. A backend should still
  treat a negative seed as "pick one", for other callers.A value without braces is sent verbatim, so
  model-specific constants such as `"model_name": "large_44k_v2"` go here too.
- `response.kind` is `audio_file` (the body is the audio) or `json`; with
  `json`, `audio_url_field` or `audio_path_field` names the JSON key that holds
  the file's URL or server path.
- `supports` decides which controls the plugin enables: `negative_prompt`,
  `seed`, `duration`, and `text_only` for a model that can generate without a
  video (T2A mode).
- `duration` states the lengths in seconds the model accepts. The plugin offers
  whole seconds between `min` and `max` in its text-to-audio list, preselects
  `default`, and skips video clips outside the range. Without the block the
  plugin assumes 4 to 12 seconds.

The full format is documented in the plugin repository,
`companion/api/adapters.py`.

## Wrapping your own model

1. Copy `template/` to `generation/<name>/`.
2. Fill in `load_model()` and `generate()` in `main.py`; the endpoints are
   already there. Add the model's dependencies to `requirements.txt` and a
   CUDA base image to the `Dockerfile` if it needs a GPU.
3. Add a service to `docker-compose.yml` (the `mmaudio-api` block is a
   pattern; give it a port of its own).
4. Write a profile like the one above into the plugin's adapters folder, or
   press *Open Adapter Folder* in the plugin's Settings. The new backend then
   appears in the plugin's Backend list of the Audio Generation mode.

A service does not have to live in this stack at all: any HTTP server that
fulfils the contract, on any machine, works with a profile that names its
address.

## MMAudio (reference)

`mmaudio/` serves [MMAudio](https://github.com/hkchengrex/MMAudio) on port
8000. Its `/generate` takes `video`, `prompt`, `negative_prompt`, `seed`,
`duration`, `model_name`, `num_steps`, `cfg_strength`, `output_format`,
`full_precision`, and answers with the audio file. The weights (about 10 GB)
are fetched on the first start into named volumes; see the top-level README.
