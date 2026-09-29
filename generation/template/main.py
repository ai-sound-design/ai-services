"""
Template for a generation service of your own.

Fill in `load_model()` and `generate()`, build the image, add a profile in the
plugin's adapters folder that names this service, and the plugin can use it
without any change to the plugin itself. See generation/README.md for the
contract this file fulfils and for the profile format.

Run locally without Docker:
    pip install -r requirements.txt
    python main.py
"""
from __future__ import annotations

import logging
import os
import random
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "info").upper(),
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("generation")

SERVICE_NAME = os.environ.get("SERVICE_NAME", "My generation model")
API_PORT = int(os.environ.get("API_PORT", "8010"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", tempfile.gettempdir())) / "generation-output"

app = FastAPI(title=SERVICE_NAME, version="0.1.0")
model = None


# ── The two functions to fill in ─────────────────────────────────────────────

def load_model():
    """Load your model once at start-up and return whatever generate() needs."""
    raise NotImplementedError("load your model here")


def generate(video_path: Optional[Path], prompt: str, negative_prompt: str,
             seed: int, duration: Optional[float], output_path: Path) -> Path:
    """
    Produce audio for `video_path` (None in text-only mode) and write it to
    `output_path` as WAV or FLAC. Return the path written.

    duration is None when the plugin did not ask for a specific length; use the
    video's length then, or your model's default for text-only generation.
    """
    raise NotImplementedError("run your model here")


# ── The contract; nothing below needs changing ───────────────────────────────

@app.on_event("startup")
def _startup():
    global model
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        model = load_model()
        log.info("%s ready on port %d", SERVICE_NAME, API_PORT)
    except NotImplementedError as exc:
        log.warning("%s: %s (the service answers /health but /generate will fail)", SERVICE_NAME, exc)


@app.get("/health")
def health():
    """200 with a small JSON body means: reachable. The plugin's Test button calls this."""
    return {"status": "ok", "service": SERVICE_NAME, "model_loaded": model is not None}


@app.post("/generate")
async def generate_endpoint(
    video: Optional[UploadFile] = File(None),      # absent in text-only mode
    prompt: str = Form(""),
    negative_prompt: str = Form(""),
    seed: int = Form(42),
    duration: Optional[float] = Form(None),
):
    """Multipart in, audio file out. Field names are whatever your profile maps to them."""
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    video_path: Optional[Path] = None
    if video is not None:
        suffix = Path(video.filename or "input.mp4").suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=OUTPUT_DIR) as handle:
            handle.write(await video.read())
            video_path = Path(handle.name)

    if seed < 0:  # the plugin sends a concrete seed; -1 from other callers means "pick one"
        seed = random.randint(0, 2**31 - 1)
    output_path = OUTPUT_DIR / f"generated_{seed}_{os.getpid()}_{id(prompt)}.wav"
    try:
        written = generate(video_path, prompt, negative_prompt, seed, duration, output_path)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    except Exception as exc:
        log.exception("generation failed")
        raise HTTPException(status_code=500, detail=f"generation failed: {exc}")
    finally:
        if video_path is not None:
            video_path.unlink(missing_ok=True)

    media_type = "audio/flac" if written.suffix.lower() == ".flac" else "audio/wav"
    return FileResponse(written, media_type=media_type, filename=written.name)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
