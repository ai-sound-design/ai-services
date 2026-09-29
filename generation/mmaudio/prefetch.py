"""Fetch every model file the API needs, before the server accepts requests.

MMAudio downloads its checkpoints lazily, on the first generation request. Two
things follow from that, and both are bad for a tool someone works with: the
first request of a fresh container takes a quarter of an hour, and a network
failure surfaces as a failed job in the middle of the user's session rather
than as a service that never came up.

This script moves the download to container start. The entrypoint runs it before
the server, so the container is either ready or it is not running.

It is idempotent and cheap to repeat. A sentinel beside the checkpoints records
which model was verified and how large its files were, so later starts cost a
stat() per file instead of an md5 over six gigabytes of weights.

Environment:
  MMAUDIO_MODEL      model variant to prepare (default: large_44k_v2)
  MMAUDIO_SENTINEL   path of the sentinel file
"""

import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  prefetch  %(message)s")
log = logging.getLogger("prefetch")

MODEL_NAME = os.environ.get("MMAUDIO_MODEL", "large_44k_v2")
# The sentinel lives in the Hugging Face cache, not beside the checkpoints, on
# purpose: the CLIP and vocoder weights it vouches for are in that volume. Wipe
# the cache and the sentinel goes with it, so the next start fetches again.
# Wipe the checkpoints instead and the size fingerprint below no longer matches.
SENTINEL = Path(os.environ.get(
    "MMAUDIO_SENTINEL",
    os.path.join(os.environ.get("HF_HOME", "/cache/huggingface"), ".mmaudio-prefetch.json"),
))


def checkpoint_paths(cfg) -> list[Path]:
    """The .pth files this variant needs, in the order MMAudio downloads them."""
    candidates = [cfg.model_path, cfg.vae_path, cfg.bigvgan_16k_path, cfg.synchformer_ckpt]
    return [Path(p) for p in candidates if p is not None]


def fingerprint(paths: list[Path]) -> dict:
    """Name and size of each checkpoint. Missing files are recorded as absent."""
    return {str(p): (p.stat().st_size if p.exists() else None) for p in paths}


def already_prepared(cfg) -> bool:
    """True if a previous run left this exact set of files behind, unchanged."""
    if not SENTINEL.exists():
        return False
    try:
        recorded = json.loads(SENTINEL.read_text())
    except (OSError, ValueError):
        return False
    if recorded.get("model") != MODEL_NAME:
        return False
    return recorded.get("files") == fingerprint(checkpoint_paths(cfg))


def main() -> int:
    from mmaudio.eval_utils import all_model_cfg

    if MODEL_NAME not in all_model_cfg:
        log.error("unknown model variant %r; known: %s", MODEL_NAME, ", ".join(all_model_cfg))
        return 2
    cfg = all_model_cfg[MODEL_NAME]

    if already_prepared(cfg):
        log.info("%s is already prepared, nothing to fetch", MODEL_NAME)
        return 0

    log.info("preparing %s; the first run downloads roughly 10 GB", MODEL_NAME)
    cfg.download_if_needed()

    # Constructing the feature extractor is what pulls the CLIP and vocoder
    # weights off the Hugging Face hub. Build it on the CPU and drop it again:
    # the point is the download, not the model. The server loads its own copy
    # onto the GPU later.
    log.info("fetching the text and video encoders")
    from mmaudio.model.utils.features_utils import FeaturesUtils

    feature_utils = FeaturesUtils(
        tod_vae_ckpt=cfg.vae_path,
        synchformer_ckpt=cfg.synchformer_ckpt,
        enable_conditions=True,
        mode=cfg.mode,
        bigvgan_vocoder_ckpt=cfg.bigvgan_16k_path,
        need_vae_encoder=False,
    )
    del feature_utils

    SENTINEL.parent.mkdir(parents=True, exist_ok=True)
    SENTINEL.write_text(
        json.dumps({"model": MODEL_NAME, "files": fingerprint(checkpoint_paths(cfg))}, indent=2)
    )
    log.info("%s is ready", MODEL_NAME)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        # Exit non-zero so the entrypoint refuses to start the server. With
        # restart: unless-stopped, Docker retries, which rides out the kind of
        # transient DNS failure that cost us a generation run before.
        log.exception("could not prepare the model")
        sys.exit(1)
