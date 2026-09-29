#!/usr/bin/env python3
"""
Index a folder of your own sounds as a library.

Every audio file below the folder becomes one row: the search needs a sentence
per sound, and this script builds it from, in order of preference,

  1. a sidecar table `index.csv` in the library root, with the columns
     `file` (path relative to the root), `description` and optionally
     `category`;
  2. the file's own tags (title and comment, as written by most librarians and
     recorders);
  3. the file name and its folders, split into words: a file
     `Doors/Wooden/DOOR_Wooden_Creak_03.wav` becomes "doors wooden door wooden
     creak".

The search compares video with text, so a library whose files carry telling
names or tags works well; one full of `take_017.wav` does not.

The library folder is mounted read-only at LIBRARIES_DIR (default
/data/libraries), one subfolder per library. Rows of files that have vanished
are removed with --prune. Embeddings are computed at the end unless --no-embed
is given: the text embeddings for the search by video or prompt, and the audio
windows for the search by sound (embed_audio.py). Re-running only touches what
changed.

Usage:
    python index_folder.py mine                 # /data/libraries/mine
    python index_folder.py mine --path /some/other/folder
    python index_folder.py mine --prune
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path
from typing import Any

from common import db

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
log = logging.getLogger("index_folder")

AUDIO_SUFFIXES = {".wav", ".wave", ".aif", ".aiff", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".caf"}
LIBRARY_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def words_from_name(name: str) -> str:
    """'DOOR_Wooden-Creak.03' -> 'door wooden creak'. Numbers and take counters go."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)          # camelCase -> camel Case
    text = re.sub(r"[_\-.,;:()\[\]{}+#]+", " ", text)
    tokens = [t.lower() for t in text.split() if not re.fullmatch(r"(v|take|tk|t|no|nr)?\d+[a-z]?", t.lower())]
    return " ".join(tokens)


def read_tags(path: Path) -> tuple[str | None, float | None]:
    """(description from title/comment tags, duration in seconds), best effort."""
    try:
        from mutagen import File as MutagenFile
        audio = MutagenFile(path, easy=True)
    except Exception:
        return None, None
    if audio is None:
        return None, None
    duration = None
    try:
        duration = float(audio.info.length) if audio.info and audio.info.length else None
    except Exception:
        pass
    description = None
    tags = audio.tags or {}
    for key in ("title", "comment", "description"):
        try:
            values = tags.get(key)
        except Exception:
            values = None
        if values:
            value = values[0] if isinstance(values, (list, tuple)) else values
            value = str(value).strip()
            if value and value.lower() != path.stem.lower():
                description = value
                break
    return description, duration


def read_sidecar(root: Path) -> dict[str, dict[str, str]]:
    """index.csv -> {relative file path: {description, category}}."""
    sidecar = root / "index.csv"
    if not sidecar.exists():
        return {}
    entries: dict[str, dict[str, str]] = {}
    with sidecar.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            file = (row.get("file") or "").strip().replace("\\", "/").lstrip("./")
            if file:
                entries[file] = {"description": (row.get("description") or "").strip(),
                                 "category": (row.get("category") or "").strip()}
    log.info("index.csv describes %d files", len(entries))
    return entries


def describe(root: Path, path: Path, sidecar: dict[str, dict[str, str]]) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    folders = [words_from_name(part) for part in Path(relative).parts[:-1]]
    entry = sidecar.get(relative, {})
    tagged, duration = read_tags(path)

    description = entry.get("description") or tagged or ""
    if not description:
        description = " ".join(w for w in folders + [words_from_name(path.stem)] if w)
    category = entry.get("category") or (Path(relative).parts[0] if len(Path(relative).parts) > 1 else None)

    return {
        "external_id": relative,
        "description": description or path.stem,
        "category": category,
        "collection": Path(relative).parts[0] if len(Path(relative).parts) > 1 else None,
        "duration_seconds": duration,
        "file_path": str(path),
        "file_exists": True,
        "media_url": None,
        "extra": {"file_name": path.name, "described_by": "sidecar" if entry.get("description")
                  else "tags" if tagged else "file name"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("library", help="library name: lower-case letters, digits, - and _")
    parser.add_argument("--path", help=f"folder to index (default: {db.LIBRARIES_DIR}/<library>)")
    parser.add_argument("--prune", action="store_true", help="remove rows whose file is gone")
    parser.add_argument("--no-embed", action="store_true", help="skip the embedding step")
    args = parser.parse_args()

    if not LIBRARY_NAME.match(args.library) or args.library == "bbc":
        parser.error("library name must match [a-z0-9][a-z0-9_-]* and not be 'bbc'")
    root = Path(args.path) if args.path else db.LIBRARIES_DIR / args.library
    if not root.is_dir():
        parser.error(f"{root} is not a folder (is LIBRARIES_HOST_DIR mounted?)")

    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES)
    log.info("library '%s': %d audio files under %s", args.library, len(files), root)
    sidecar = read_sidecar(root)

    connection = db.connect()
    db.ensure_schema(connection)

    rows, batch = 0, []
    for path in files:
        row = describe(root, path, sidecar)
        row["library"] = args.library
        batch.append(row)
        if len(batch) >= 500:
            rows += db.upsert_sounds(connection, batch)
            batch.clear()
            log.info("%d / %d rows written", rows, len(files))
    rows += db.upsert_sounds(connection, batch)
    log.info("%d rows written", rows)

    if args.prune:
        removed = db.delete_missing(connection, args.library, {r.relative_to(root).as_posix() for r in files})
        log.info("%d rows removed (files no longer present)", removed)

    if not args.no_embed and rows:
        from embed import embed_missing
        from embed_audio import embed_pending
        embed_missing(connection, library=args.library)
        embed_pending(connection, library=args.library)

    for stat in db.library_stats(connection):
        if stat["library"] == args.library:
            log.info("library '%s': %d sounds, %d embedded", args.library, stat["sounds"], stat["embedded"])
    connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
