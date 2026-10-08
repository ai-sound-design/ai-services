#!/usr/bin/env python3
"""
Register an existing copy of the BBC Sound Effects archive instead of downloading it.

Points the `bbc` rows at the files of an archive folder on disk, which is mounted
read-only: every file whose name is a BBC id (`07011191.wav`, `07011191.mp3`,
in any subfolder) becomes the audio of that sound. Nothing is copied or written
to the folder; the search reads the files where they are. Rows without a file in
the folder stay as they were (fetched on demand, or downloaded by download.py).

Run harvest.py first, so the rows exist. Safe to run again after the folder
changed.

Usage:
    python sources/bbc/register.py                    # the folder at BBC_ARCHIVE_DIR (/data/archive)
    python sources/bbc/register.py --path /mnt/bbc    # another folder inside the container
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from psycopg2.extras import execute_batch

from common import db

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
log = logging.getLogger("bbc.register")

ARCHIVE_DIR = Path(os.getenv("BBC_ARCHIVE_DIR", "/data/archive"))
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".aif", ".aiff", ".ogg", ".m4a"}
PREFERENCE = [".wav", ".flac", ".aiff", ".aif", ".mp3", ".ogg", ".m4a"]       # when an id comes in two formats


def scan(folder: Path) -> dict[str, Path]:
    """BBC id -> file, the best format of each."""
    found: dict[str, Path] = {}
    for root, _, files in os.walk(folder):
        for name in files:
            path = Path(root) / name
            if path.suffix.lower() not in AUDIO_SUFFIXES:
                continue
            stem = path.stem
            if stem not in found or PREFERENCE.index(path.suffix.lower()) < PREFERENCE.index(found[stem].suffix.lower()):
                found[stem] = path
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", default=str(ARCHIVE_DIR), help="the archive folder inside the container")
    args = parser.parse_args()
    folder = Path(args.path)
    if not folder.is_dir():
        log.error("%s is not a folder", folder)
        return 1
    log.info("scanning %s", folder)
    found = scan(folder)
    log.info("%d audio files named like BBC ids", len(found))
    if not found:
        return 0

    connection = db.connect()
    db.ensure_schema(connection)
    with connection.cursor() as cursor:
        cursor.execute("SELECT id, external_id, file_path, file_exists FROM sounds WHERE library = 'bbc'")
        rows = cursor.fetchall()
        if not rows:
            log.error("no BBC rows yet: run sources/bbc/harvest.py first")
            return 1
        updates = []
        for sound_id, external_id, file_path, file_exists in rows:
            path = found.get(external_id)
            if path and (not file_exists or file_path != str(path)):
                updates.append((str(path), sound_id))
        execute_batch(cursor, "UPDATE sounds SET file_path = %s, file_exists = TRUE, updated_at = CURRENT_TIMESTAMP "
                              "WHERE id = %s", updates, page_size=500)
        matched = sum(1 for _, external_id, _, _ in rows if external_id in found)
    connection.commit()
    connection.close()
    log.info("done: %d of %d BBC sounds have a file in the archive folder (%d rows updated)", matched, len(rows), len(updates))
    if matched < len(rows):
        log.info("the other %d are fetched on demand, or downloaded by sources/bbc/download.py", len(rows) - matched)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
