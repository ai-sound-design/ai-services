#!/usr/bin/env python3
"""
Download the BBC Sound Effects audio for offline use.

Not required for searching: the search API fetches a sound from the BBC the
first time it is previewed or imported (AUDIO_FETCH=on_demand). Run this to
have the whole archive on disk, for a machine without internet access or with
AUDIO_FETCH=local_only.

Files are fetched concurrently and skipped when already on disk, so the script
can be interrupted and resumed.

Formats:
    mp3   ~0.3 MB per sound, about 10 GB for the full archive (default)
    wav   ~2.2 MB per sound compressed, about 73 GB for the full archive

Licensing: the archive is free for personal, educational and research use under
the BBC's terms. See https://sound-effects.bbcrewind.co.uk/licensing

Usage:
    python sources/bbc/download.py --limit 2000        # a subset for a first test
    python sources/bbc/download.py                     # everything in the table
    python sources/bbc/download.py --format wav
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import random
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common import db

MEDIA_BASE = "https://sound-effects-media.bbcrewind.co.uk"
USER_AGENT = "ai-sound-services-indexer"
LIBRARY = "bbc"
AUDIO_DIR = db.AUDIO_DIR / LIBRARY

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
log = logging.getLogger("download")


def fetch_one(sound_id: int, external_id: str, audio_format: str, retries: int = 3) -> tuple[int, Path | None, str]:
    """Download a single sound. Returns (row id, path or None, note)."""
    target = AUDIO_DIR / f"{external_id}.{audio_format}"
    if target.exists() and target.stat().st_size > 0:
        return sound_id, target, "already present"

    url = (f"{MEDIA_BASE}/mp3/{external_id}.mp3" if audio_format == "mp3"
           else f"{MEDIA_BASE}/zip/{external_id}.wav.zip")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read()

            temporary = target.with_suffix(target.suffix + ".part")
            if audio_format == "wav":
                # The archive serves WAV inside a zip container.
                with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                    names = [n for n in archive.namelist() if n.lower().endswith(".wav")]
                    if not names:
                        return sound_id, None, "zip contained no wav"
                    temporary.write_bytes(archive.read(names[0]))
            else:
                temporary.write_bytes(payload)

            temporary.replace(target)
            return sound_id, target, f"{target.stat().st_size} bytes"

        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return sound_id, None, "not available (404)"
            if attempt == retries - 1:
                return sound_id, None, f"HTTP {exc.code}"
            time.sleep(2 ** attempt + random.random())
        except Exception as exc:  # network hiccups, truncated zips
            if attempt == retries - 1:
                return sound_id, None, str(exc)[:80]
            # The server throttles bursts, so wait before trying again.
            time.sleep(2 ** attempt + random.random())
    return sound_id, None, "retries exhausted"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=0, help="download at most this many (0 = all)")
    parser.add_argument("--format", choices=("mp3", "wav"),
                        default=os.getenv("BBC_AUDIO_FORMAT", "mp3"))
    parser.add_argument("--workers", type=int, default=8, help="parallel downloads")
    args = parser.parse_args()

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    connection = db.connect()
    db.ensure_schema(connection)

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT id, external_id FROM sounds WHERE library = %s AND NOT file_exists ORDER BY id"
            + (" LIMIT %s" if args.limit else ""),
            (LIBRARY, args.limit) if args.limit else (LIBRARY,),
        )
        pending = cursor.fetchall()

    if not pending:
        log.info("nothing to download; every row already points at an existing file")
        connection.close()
        return 0

    log.info("downloading %d sounds as %s into %s", len(pending), args.format, AUDIO_DIR)

    failures: dict[str, int] = {}
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, sound_id, external_id, args.format): sound_id
                   for sound_id, external_id in pending}
        for done, future in enumerate(as_completed(futures), start=1):
            sound_id, path, note = future.result()
            if path is not None:
                db.mark_file(connection, sound_id, path)
            else:
                failed += 1
                reason = "not available (404)" if "404" in note else note[:40]
                failures[reason] = failures.get(reason, 0) + 1
                log.debug("failed %s: %s", sound_id, note)

            if done % 200 == 0 or done == len(pending):
                log.info("%d / %d processed, %d failed", done, len(pending), failed)

    stats = next((s for s in db.library_stats(connection) if s["library"] == LIBRARY), None)
    connection.close()

    if stats:
        log.info("done: %d of %d BBC sounds now on disk, %d downloads failed",
                 stats["local_files"], stats["sounds"], failed)
    for reason, count in sorted(failures.items(), key=lambda item: -item[1]):
        log.info("  %5d x %s", count, reason)
    if failed:
        log.info("run this script again to retry them; files already on disk are skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
