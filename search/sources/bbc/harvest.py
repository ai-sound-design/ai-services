#!/usr/bin/env python3
"""
Harvest BBC Sound Effects metadata into the sound index (library 'bbc').

Reads the public search API of the BBC Sound Effects archive and upserts every
record into the `sounds` table with a media URL, so the search API can fetch
the audio on demand. No audio is downloaded here; download.py does that for
offline use.

The API refuses any request where from + size exceeds 1000, so plain offset
paging can only ever reach the first 1000 of the ~33k sounds. This script works
around that by slicing the archive along the duration filter: a slice with at
most 1000 hits is fetched in one request, a larger slice is split in half and
retried. Slices that cannot be split further fall back to slicing by category.

The archive is free to use for personal, educational and research purposes under
the BBC's licensing terms. Check https://sound-effects.bbcrewind.co.uk/licensing
before using any of it.

Usage:
    python sources/bbc/harvest.py                 # all ~33k records
    python sources/bbc/harvest.py --limit 2000    # a subset, useful for a quick test
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
import urllib.error
import urllib.request

from common import db

API_URL = "https://sound-effects-api.bbcrewind.co.uk/api/sfx/search"
MEDIA_BASE = "https://sound-effects-media.bbcrewind.co.uk"
USER_AGENT = "ai-sound-services-indexer"
LIBRARY = "bbc"
AUDIO_FORMAT = os.getenv("BBC_AUDIO_FORMAT", "mp3").lower()
AUDIO_DIR = db.AUDIO_DIR / LIBRARY

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
log = logging.getLogger("harvest")

MAX_WINDOW = 1000          # hard limit of the API: from + size <= 1000
MAX_DURATION = 99999       # wide enough to cover the longest recording

CATEGORIES = [
    "Aircraft", "Animals", "Applause", "Atmosphere", "Bells", "Birds", "Clocks",
    "Comedy", "Crowds", "Daily_Life", "Destruction", "Electronics", "Events",
    "Fire", "Footsteps", "Industry", "Machines", "Medical", "Military", "Nature",
    "Sport", "Toys", "Transport",
]


def query(offset: int, size: int, duration: tuple[int, int] | None = None,
          categories: list[str] | None = None, retries: int = 4) -> tuple[int, list[dict]]:
    """One search request. Returns (total_for_this_filter, results)."""
    criteria = {
        "from": offset, "size": size,
        "tags": None, "categories": categories,
        "durations": [{"min": duration[0], "max": duration[1]}] if duration else None,
        "continents": None, "sortBy": None, "source": None, "habitats": None,
    }
    request = urllib.request.Request(
        API_URL, data=json.dumps({"criteria": criteria}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.load(response)
            return body.get("total", 0), body.get("results") or []
        except Exception as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"search failed for {duration} {categories}: {exc}") from exc
            time.sleep(2 ** attempt)
    return 0, []


def slices(duration: tuple[int, int],
           categories: list[str] | None = None) -> "list[tuple[tuple[int,int], list[str] | None]]":
    """Split the archive into chunks the API is willing to return in full."""
    low, high = duration
    total, _ = query(0, 1, duration, categories)

    if total == 0:
        return []
    if total <= MAX_WINDOW:
        return [(duration, categories)]

    # Durations are fractional seconds, so the two halves must share their
    # boundary. A gap like [low, mid] + [mid + 1, high] would silently drop
    # everything between mid and mid + 1; the overlap is removed by the
    # de-duplication in main().
    if high - low > 1:
        middle = low + (high - low) // 2
        return slices((low, middle), categories) + slices((middle, high), categories)

    # A single duration value with more than 1000 hits: slice by category instead.
    if categories is None:
        log.info("duration %ds holds %d sounds, slicing it by category", low, total)
        out = []
        for category in CATEGORIES:
            out.extend(slices(duration, [category]))
        return out

    log.warning("cannot split further: duration %ds, categories %s, %d hits (capping at %d)",
                low, categories, total, MAX_WINDOW)
    return [(duration, categories)]


def media_url(sound_id: str) -> str:
    return f"{MEDIA_BASE}/mp3/{sound_id}.mp3"


def to_row(record: dict) -> dict | None:
    """Map one API record onto a `sounds` row. None if unusable."""
    sound_id = record.get("id")
    description = (record.get("description") or "").strip()
    if not sound_id or not description:
        return None

    technical = record.get("technicalMetadata") or {}
    # technicalMetadata.duration is seconds as a string; the top-level
    # duration field is milliseconds. Prefer the former.
    try:
        duration = float(technical.get("duration"))
    except (TypeError, ValueError):
        try:
            duration = float(record.get("duration", 0)) / 1000.0
        except (TypeError, ValueError):
            duration = 0.0

    categories = record.get("categories") or []
    sub_categories = record.get("subCategories") or []
    top_category = (categories[0] or {}).get("className") if categories else None
    sub_category = (sub_categories[0] or {}).get("className") if sub_categories else None

    local_file = AUDIO_DIR / f"{sound_id}.{AUDIO_FORMAT}"
    return {
        "library": LIBRARY,
        "external_id": str(sound_id),
        "description": description,
        "category": sub_category or top_category,
        "collection": top_category,
        "duration_seconds": duration,
        "file_path": str(local_file) if local_file.exists() else None,
        "file_exists": local_file.exists(),
        "media_url": media_url(str(sound_id)),
        "extra": {k: v for k, v in {"source": record.get("source"),
                                    "location": technical.get("file_name")}.items() if v},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after roughly this many records (0 = all)")
    args = parser.parse_args()

    connection = db.connect()
    db.ensure_schema(connection)
    log.info("connected to the database")

    archive_total, _ = query(0, 1)
    log.info("archive holds %d sounds", archive_total)

    log.info("planning slices (the API caps every request at from + size <= %d)", MAX_WINDOW)
    plan = slices((0, MAX_DURATION))
    log.info("%d slices to fetch", len(plan))

    seen: set[str] = set()
    imported = skipped = 0

    for index, (duration, categories) in enumerate(plan, start=1):
        total, results = query(0, MAX_WINDOW, duration, categories)
        rows = []
        for record in results:
            row = to_row(record)
            if not row:
                skipped += 1
                continue
            if row["external_id"] in seen:          # overlapping slices are expected
                continue
            seen.add(row["external_id"])
            rows.append(row)

        imported += db.upsert_sounds(connection, rows)

        if index % 10 == 0 or index == len(plan):
            log.info("slice %d / %d, %d unique sounds imported", index, len(plan), imported)

        if args.limit and imported >= args.limit:
            log.info("reached the requested limit of %d", args.limit)
            break

    stats = next((s for s in db.library_stats(connection) if s["library"] == LIBRARY), None)
    connection.close()

    log.info("done: %d imported this run, %d records skipped as unusable", imported, skipped)
    if stats:
        log.info("library 'bbc' now holds %d sounds, %d with audio on disk, %d embedded",
                 stats["sounds"], stats["local_files"], stats["embedded"])
        if not args.limit and stats["sounds"] < archive_total * 0.95:
            log.warning("only %d of %d sounds were reached; some slices may have been capped",
                        stats["sounds"], archive_total)
    log.info("next: python embed.py --library bbc   (the search fetches audio on demand; "
             "run sources/bbc/download.py for offline use)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
