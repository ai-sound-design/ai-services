#!/usr/bin/env python3
"""
Build the adaptive audio index for the sounds that are on disk.

Every file is cut into 2 s CLAP windows on a 1 s grid (common/audio.py). Windows
that follow each other and sound alike are merged into one segment, whose vector
is the mean of theirs; a window that sounds different starts a new segment, a
pause ends one. The segments go to `sound_segments` (halfvec, HNSW index) and are
what /search/by_audio searches; the 2 s windows themselves go to `sound_fine`
(one row per sound, not indexed), so the search can find the best stretch inside
a matched segment without decoding the file again.

Processes sounds without a `sound_fine` row, so it is safe to interrupt and
rerun. The search uses the segments once every sound with 10 s windows has them
(AUDIO_INDEX=auto); until then it stays on the 10 s windows.

Usage:
    python embed_segments.py                    # every library
    python embed_segments.py --library mine     # one library
    python embed_segments.py --rebuild          # recompute sounds that already have segments
    python embed_segments.py --limit 500        # stop after 500 files (for a first look)
    python embed_segments.py --ids 8025,2513    # these sounds now, ahead of the rest

Time: about three seconds per BBC file on a laptop GPU, a day for the archive.
"""
from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from psycopg2.extras import execute_batch

from common import audio, db

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
log = logging.getLogger("embed_segments")

INSERT_SEGMENT = """
    INSERT INTO sound_segments (sound_id, start_seconds, end_seconds, audio_embedding)
    VALUES (%s, %s, %s, %s::halfvec)
    ON CONFLICT (sound_id, start_seconds) DO UPDATE SET
        end_seconds = EXCLUDED.end_seconds, audio_embedding = EXCLUDED.audio_embedding
"""
INSERT_FINE = """
    INSERT INTO sound_fine (sound_id, window_seconds, offsets, levels, vectors)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (sound_id) DO UPDATE SET
        window_seconds = EXCLUDED.window_seconds, offsets = EXCLUDED.offsets,
        levels = EXCLUDED.levels, vectors = EXCLUDED.vectors
"""


def pending_sounds(connection, library: str | None, rebuild: bool, limit: int | None,
                   ids: list[int] | None = None) -> list[tuple]:
    conditions = ["file_exists", "file_path IS NOT NULL"]
    params: list = []
    if ids:
        conditions.append("id = ANY(%s)")
        params.append(ids)
    if not rebuild:
        conditions.append("NOT EXISTS (SELECT 1 FROM sound_fine f WHERE f.sound_id = sounds.id)")
    if library:
        conditions.append("library = %s")
        params.append(library)
    sql = f"SELECT id, file_path, duration_seconds FROM sounds WHERE {' AND '.join(conditions)} ORDER BY id"
    if limit:
        sql += " LIMIT %s"
        params.append(limit)
    with connection.cursor() as cursor:
        cursor.execute(sql, params)
        return cursor.fetchall()


def load_windows(row: tuple) -> tuple[int, float, list[float], list, list[float], str | None]:
    """Decode one file and cut its 2 s windows: (sound_id, duration, offsets, pieces, levels, error)."""
    sound_id, path, _ = row
    try:
        samples = audio.decode(path)
    except Exception as exc:  # noqa: BLE001  (a broken file must not stop the run)
        return sound_id, 0.0, [], [], [], f"{Path(path).name}: {str(exc)[:160]}"
    duration = len(samples) / audio.SAMPLE_RATE
    if duration < 0.1:
        return sound_id, 0.0, [], [], [], f"{Path(path).name}: no audio"
    offsets = audio.fine_offsets(duration)
    pieces = [audio.slice_window(samples, o, audio.FINE_SECONDS) for o in offsets]
    levels = [round(audio.rms_db(p), 2) for p in pieces]
    return sound_id, duration, offsets, pieces, levels, None


def embed_pending(connection, library: str | None = None, rebuild: bool = False,
                  limit: int | None = None, batch_windows: int = 64, decoders: int = 4,
                  ids: list[int] | None = None) -> int:
    rows = pending_sounds(connection, library, rebuild, limit, ids)
    if not rows:
        log.info("every local sound already has segments")
        return 0
    hours = sum(float(r[2] or 0) for r in rows) / 3600
    log.info("segmenting %d files (%.1f hours of audio, threshold %.2f)", len(rows), hours, audio.SEGMENT_THRESHOLD)
    encoder = audio.ClapEncoder()
    started = time.time()
    done_files = done_segments = done_windows = failed = 0
    with ThreadPoolExecutor(max_workers=decoders) as pool:
        # Bounded prefetch, as in embed_audio.py: decoded files are large as float32.
        def decoded():
            pending = []
            for row in rows:
                pending.append(pool.submit(load_windows, row))
                if len(pending) > decoders:
                    yield pending.pop(0).result()
            for future in pending:
                yield future.result()

        for sound_id, duration, offsets, pieces, levels, error in decoded():
            if error:
                failed += 1
                log.warning("skipped %s", error)
                continue
            vectors = np.concatenate([encoder.encode_audio(pieces[i:i + batch_windows])
                                      for i in range(0, len(pieces), batch_windows)])
            found = audio.segments(offsets, vectors, levels, duration)
            with connection.cursor() as cursor:
                if rebuild:
                    cursor.execute("DELETE FROM sound_segments WHERE sound_id = %s", (sound_id,))
                execute_batch(cursor, INSERT_SEGMENT,
                              [(sound_id, start, end, str([round(float(x), 5) for x in vector]))
                               for start, end, vector in found], page_size=200)
                cursor.execute(INSERT_FINE, (sound_id, audio.FINE_SECONDS, [float(o) for o in offsets], levels,
                                             vectors.astype(np.float16).tobytes()))
            connection.commit()
            done_files += 1
            done_segments += len(found)
            done_windows += len(offsets)
            if done_files % 100 == 0 or done_files == len(rows) - failed:
                elapsed = time.time() - started
                rate = done_files / max(elapsed, 1e-6)
                remaining = (len(rows) - failed - done_files) / max(rate, 1e-6)
                log.info("%d / %d files, %d segments from %d windows (%.2f files/s, about %.1f h left)",
                         done_files, len(rows), done_segments, done_windows, rate, remaining / 3600)
    log.info("done: %d files, %d segments from %d windows in %.1f hours; %d files skipped",
             done_files, done_segments, done_windows, (time.time() - started) / 3600, failed)
    return done_files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--library", help="only this library")
    parser.add_argument("--rebuild", action="store_true", help="recompute sounds that already have segments")
    parser.add_argument("--limit", type=int, help="stop after this many files")
    parser.add_argument("--ids", help="only these sound ids, comma-separated (to try a few ahead of the rest)")
    parser.add_argument("--batch-windows", type=int, default=64, help="windows per model call")
    parser.add_argument("--decoders", type=int, default=4, help="parallel ffmpeg decoders")
    args = parser.parse_args()

    connection = db.connect()
    db.ensure_schema(connection)
    ids = [int(x) for x in (args.ids or "").split(",") if x.strip().isdigit()] or None
    embed_pending(connection, args.library, args.rebuild, args.limit, args.batch_windows, args.decoders, ids)
    connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
