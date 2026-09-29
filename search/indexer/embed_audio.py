#!/usr/bin/env python3
"""
Compute CLAP audio embeddings for the sounds that are on disk.

Every file is cut into overlapping 10 s windows (5 s hop, see common/audio.py)
and each window gets one embedding in `sound_windows`. The search's
/search/by_audio then finds, for a given sound, the library windows that sound
alike: the answer is "file X from second 140", not just "file X".

Only sounds with a local file can be embedded (the audio has to be read), so
for the BBC archive run sources/bbc/download.py first. Processes sounds that
have no windows yet, so it is safe to interrupt and rerun; the HNSW index is
declared in schema.sql and grows with every insert, so the partial index is
searchable while this runs.

Usage:
    python embed_audio.py                       # every library
    python embed_audio.py --library mine        # one library
    python embed_audio.py --rebuild             # recompute windows that already exist
    python embed_audio.py --limit 500           # stop after 500 files (for a first look)

Time: roughly one to three hours per 300 hours of audio on a laptop GPU;
decoding the files is the larger part.
"""
from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from psycopg2.extras import execute_batch

from common import audio, db

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
log = logging.getLogger("embed_audio")

INSERT = """
    INSERT INTO sound_windows (sound_id, offset_seconds, length_seconds, audio_embedding)
    VALUES (%s, %s, %s, %s::vector)
    ON CONFLICT (sound_id, offset_seconds) DO UPDATE SET
        audio_embedding = EXCLUDED.audio_embedding, length_seconds = EXCLUDED.length_seconds
"""


def pending_sounds(connection, library: str | None, rebuild: bool, limit: int | None) -> list[tuple]:
    conditions = ["file_exists", "file_path IS NOT NULL"]
    params: list = []
    if not rebuild:
        conditions.append("NOT EXISTS (SELECT 1 FROM sound_windows w WHERE w.sound_id = sounds.id)")
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


def load_windows(row: tuple) -> tuple[int, list[tuple[float, float, "np.ndarray"]] | None, str | None]:
    """Decode one file and cut its windows: (sound_id, [(offset, length, samples)], error)."""
    sound_id, path, _ = row
    try:
        samples = audio.decode(path)
    except Exception as exc:  # noqa: BLE001  (a broken file must not stop the run)
        return sound_id, None, f"{Path(path).name}: {str(exc)[:160]}"
    duration = len(samples) / audio.SAMPLE_RATE
    if duration < 0.1:
        return sound_id, None, f"{Path(path).name}: no audio"
    windows = []
    for offset in audio.window_offsets(duration):
        piece = audio.slice_window(samples, offset)
        windows.append((offset, round(len(piece) / audio.SAMPLE_RATE, 3), piece))
    return sound_id, windows, None


def embed_pending(connection, library: str | None = None, rebuild: bool = False,
                  limit: int | None = None, batch_windows: int = 32, decoders: int = 4) -> int:
    rows = pending_sounds(connection, library, rebuild, limit)
    if not rows:
        log.info("every local sound already has audio windows")
        return 0
    hours = sum(float(r[2] or 0) for r in rows) / 3600
    log.info("embedding %d files (%.1f hours of audio)", len(rows), hours)
    encoder = audio.ClapEncoder()
    started = time.time()
    done_files = done_windows = failed = 0
    with ThreadPoolExecutor(max_workers=decoders) as pool:
        # A bounded prefetch: pool.map would decode far ahead of the GPU and keep every
        # decoded file in memory (a long recording is hundreds of megabytes as float32).
        def decoded():
            pending = []
            for row in rows:
                pending.append(pool.submit(load_windows, row))
                if len(pending) > decoders:
                    yield pending.pop(0).result()
            for future in pending:
                yield future.result()

        for sound_id, windows, error in decoded():
            if error:
                failed += 1
                log.warning("skipped %s", error)
                continue
            vectors = []
            for i in range(0, len(windows), batch_windows):
                vectors.extend(encoder.encode_audio([w[2] for w in windows[i:i + batch_windows]]).tolist())
            with connection.cursor() as cursor:
                if rebuild:
                    cursor.execute("DELETE FROM sound_windows WHERE sound_id = %s", (sound_id,))
                execute_batch(cursor, INSERT,
                              [(sound_id, offset, length, str(vector))
                               for (offset, length, _), vector in zip(windows, vectors)], page_size=200)
            connection.commit()
            done_files += 1
            done_windows += len(windows)
            if done_files % 100 == 0 or done_files == len(rows) - failed:
                elapsed = time.time() - started
                rate = done_files / max(elapsed, 1e-6)
                remaining = (len(rows) - failed - done_files) / max(rate, 1e-6)
                log.info("%d / %d files, %d windows (%.1f files/s, about %.0f min left)",
                         done_files, len(rows), done_windows, rate, remaining / 60)
    log.info("done: %d files, %d windows in %.1f minutes; %d files skipped",
             done_files, done_windows, (time.time() - started) / 60, failed)
    return done_files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--library", help="only this library")
    parser.add_argument("--rebuild", action="store_true", help="recompute windows that already exist")
    parser.add_argument("--limit", type=int, help="stop after this many files")
    parser.add_argument("--batch-windows", type=int, default=32, help="windows per model call")
    parser.add_argument("--decoders", type=int, default=4, help="parallel ffmpeg decoders")
    args = parser.parse_args()

    connection = db.connect()
    db.ensure_schema(connection)
    embed_pending(connection, args.library, args.rebuild, args.limit, args.batch_windows, args.decoders)
    connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
