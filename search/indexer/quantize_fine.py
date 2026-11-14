#!/usr/bin/env python3
"""
Convert the fine windows stored as float16 (rows written before int8 became the
format, see common/audio.py) to int8 with a float16 scale per window, in place:
half the size, the same rankings. Rows already converted are skipped, so it is
safe to interrupt and run again.

    python quantize_fine.py             # DATABASE_URL as the indexer uses it
    python quantize_fine.py --vacuum    # VACUUM FULL sound_fine afterwards: gives the space back
                                        # to the disk (a short exclusive lock on the table)
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
from psycopg2.extras import execute_batch

from common import audio, db

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
log = logging.getLogger("quantize_fine")
BATCH = 200


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--vacuum", action="store_true", help="VACUUM FULL sound_fine afterwards")
    args = parser.parse_args()
    connection = db.connect()
    db.ensure_schema(connection)
    with connection.cursor() as cursor:
        cursor.execute("SELECT COUNT(*), COALESCE(SUM(octet_length(vectors)), 0) FROM sound_fine WHERE scales IS NULL")
        todo, todo_bytes = cursor.fetchone()
    if not todo:
        log.info("every row of sound_fine is int8 already")
    else:
        log.info("%s sounds with float16 windows (%.2f GB) to convert", f"{todo:,}", todo_bytes / 1e9)
    done, started = 0, time.time()
    while todo:
        with connection.cursor() as cursor:
            cursor.execute("SELECT sound_id, array_length(offsets, 1), vectors FROM sound_fine WHERE scales IS NULL "
                           "ORDER BY sound_id LIMIT %s", (BATCH,))
            rows = cursor.fetchall()
            if not rows:
                break
            updates = []
            for sound_id, count, blob in rows:
                vectors = audio.unpack_fine(blob, None, int(count))
                packed, scales = audio.pack_fine(vectors)
                updates.append((packed, scales, sound_id))
            execute_batch(cursor, "UPDATE sound_fine SET vectors = %s, scales = %s WHERE sound_id = %s", updates)
        connection.commit()
        done += len(rows)
        if done % (BATCH * 10) == 0 or done >= todo:
            rate = done / max(time.time() - started, 1e-6)
            log.info("%s of %s converted (%.0f/s, %.0f s left)", f"{done:,}", f"{todo:,}", rate, (todo - done) / max(rate, 1e-6))
    if args.vacuum:
        log.info("VACUUM FULL sound_fine (the table is locked for a few minutes)")
        connection.rollback()            # no transaction may be open when autocommit is switched on
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute("VACUUM FULL ANALYZE sound_fine")
            cursor.execute("SELECT pg_size_pretty(pg_total_relation_size('sound_fine'))")
            log.info("sound_fine is now %s", cursor.fetchone()[0])
    connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
