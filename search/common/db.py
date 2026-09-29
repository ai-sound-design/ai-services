"""
Database access shared by the search API and the indexer scripts.

Everything speaks to one table, `sounds` (see schema.sql). A row is identified
by (library, external_id); sources upsert rows, embed.py fills the vector
columns, the API reads them.

Environment:
    DATABASE_URL    postgresql://user:password@host:5432/dbname
    AUDIO_DIR       where fetched audio is cached, one subfolder per library
                    (default /data/audio)
    LIBRARIES_DIR   root of the user's own libraries, mounted read-only
                    (default /data/libraries)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterable

import psycopg2
import psycopg2.errors
from psycopg2.extras import Json, RealDictCursor, execute_batch

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://sound_user:change_me@postgres:5432/sounds")
AUDIO_DIR = Path(os.getenv("AUDIO_DIR", "/data/audio"))
LIBRARIES_DIR = Path(os.getenv("LIBRARIES_DIR", "/data/libraries"))
SCHEMA_FILE = Path(__file__).with_name("schema.sql")

EMBEDDING_COLUMNS = {
    "base":  {"model": "microsoft/xclip-base-patch32", "column": "text_embedding",       "dim": 512, "index": "idx_text_emb"},
    "large": {"model": "microsoft/xclip-large-patch14", "column": "text_embedding_large", "dim": 768, "index": "idx_text_emb_large"},
}

log = logging.getLogger("db")


def connect(url: str | None = None):
    """A connection with autocommit off; callers commit."""
    return psycopg2.connect(url or DATABASE_URL)


# ── Schema ───────────────────────────────────────────────────────────────────

def ensure_schema(connection) -> None:
    """Create or update the tables, then carry over a legacy BBC-only database."""
    with connection.cursor() as cursor:
        # CREATE OR REPLACE VIEW needs an exclusive lock; a client that left a
        # transaction open would make this wait forever. Fail loudly instead.
        cursor.execute("SET lock_timeout = '30s'")
        try:
            cursor.execute(SCHEMA_FILE.read_text(encoding="utf-8"))
        except psycopg2.errors.LockNotAvailable as exc:
            connection.rollback()
            raise RuntimeError("the sound index is locked by another connection (an API or indexer "
                               "still holding a transaction); retry in a moment") from exc
        cursor.execute("SET lock_timeout = 0")
    connection.commit()
    _migrate_legacy_bbc_table(connection)


def _migrate_legacy_bbc_table(connection) -> None:
    """
    Earlier versions kept a BBC-specific table `bbc_sounds`. Copy it into
    `sounds` once (library 'bbc'), keeping the embeddings so nothing has to be
    recomputed. The old table is left in place; drop it when you are satisfied.
    """
    with connection.cursor() as cursor:
        cursor.execute("SELECT to_regclass('public.bbc_sounds') IS NOT NULL")
        legacy = cursor.fetchone()[0]
        if legacy:
            cursor.execute("SELECT COUNT(*) FROM sounds WHERE library = 'bbc'")
            legacy = cursor.fetchone()[0] == 0
        if not legacy:
            connection.commit()           # end the read transaction; nothing to do
            return
        log.info("migrating the legacy bbc_sounds table into sounds")
        cursor.execute(
            """
            INSERT INTO sounds (library, external_id, description, category, collection,
                                duration_seconds, file_path, file_exists, media_url, extra,
                                text_embedding, text_embedding_large)
            SELECT 'bbc',
                   regexp_replace(location, '\\.[^.]+$', ''),
                   description, category, cdname, duration_seconds,
                   replace(file_path, '/data/bbc-audio/', %s),
                   file_exists,
                   'https://sound-effects-media.bbcrewind.co.uk/mp3/'
                       || regexp_replace(location, '\\.[^.]+$', '') || '.mp3',
                   jsonb_strip_nulls(jsonb_build_object('source', cdnumber, 'location', location)),
                   text_embedding, text_embedding_large
            FROM bbc_sounds
            ON CONFLICT (library, external_id) DO NOTHING
            """,
            (str(AUDIO_DIR / "bbc") + "/",),
        )
        migrated = cursor.rowcount
    connection.commit()
    log.info("migrated %d sounds from bbc_sounds; the old table can be dropped", migrated)
    for spec in EMBEDDING_COLUMNS.values():
        build_vector_index(connection, spec)


def build_vector_index(connection, spec: dict) -> int:
    """(Re)build the IVFFlat index of one embedding column. Returns the vector count."""
    column, index = spec["column"], spec["index"]
    with connection.cursor() as cursor:
        cursor.execute(f"SELECT COUNT(*) FROM sounds WHERE {column} IS NOT NULL")
        count = cursor.fetchone()[0]
        if count == 0:
            return 0
        # lists ~ sqrt(rows) is the usual rule of thumb.
        lists = max(10, min(1000, int(count ** 0.5)))
        cursor.execute(f"DROP INDEX IF EXISTS {index}")
        cursor.execute(f"CREATE INDEX {index} ON sounds USING ivfflat ({column} vector_cosine_ops) "
                       f"WITH (lists = {lists})")
        cursor.execute("ANALYZE sounds")
    connection.commit()
    return count


# ── Rows ─────────────────────────────────────────────────────────────────────

UPSERT = """
    INSERT INTO sounds (library, external_id, description, category, collection,
                        duration_seconds, file_path, file_exists, media_url, extra)
    VALUES (%(library)s, %(external_id)s, %(description)s, %(category)s, %(collection)s,
            %(duration_seconds)s, %(file_path)s, %(file_exists)s, %(media_url)s, %(extra)s)
    ON CONFLICT (library, external_id) DO UPDATE SET
        description      = EXCLUDED.description,
        category         = EXCLUDED.category,
        collection       = EXCLUDED.collection,
        duration_seconds = EXCLUDED.duration_seconds,
        file_path        = COALESCE(EXCLUDED.file_path, sounds.file_path),
        file_exists      = EXCLUDED.file_exists OR sounds.file_exists,
        media_url        = COALESCE(EXCLUDED.media_url, sounds.media_url),
        extra            = sounds.extra || EXCLUDED.extra,
        -- a changed description needs new embeddings
        text_embedding       = CASE WHEN sounds.description = EXCLUDED.description THEN sounds.text_embedding END,
        text_embedding_large = CASE WHEN sounds.description = EXCLUDED.description THEN sounds.text_embedding_large END,
        updated_at       = CURRENT_TIMESTAMP
"""


def upsert_sounds(connection, rows: Iterable[dict[str, Any]]) -> int:
    """Insert or update rows; each needs library, external_id and description."""
    prepared = []
    for row in rows:
        prepared.append({
            "library": row["library"],
            "external_id": str(row["external_id"]),
            "description": row["description"],
            "category": row.get("category"),
            "collection": row.get("collection"),
            "duration_seconds": row.get("duration_seconds"),
            "file_path": row.get("file_path"),
            "file_exists": bool(row.get("file_exists", False)),
            "media_url": row.get("media_url"),
            "extra": Json(row.get("extra") or {}),
        })
    if not prepared:
        return 0
    with connection.cursor() as cursor:
        execute_batch(cursor, UPSERT, prepared, page_size=500)
    connection.commit()
    return len(prepared)


def mark_file(connection, sound_id: int, path: Path | str) -> None:
    """Record that the audio of a sound is on disk at `path`."""
    with connection.cursor() as cursor:
        cursor.execute("UPDATE sounds SET file_path = %s, file_exists = TRUE, updated_at = CURRENT_TIMESTAMP "
                       "WHERE id = %s", (str(path), sound_id))
    connection.commit()


def delete_missing(connection, library: str, present_ids: set[str]) -> int:
    """Remove rows of a library whose external_id is not in `present_ids`."""
    with connection.cursor() as cursor:
        cursor.execute("SELECT id, external_id FROM sounds WHERE library = %s", (library,))
        gone = [row[0] for row in cursor.fetchall() if row[1] not in present_ids]
        if gone:
            cursor.execute("DELETE FROM sounds WHERE id = ANY(%s)", (gone,))
    connection.commit()
    return len(gone)


def library_stats(connection) -> list[dict[str, Any]]:
    with connection.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute("SELECT * FROM library_stats")
        return [dict(row) for row in cursor.fetchall()]
