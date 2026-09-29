"""
Read side of the sound index for the search API.

Works on the `sounds` table (search/schema.sql): every library in one table,
distinguished by the `library` column. Writes are limited to recording where a
fetched audio file was stored.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from psycopg2.extras import RealDictCursor

from common import db

logger = logging.getLogger(__name__)

RESULT_COLUMNS = "id, library, external_id, description, category, collection, duration_seconds, file_exists"


class DatabaseClient:
    """PostgreSQL client for the sound index."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.embedding_column = "text_embedding"
        self.conn = db.connect(database_url)
        # Reads only (plus the occasional mark_file, which commits itself). With
        # autocommit no transaction stays open between requests, so an indexer
        # running CREATE OR REPLACE VIEW is not blocked by this connection's
        # last SELECT, which would otherwise hold its lock indefinitely.
        self.conn.autocommit = True
        db.ensure_schema(self.conn)
        with self.conn.cursor() as cursor:
            cursor.execute("SET hnsw.ef_search = 120")     # wider candidate list for the window search
        logger.info("Database connection established (using %s)", self.embedding_column)

    def set_embedding_column(self, embedding_dim: int) -> None:
        """Pick the vector column that matches the loaded model (512 base, 768 large)."""
        self.embedding_column = "text_embedding_large" if embedding_dim == 768 else "text_embedding"
        logger.info("Switched to embedding column: %s", self.embedding_column)

    # ── Statistics ───────────────────────────────────────────────────────────

    def libraries(self) -> list[dict[str, Any]]:
        """One entry per library: sounds, embedded, local_files, hours."""
        stats = db.library_stats(self.conn)
        for entry in stats:
            entry["hours"] = round(float(entry["hours"] or 0), 1)
        return stats

    def get_stats(self) -> dict[str, Any]:
        libraries = self.libraries()
        return {
            "libraries": libraries,
            "total_sounds": sum(s["sounds"] for s in libraries),
            "sounds_with_embeddings": sum(s["embedded"] for s in libraries),
            "sounds_with_local_files": sum(s["local_files"] for s in libraries),
        }

    def get_categories(self, library: Optional[str] = None, limit: int = 50) -> list[dict[str, Any]]:
        with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                "SELECT library, category, COUNT(*) AS sounds FROM sounds "
                + ("WHERE library = %s " if library else "")
                + "GROUP BY library, category ORDER BY sounds DESC LIMIT %s",
                (library, limit) if library else (limit,))
            return [dict(row) for row in cursor.fetchall()]

    # ── Search ───────────────────────────────────────────────────────────────

    def vector_search(self, query_embedding: np.ndarray, limit: int = 10, threshold: float = 0.0,
                      libraries: Optional[list[str]] = None, local_only: bool = False) -> list[dict[str, Any]]:
        """Nearest descriptions by cosine similarity, optionally within given libraries."""
        column = self.embedding_column
        embedding = query_embedding.tolist()
        conditions = [f"{column} IS NOT NULL", f"1 - ({column} <=> %s::vector) >= %s"]
        params: list[Any] = [embedding, threshold]
        if libraries:
            conditions.append("library = ANY(%s)")
            params.append(libraries)
        if local_only:
            conditions.append("file_exists")
        params += [embedding, limit]          # ORDER BY, LIMIT
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    f"SELECT {RESULT_COLUMNS}, 1 - ({column} <=> %s::vector) AS similarity "
                    f"FROM sounds WHERE {' AND '.join(conditions)} "
                    f"ORDER BY {column} <=> %s::vector LIMIT %s",
                    [embedding] + params)
                results = []
                for row in cursor.fetchall():
                    result = dict(row)
                    result["similarity"] = round(float(result["similarity"]), 4)
                    results.append(result)
                return results
        except Exception:
            self.conn.rollback()
            raise

    def window_search(self, query_embedding: np.ndarray, limit: int = 40,
                      libraries: Optional[list[str]] = None, category: Optional[str] = None,
                      exclude: Optional[list[int]] = None) -> list[dict[str, Any]]:
        """Nearest 10 s audio windows (CLAP) with their sound's metadata."""
        embedding = query_embedding.tolist()
        conditions, params = ["TRUE"], []
        if libraries:
            conditions.append("s.library = ANY(%s)")
            params.append(libraries)
        if category:
            conditions.append("s.category ILIKE %s")
            params.append(f"%{category}%")
        if exclude:
            conditions.append("NOT (s.id = ANY(%s))")
            params.append(exclude)
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT s.id, s.library, s.external_id, s.description, s.category, s.collection, "
                    "       s.duration_seconds, s.file_path, w.offset_seconds, w.length_seconds, "
                    "       1 - (w.audio_embedding <=> %s::vector) AS similarity "
                    "FROM sound_windows w JOIN sounds s ON s.id = w.sound_id "
                    f"WHERE {' AND '.join(conditions)} "
                    "ORDER BY w.audio_embedding <=> %s::vector LIMIT %s",
                    [embedding] + params + [embedding, limit])
                results = []
                for row in cursor.fetchall():
                    result = dict(row)
                    result["similarity"] = round(float(result["similarity"]), 4)
                    results.append(result)
                return results
        except Exception:
            self.conn.rollback()
            raise

    def audio_windows(self) -> dict[str, int]:
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*), COUNT(DISTINCT sound_id) FROM sound_windows")
            windows, sounds = cursor.fetchone()
        return {"windows": int(windows), "sounds": int(sounds)}

    # ── Single sounds ────────────────────────────────────────────────────────

    def get_sound_by_id(self, sound_id: int) -> Optional[dict[str, Any]]:
        with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                f"SELECT {RESULT_COLUMNS}, file_path, media_url, extra FROM sounds WHERE id = %s",
                (sound_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def set_file(self, sound_id: int, path: str) -> None:
        db.mark_file(self.conn, sound_id, path)

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
