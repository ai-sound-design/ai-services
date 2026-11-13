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
        self._conn = None
        db.ensure_schema(self.conn)
        logger.info("Database connection established (using %s)", self.embedding_column)

    @property
    def conn(self):
        """The connection; opened again once the database went away (PostgreSQL restarted
        under the Windows or Mac backend, the laptop slept) instead of failing for good."""
        if self._conn is None or self._conn.closed:
            if self._conn is not None:
                logger.warning("database connection lost; connecting again")
            self._conn = db.connect(self.database_url)
            # Reads only (plus the occasional mark_file, which commits itself). With
            # autocommit no transaction stays open between requests, so an indexer
            # running CREATE OR REPLACE VIEW is not blocked by this connection's
            # last SELECT, which would otherwise hold its lock indefinitely.
            self._conn.autocommit = True
            with self._conn.cursor() as cursor:
                cursor.execute("SET hnsw.ef_search = 120")     # wider candidate list for the window search
        return self._conn

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
                      exclude: Optional[list[int]] = None, only: Optional[list[int]] = None) -> list[dict[str, Any]]:
        """Nearest 10 s audio windows (CLAP) with their sound's metadata. `only`: look
        inside these sounds alone (a sketch search names its candidates by description first)."""
        return self._audio_search("sound_windows", "vector", "w.offset_seconds, w.length_seconds",
                                  query_embedding, limit, libraries, category, exclude, only)

    def segment_search(self, query_embedding: np.ndarray, limit: int = 40,
                       libraries: Optional[list[str]] = None, category: Optional[str] = None,
                       exclude: Optional[list[int]] = None, only: Optional[list[int]] = None) -> list[dict[str, Any]]:
        """Nearest segments of the adaptive index, in the same form as window_search:
        `offset_seconds`/`length_seconds` are the segment's start and length."""
        return self._audio_search("sound_segments", "halfvec",
                                  "w.start_seconds AS offset_seconds, w.end_seconds - w.start_seconds AS length_seconds",
                                  query_embedding, limit, libraries, category, exclude, only)

    def _audio_search(self, table: str, vector_type: str, span: str, query_embedding: np.ndarray, limit: int,
                      libraries: Optional[list[str]], category: Optional[str],
                      exclude: Optional[list[int]], only: Optional[list[int]]) -> list[dict[str, Any]]:
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
        if only:
            conditions.append("s.id = ANY(%s)")
            params.append(only)
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT s.id, s.library, s.external_id, s.description, s.category, s.collection, "
                    f"       s.duration_seconds, s.file_path, {span}, "
                    f"       1 - (w.audio_embedding <=> %s::{vector_type}) AS similarity "
                    f"FROM {table} w JOIN sounds s ON s.id = w.sound_id "
                    f"WHERE {' AND '.join(conditions)} "
                    f"ORDER BY w.audio_embedding <=> %s::{vector_type} LIMIT %s",
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

    def fine_windows(self, sound_id: int) -> Optional[dict[str, Any]]:
        """The 2 s windows of one sound from the adaptive index: window_seconds, offsets,
        levels (dBFS) and vectors (float32, one row per window); None if not indexed."""
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT window_seconds, offsets, levels, vectors FROM sound_fine WHERE sound_id = %s",
                           (sound_id,))
            row = cursor.fetchone()
        if not row:
            return None
        offsets = np.asarray(row[1], dtype=np.float32)
        vectors = np.frombuffer(bytes(row[3]), dtype=np.float16).astype(np.float32).reshape(len(offsets), -1)
        return {"window_seconds": float(row[0]), "offsets": offsets,
                "levels": np.asarray(row[2], dtype=np.float32), "vectors": vectors}

    def segments_ready(self) -> bool:
        """True once every sound with 10 s windows also has its segments."""
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT EXISTS (SELECT 1 FROM sound_segments) AND NOT EXISTS ("
                           "  SELECT 1 FROM sounds s WHERE s.file_exists"
                           "  AND EXISTS (SELECT 1 FROM sound_windows w WHERE w.sound_id = s.id)"
                           "  AND NOT EXISTS (SELECT 1 FROM sound_fine f WHERE f.sound_id = s.id))")
            return bool(cursor.fetchone()[0])

    def audio_segments(self) -> dict[str, int]:
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT (SELECT COUNT(*) FROM sound_segments), (SELECT COUNT(*) FROM sound_fine)")
            segments, sounds = cursor.fetchone()
        return {"segments": int(segments), "sounds": int(sounds)}

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
