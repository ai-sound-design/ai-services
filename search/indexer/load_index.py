#!/usr/bin/env python3
"""Loads the prebuilt BBC index into the sound index: the metadata and text embeddings of
the archive's 33,000 recordings, the segment vectors and the fine windows of the audio
index, as export_index.py wrote them. From Hugging Face (BBC_INDEX_REPO, default
ai-sound-design/bbc-sound-index) or from a folder with the files (--files). No audio is
in it: a recording is fetched from the BBC when it is previewed or imported.

    python load_index.py                  # DATABASE_URL as the indexer uses it
    python load_index.py --files /path    # the export folder instead of the download

Rows are matched by the BBC's own ids, so an index that is already there (built here,
or loaded before) is replaced in place and the ids of other libraries are untouched. The
local-file columns of the sounds keep what they are. Progress goes to the setup page
through the setup_state table (key bbc_index).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import psycopg2
from psycopg2.extras import Json

from common import db

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
log = logging.getLogger("load_index")

REPO_ID = os.getenv("BBC_INDEX_REPO", "ai-sound-design/bbc-sound-index")
REVISION = os.getenv("BBC_INDEX_REVISION") or None
DOWNLOAD_DIR = Path(os.getenv("INDEX_DOWNLOAD_DIR") or Path(os.getenv("SETUP_LOG_DIR", "/data/setup/logs")).parent / "index-download")
STAGING = "bbc_index_staging"
TABLES = ("sounds", "sound_segments", "sound_fine")
# The columns of sounds the index brings; the local-file columns stay as they are here
SOUND_COLUMNS = ("description", "category", "collection", "duration_seconds", "media_url", "extra",
                 "text_embedding", "text_embedding_large")


class Progress:
    """What the setup page shows while this runs: phase, done, total, a line of text."""

    def __init__(self) -> None:
        # A connection of its own: the loading connection is busy with a COPY for minutes
        self.connection = db.connect()
        self.connection.autocommit = True
        self.last = 0.0

    def set(self, phase: str, done: int = 0, total: int = 0, message: str = "", quiet: bool = False, **extra) -> None:
        state = {"phase": phase, "done": done, "total": total, "message": message, "at": time.time(), **extra}
        with self.connection.cursor() as cursor:
            cursor.execute("INSERT INTO setup_state (key, value) VALUES ('bbc_index', %s) "
                           "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value", (Json(state),))
        if message and not quiet:
            log.info("%s", message)

    def tick(self, phase: str, done: int, total: int, message: str = "") -> None:
        """The numbers every two seconds, for the page; the log gets the message once (set)."""
        if time.time() - self.last > 2:
            self.last = time.time()
            self.set(phase, done, total, message, quiet=True)


def loaded_before(connection) -> dict:
    with connection.cursor() as cursor:
        cursor.execute("CREATE TABLE IF NOT EXISTS setup_state (key TEXT PRIMARY KEY, value JSONB NOT NULL)")
        connection.commit()
        cursor.execute("SELECT value FROM setup_state WHERE key = 'bbc_index'")
        row = cursor.fetchone()
    return (row[0] if row else {}) or {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(progress: Progress) -> Path:
    """The manifest and the three files from Hugging Face into DOWNLOAD_DIR, resumable;
    the bytes on disk so far are the progress."""
    from huggingface_hub import hf_hub_download

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    progress.set("download", 0, 0, f"fetching the index from huggingface.co/datasets/{REPO_ID}")
    manifest_path = Path(hf_hub_download(REPO_ID, "manifest.json", repo_type="dataset", revision=REVISION, local_dir=DOWNLOAD_DIR))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    total = sum(t["bytes"] for t in manifest["tables"].values())
    stop = threading.Event()

    def watch() -> None:
        while not stop.wait(2):
            done = sum(p.stat().st_size for p in DOWNLOAD_DIR.rglob("*") if p.is_file() and p.name != "manifest.json")
            progress.set("download", min(done, total), total, f"downloading the index: {min(done, total) / 1e9:.1f} of {total / 1e9:.1f} GB")

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        for table in manifest["tables"].values():
            hf_hub_download(REPO_ID, table["file"], repo_type="dataset", revision=REVISION, local_dir=DOWNLOAD_DIR)
    finally:
        stop.set()
        watcher.join()
    return DOWNLOAD_DIR


def verify(folder: Path, manifest: dict, progress: Progress) -> None:
    for name, table in manifest["tables"].items():
        path = folder / table["file"]
        if not path.exists():
            raise SystemExit(f"{path} is missing")
        progress.set("verify", 0, 0, f"checking {path.name}")
        if sha256(path) != table["sha256"]:
            path.unlink(missing_ok=True)
            raise SystemExit(f"{path.name} is damaged (checksum differs); deleted, run this again to fetch it anew")


class Reading:
    """A gzip stream that reports how much of the compressed file has gone by."""

    def __init__(self, path: Path, progress: Progress, phase: str, done_before: int, total: int, label: str) -> None:
        self.raw = path.open("rb")
        self.stream = gzip.GzipFile(fileobj=self.raw)
        self.progress, self.phase, self.before, self.total, self.label = progress, phase, done_before, total, label

    def read(self, size: int = -1) -> bytes:
        data = self.stream.read(size)
        self.progress.tick(self.phase, self.before + self.raw.tell(), self.total, self.label)
        return data

    def readline(self, size: int = -1) -> bytes:  # psycopg2 asks for it, binary COPY never uses it
        return self.stream.readline(size)

    def close(self) -> None:
        self.stream.close()
        self.raw.close()


def load(connection, folder: Path, manifest: dict, progress: Progress) -> dict:
    """The files into staging tables, then merged into the index by the BBC ids."""
    tables = manifest["tables"]
    total = sum(t["bytes"] for t in tables.values())
    done = 0
    with connection.cursor() as cursor:
        cursor.execute(f"DROP SCHEMA IF EXISTS {STAGING} CASCADE")
        cursor.execute(f"CREATE SCHEMA {STAGING}")
        for name in TABLES:
            cursor.execute(f"CREATE TABLE {STAGING}.{name} (LIKE public.{name} INCLUDING DEFAULTS)")
        connection.commit()
        for name in TABLES:
            table = tables[name]
            columns = ", ".join(table["columns"])
            label = f"reading {name}: {table['rows']:,} rows"
            progress.set("import", done, total, label)
            reader = Reading(folder / table["file"], progress, "import", done, total, label)
            try:
                cursor.copy_expert(f"COPY {STAGING}.{name} ({columns}) FROM STDIN (FORMAT binary)", reader)
            finally:
                reader.close()
            connection.commit()
            done += table["bytes"]
            cursor.execute(f"SELECT COUNT(*) FROM {STAGING}.{name}")
            got = cursor.fetchone()[0]
            if got != table["rows"]:
                raise SystemExit(f"{name}: {got:,} rows read, {table['rows']:,} expected")

        progress.set("merge", 0, 0, "merging the sounds into the index")
        assignments = ", ".join(f"{c} = EXCLUDED.{c}" for c in SOUND_COLUMNS)
        cursor.execute(f"INSERT INTO sounds (library, external_id, {', '.join(SOUND_COLUMNS)}) "
                       f"SELECT library, external_id, {', '.join(SOUND_COLUMNS)} FROM {STAGING}.sounds "
                       f"ON CONFLICT (library, external_id) DO UPDATE SET {assignments}, updated_at = CURRENT_TIMESTAMP")
        cursor.execute(f"CREATE TEMP TABLE idmap AS SELECT s.id AS old_id, p.id AS new_id FROM {STAGING}.sounds s "
                       "JOIN sounds p ON p.library = s.library AND p.external_id = s.external_id")
        cursor.execute("CREATE INDEX ON idmap (old_id)")
        progress.set("merge", 0, 0, "replacing the audio index of those sounds")
        # Without the HNSW index the segments go in at copy speed; it is built again below
        cursor.execute("DROP INDEX IF EXISTS idx_segment_emb")
        cursor.execute("DELETE FROM sound_segments WHERE sound_id IN (SELECT new_id FROM idmap)")
        cursor.execute(f"INSERT INTO sound_segments (sound_id, start_seconds, end_seconds, audio_embedding) "
                       f"SELECT m.new_id, g.start_seconds, g.end_seconds, g.audio_embedding FROM {STAGING}.sound_segments g "
                       "JOIN idmap m ON m.old_id = g.sound_id")
        segments = cursor.rowcount
        cursor.execute("DELETE FROM sound_fine WHERE sound_id IN (SELECT new_id FROM idmap)")
        cursor.execute(f"INSERT INTO sound_fine (sound_id, window_seconds, offsets, levels, vectors) "
                       f"SELECT m.new_id, f.window_seconds, f.offsets, f.levels, f.vectors FROM {STAGING}.sound_fine f "
                       "JOIN idmap m ON m.old_id = f.sound_id")
        fine = cursor.rowcount
        cursor.execute("SELECT COUNT(*) FROM idmap")
        sounds = cursor.fetchone()[0]
        cursor.execute(f"DROP SCHEMA {STAGING} CASCADE")
        connection.commit()
        progress.set("index", 0, 0, f"building the vector index over {segments:,} segments (a few minutes)")
        # In one process: parallel workers would share the graph through /dev/shm, which is
        # 64 MB in a container; the memory for the graph itself comes from the session
        cursor.execute("SET max_parallel_maintenance_workers = 0")
        cursor.execute("SET maintenance_work_mem = '1GB'")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_segment_emb ON sound_segments "
                       "USING hnsw (audio_embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 64)")
        connection.commit()
        # The rows an earlier index left behind go back to the disk; fresh statistics for the planner
        progress.set("index", 0, 0, "tidying the tables")
        connection.autocommit = True
        cursor.execute("VACUUM ANALYZE sounds, sound_segments, sound_fine")
        connection.autocommit = False
    return {"sounds": sounds, "segments": segments, "fine": fine}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--files", help="a folder with manifest.json and the three files, instead of the download")
    parser.add_argument("--keep", action="store_true", help="keep the downloaded files afterwards")
    parser.add_argument("--force", action="store_true", help="load again even if this index is in place")
    args = parser.parse_args()
    connection = db.connect()
    connection.autocommit = False
    before = loaded_before(connection)
    progress = Progress()
    try:
        if args.files:
            folder = Path(args.files)
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        else:
            folder = download(progress)
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        key = hashlib.sha256(json.dumps(manifest["tables"], sort_keys=True).encode()).hexdigest()[:16]
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM sounds s WHERE s.library = %s AND EXISTS "
                           "(SELECT 1 FROM sound_segments g WHERE g.sound_id = s.id)", (manifest["library"],))
            indexed = cursor.fetchone()[0]
        if not args.force and before.get("loaded") == key and indexed >= manifest["tables"]["sound_fine"]["rows"] * 0.99:
            progress.set("done", 1, 1, f"the index of {manifest['built']} is in place: {indexed:,} sounds with audio index",
                         loaded=key, built=manifest["built"])
            return 0
        verify(folder, manifest, progress)
        counts = load(connection, folder, manifest, progress)
        progress.set("done", 1, 1, f"index of {manifest['built']} loaded: {counts['sounds']:,} sounds, {counts['segments']:,} segments",
                     loaded=key, built=manifest["built"], **counts)
        if not args.files and not args.keep:
            shutil.rmtree(DOWNLOAD_DIR, ignore_errors=True)
        return 0
    except Exception as error:  # noqa: BLE001
        try:
            connection.rollback()
        except Exception:  # noqa: BLE001
            pass
        progress.set("error", 0, 0, f"failed: {error}")
        raise
    finally:
        try:
            connection.close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    sys.exit(main())
