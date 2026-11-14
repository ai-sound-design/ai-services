#!/usr/bin/env python3
"""Exports the BBC part of the sound index as the files the prebuilt index consists of:
the metadata and text embeddings (sounds), the segment vectors (sound_segments) and the
fine windows (sound_fine), each as a gzip-compressed binary COPY stream, plus a manifest
with row counts, checksums, column lists and the parameters the index was built with.
load_index.py reads them back on any installation; the files hold no audio.

    python export_index.py --out /path/to/export     # DATABASE_URL as the indexer uses it

The local-file columns of sounds (file_path, file_exists) are exported empty: whoever
loads the index has no files yet, and the search fetches a recording when it is used.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import psycopg2

REPO_ID = "ai-sound-design/bbc-sound-index"
LIBRARY = "bbc"
# The forms of the fine windows' vectors (common/audio.py pack_fine), as the manifest describes them.
# The int8 numbers were measured on 2026-11-14 against the float16 windows of the same index.
FINE_FORMS = {
    "int8": {"form": "int8",
             "layout": "int8 x 512 per 2 s window, row-major, in sound_fine.vectors; one float16 scale per window "
                       "(max |v| / 127) in sound_fine.scales; a vector is int8 * scale",
             "precision": "cosine error to the float16 vectors: mean 2.7e-5, p99 4.3e-5, max 5.7e-5 (6,577 windows). "
                          "Search by sound, refine stage, 300 queries over the top 15 candidates: the top result the "
                          "same in 100 % (1 s queries) and 99.7 % (3 s), scores within 0.0015; 60 query clips through "
                          "the API without audio on disk: the top result the same in 59, the top 3 in all, scores "
                          "within 0.0007"},
    "fp16": {"form": "fp16",
             "layout": "float16 x 512 per 2 s window, row-major, in sound_fine.vectors (sound_fine.scales NULL)",
             "precision": "the vectors as CLAP produced them, rounded to float16: the reference"},
}
# Table -> how its rows are selected; columns come from the database itself
TABLES = {
    "sounds": {"from": "sounds t", "where": f"t.library = '{LIBRARY}'", "order": "t.id",
               "skip": ("created_at", "updated_at"),
               "override": {"file_path": "NULL::text", "file_exists": "false"}},
    "sound_segments": {"from": f"sound_segments t JOIN sounds s ON s.id = t.sound_id AND s.library = '{LIBRARY}'",
                       "where": "", "order": "t.id", "skip": ("id",), "override": {}},
    "sound_fine": {"from": f"sound_fine t JOIN sounds s ON s.id = t.sound_id AND s.library = '{LIBRARY}'",
                   "where": "", "order": "t.sound_id", "skip": (), "override": {}},
}


class Counting:
    """A file that counts what goes through it, for the progress line."""

    def __init__(self, handle, label: str) -> None:
        self.handle, self.label, self.raw, self.last = handle, label, 0, time.time()

    def write(self, data: bytes) -> int:
        self.raw += len(data)
        if time.time() - self.last > 2:
            self.last = time.time()
            print(f"\r   {self.label}: {self.raw / 1e9:.2f} GB read", end="", flush=True)
        return self.handle.write(data)


def columns(cursor, table: str) -> list[str]:
    cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_schema = 'public' AND table_name = %s "
                   "ORDER BY ordinal_position", (table,))
    return [row[0] for row in cursor.fetchall()]


def parameters() -> dict:
    """The constants of common/audio.py the index depends on, read from the file itself."""
    text = (Path(__file__).resolve().parent.parent / "common/audio.py").read_text(encoding="utf-8")
    out = {}
    for name in ("SAMPLE_RATE", "FINE_SECONDS", "FINE_HOP_SECONDS", "SEGMENT_THRESHOLD", "SEGMENT_MAX_SECONDS", "QUIET_DB", "EMBEDDING_DIM"):
        match = re.search(rf"^{name}\s*=\s*(?:float\(os\.getenv\([^,]+,\s*\"([^\"]+)\"\)\)|([0-9.]+))", text, re.M)
        if match:
            out[name.lower()] = float(match.group(1) or match.group(2))
    out["clap_model"] = os.getenv("CLAP_MODEL", "laion/clap-htsat-fused")
    return out


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True, help="folder for the export (created)")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"), help="defaults to DATABASE_URL")
    parser.add_argument("--variant", action="append", default=[], metavar="DIR",
                        help="another export whose differing files join the manifest under 'variants', named by the "
                             "form in their file name (sound_fine.fp16.copy.gz: fp16); copy those files beside this export's")
    args = parser.parse_args()
    if not args.database_url:
        sys.exit("DATABASE_URL is not set")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"name": "bbc-sound-index", "library": LIBRARY, "repo": REPO_ID,
                "built": time.strftime("%Y-%m-%d"), "format": "postgresql-copy-binary+gzip",
                "description": "Prebuilt search index of the BBC Sound Effects archive for AI Sound Design: metadata, "
                               "text embeddings and the audio index (CLAP segment vectors and fine windows). No audio.",
                "source": "BBC Sound Effects (https://sound-effects.bbcrewind.co.uk)",
                "licence": "Derived from the BBC Sound Effects archive; usable under the BBC's terms only: personal, "
                           "educational and research use (https://sound-effects.bbcrewind.co.uk/licensing). It contains no audio.",
                "text_embeddings": {"text_embedding": "X-CLIP base (microsoft/xclip-base-patch32), 512",
                                    "text_embedding_large": "X-CLIP large, 768"},
                "audio_index": parameters(),
                "tables": {}}
    with psycopg2.connect(args.database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT version()")
            manifest["postgresql"] = cursor.fetchone()[0].split(",")[0]
            cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            row = cursor.fetchone()
            manifest["pgvector"] = row[0] if row else None
            for table, spec in TABLES.items():
                names = [c for c in columns(cursor, table) if c not in spec["skip"]]
                select = ", ".join(f"{spec['override'][c]} AS {c}" if c in spec["override"] else f"t.{c}" for c in names)
                where = f" WHERE {spec['where']}" if spec["where"] else ""
                query = f"SELECT {select} FROM {spec['from']}{where} ORDER BY {spec['order']}"
                cursor.execute(f"SELECT COUNT(*) FROM {spec['from']}{where}")
                rows = cursor.fetchone()[0]
                form = ""
                if table == "sound_fine":
                    # Named by the form of the windows' vectors, so both forms can sit in one folder
                    cursor.execute("SELECT COUNT(*) FILTER (WHERE scales IS NULL) FROM sound_fine")
                    form = ".fp16" if cursor.fetchone()[0] else ".int8"
                    manifest["fine_vectors"] = FINE_FORMS[form[1:]]
                path = out / f"{table}{form}.copy.gz"
                print(f"== {table}: {rows:,} rows -> {path.name}", flush=True)
                started = time.time()
                with gzip.open(path, "wb", compresslevel=6) as compressed:
                    counting = Counting(compressed, table)
                    cursor.copy_expert(f"COPY ({query}) TO STDOUT (FORMAT binary)", counting)
                print(f"\r   {table}: {counting.raw / 1e9:.2f} GB raw, {path.stat().st_size / 1e9:.2f} GB gzip, "
                      f"{time.time() - started:.0f} s", flush=True)
                manifest["tables"][table] = {"file": path.name, "rows": rows, "columns": names,
                                             "bytes": path.stat().st_size, "raw_bytes": counting.raw, "sha256": sha256(path)}
    for other in args.variant:
        # Another form of the same index: the files whose name carries a form (sound_fine.fp16.copy.gz)
        # join under "variants"; the rest are the shared files of this export
        theirs = json.loads((Path(other) / "manifest.json").read_text(encoding="utf-8"))
        for name, table in theirs["tables"].items():
            ours = manifest["tables"][name]
            if table["rows"] != ours["rows"]:
                sys.exit(f"{other}: {name} has {table['rows']:,} rows, this export {ours['rows']:,}; a variant holds the same sounds")
            parts = Path(table["file"]).name.split(".")
            if len(parts) >= 4 and parts[1] != Path(ours["file"]).name.split(".")[1]:
                variant = manifest.setdefault("variants", {}).setdefault(parts[1], {"tables": {}})
                variant["tables"][name] = table
                if name == "sound_fine":
                    variant["fine_vectors"] = theirs.get("fine_vectors") if isinstance(theirs.get("fine_vectors"), dict) else FINE_FORMS.get(parts[1])
                print(f"== variant {parts[1]}: {table['file']} ({table['bytes'] / 1e9:.2f} GB); copy it beside the files here", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"== manifest.json written; {sum(t['bytes'] for t in manifest['tables'].values()) / 1e9:.2f} GB in all", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
