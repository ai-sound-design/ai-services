#!/usr/bin/env python3
"""
Compute X-CLIP text embeddings for sound descriptions.

The search compares a video (or prompt) embedding against these text embeddings
in X-CLIP's shared space, so this is the step that makes a library searchable.
Only the text encoder runs; no audio is read.

Processes rows without an embedding, so it is safe to interrupt and rerun.
Run it after a source has added rows (index_folder.py runs it for you unless
told otherwise).

Usage:
    python embed.py                     # every library, base model (512 dimensions)
    python embed.py --library mine      # one library
    python embed.py --model large       # large model, 768 dimensions
    python embed.py --rebuild           # recompute embeddings that already exist
"""
from __future__ import annotations

import argparse
import logging
import time

import torch
from psycopg2.extras import execute_batch
from transformers import AutoModel, AutoTokenizer

from common import db

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
log = logging.getLogger("embed")


class TextEncoder:
    def __init__(self, model_name: str, device: str):
        log.info("loading %s on %s", model_name, device)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

    @torch.no_grad()
    def encode(self, texts: list[str]) -> list[list[float]]:
        tokens = self.tokenizer(texts, padding=True, truncation=True,
                                max_length=77, return_tensors="pt").to(self.device)
        features = self.model.get_text_features(**tokens)
        # Cosine distance in pgvector expects unit-length vectors.
        features = features / features.norm(dim=-1, keepdim=True)
        return features.float().cpu().tolist()


def embed_missing(connection, model: str = "base", library: str | None = None,
                  batch_size: int = 64, rebuild: bool = False) -> int:
    """Embed every row lacking a vector (or all rows with rebuild). Returns the count."""
    spec = db.EMBEDDING_COLUMNS[model]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log.warning("no GPU visible, embedding on the CPU (slower, still fine for a few thousand sounds)")

    conditions, params = [], []
    if not rebuild:
        conditions.append(f"{spec['column']} IS NULL")
    if library:
        conditions.append("library = %s")
        params.append(library)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    with connection.cursor() as cursor:
        cursor.execute(f"SELECT id, description FROM sounds {where} ORDER BY id", params)
        rows = cursor.fetchall()
    if not rows:
        log.info("every row already has a %s embedding", model)
        return 0

    log.info("embedding %d descriptions with the %s model", len(rows), model)
    encoder = TextEncoder(spec["model"], device)
    started = time.time()
    processed = 0
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset + batch_size]
        vectors = encoder.encode([description for _, description in batch])
        with connection.cursor() as cursor:
            execute_batch(cursor,
                          f"UPDATE sounds SET {spec['column']} = %s::vector WHERE id = %s",
                          [(str(vector), row_id) for (row_id, _), vector in zip(batch, vectors)],
                          page_size=200)
        connection.commit()
        processed += len(batch)
        if offset % (batch_size * 20) == 0 or processed == len(rows):
            rate = processed / max(time.time() - started, 1e-6)
            log.info("%d / %d embedded (%.0f per second)", processed, len(rows), rate)

    count = db.build_vector_index(connection, spec)
    log.info("done: %d vectors indexed in %.1f minutes", count, (time.time() - started) / 60)
    return processed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=tuple(db.EMBEDDING_COLUMNS), default="base")
    parser.add_argument("--library", help="only this library")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rebuild", action="store_true", help="recompute embeddings that already exist")
    args = parser.parse_args()

    connection = db.connect()
    db.ensure_schema(connection)
    embed_missing(connection, args.model, args.library, args.batch_size, args.rebuild)
    connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
