-- Sound index shared by the search API and the indexer.
--
-- PostgreSQL runs this on the first start with an empty data volume. The
-- indexer and the API run it again at start-up; every statement is idempotent,
-- so an existing database is brought up to date without manual steps.
--
-- One row per sound, from any library: the BBC archive, a folder of your own
-- recordings, anything a source script can describe with a sentence. The
-- search compares a video (or a prompt) against the description's X-CLIP text
-- embedding. A second table, sound_windows, holds CLAP audio embeddings of
-- 10 s windows of the files that are on disk, for search by sound
-- (/search/by_audio); embed_audio.py fills it.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS sounds (
    id                SERIAL PRIMARY KEY,
    library           VARCHAR(64)  NOT NULL,     -- 'bbc', or the name of a folder library
    external_id       VARCHAR(512) NOT NULL,     -- id inside the library: BBC id, relative path
    description       TEXT         NOT NULL,     -- what is embedded and shown in the plugin
    category          VARCHAR(255),
    collection        VARCHAR(255),              -- BBC: category group; folder: first subfolder
    duration_seconds  DOUBLE PRECISION,
    file_path         TEXT,                      -- path inside the container; NULL until fetched
    file_exists       BOOLEAN NOT NULL DEFAULT FALSE,
    media_url         TEXT,                      -- where the audio can be fetched on demand
    extra             JSONB NOT NULL DEFAULT '{}'::jsonb,   -- anything source-specific
    text_embedding        vector(512),           -- X-CLIP base
    text_embedding_large  vector(768),           -- X-CLIP large
    created_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (library, external_id)
);

CREATE INDEX IF NOT EXISTS idx_sounds_library  ON sounds (library);
CREATE INDEX IF NOT EXISTS idx_sounds_category ON sounds (category);

-- The IVFFlat vector indexes (idx_text_emb, idx_text_emb_large) are created by
-- embed.py once embeddings exist; centroids computed on an empty table would be
-- meaningless.

-- One row per 10 s window of a local audio file (see common/audio.py): the
-- search by sound answers "file X from second 140", which is what a short
-- event inside a long recording needs.
CREATE TABLE IF NOT EXISTS sound_windows (
    id               BIGSERIAL PRIMARY KEY,
    sound_id         INTEGER NOT NULL REFERENCES sounds(id) ON DELETE CASCADE,
    offset_seconds   DOUBLE PRECISION NOT NULL,
    length_seconds   DOUBLE PRECISION NOT NULL,
    audio_embedding  vector(512) NOT NULL,       -- CLAP
    UNIQUE (sound_id, offset_seconds)
);

CREATE INDEX IF NOT EXISTS idx_sound_windows_sound ON sound_windows (sound_id);

-- HNSW needs no training data, so unlike IVFFlat it can exist from the start
-- and grows with every insert: the index is searchable while embed_audio.py
-- is still running.
CREATE INDEX IF NOT EXISTS idx_audio_emb ON sound_windows
    USING hnsw (audio_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

DROP VIEW IF EXISTS library_stats;
CREATE VIEW library_stats AS
    SELECT library,
           COUNT(*)                                             AS sounds,
           COUNT(*) FILTER (WHERE text_embedding IS NOT NULL)   AS embedded,
           COUNT(*) FILTER (WHERE file_exists)                  AS local_files,
           COUNT(*) FILTER (WHERE EXISTS (SELECT 1 FROM sound_windows w WHERE w.sound_id = sounds.id))
                                                                AS audio_indexed,
           SUM(duration_seconds) / 3600.0                       AS hours
    FROM sounds
    GROUP BY library
    ORDER BY library;
