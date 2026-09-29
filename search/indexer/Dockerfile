#
# Indexer: one-shot tooling, not a long-running service.
#
#   index_folder.py         a folder of your own sounds -> library rows + embeddings
#   embed.py                X-CLIP text embeddings for rows that lack them
#   embed_audio.py          CLAP audio embeddings (10 s windows) for files on disk
#   sources/bbc/harvest.py  BBC Sound Effects metadata from the public API
#   sources/bbc/download.py the BBC audio, for offline use
#
# Shares the PyTorch base image with the other services, so the large layers
# are already present after one of them has been built.

ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libpq5 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install \
        "transformers>=4.40,<5" \
        "psycopg2-binary>=2.9" \
        "sentencepiece>=0.1.99" \
        "mutagen>=1.47"

COPY search/common /app/common
COPY search/schema.sql /app/common/schema.sql
COPY search/indexer/*.py /app/
COPY search/sources /app/sources

ENV HF_HOME=/cache/huggingface \
    AUDIO_DIR=/data/audio \
    LIBRARIES_DIR=/data/libraries \
    PYTHONPATH=/app

WORKDIR /app

# Nothing runs automatically; invoke a script explicitly, for example
#   docker compose run --rm indexer python index_folder.py mine
CMD ["python", "-c", "print('Run index_folder.py, embed.py or a script under sources/ explicitly.')"]
