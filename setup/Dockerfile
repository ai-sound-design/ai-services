#
# Setup page: what is installed, what is still being fetched or indexed, with
# progress; the long steps started, paused and resumed from the browser.
#
# The same image as the indexer plus a small web server: the steps it runs are
# the indexer's own scripts, as subprocesses, so the page needs nothing the
# command line does not have.

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

RUN pip install "fastapi>=0.110" "uvicorn[standard]>=0.27" "httpx>=0.27"

COPY search/common /app/common
COPY search/schema.sql /app/common/schema.sql
COPY search/indexer/*.py /app/
COPY search/sources /app/sources
COPY setup/main.py setup/page.html /app/

ENV HF_HOME=/cache/huggingface \
    AUDIO_DIR=/data/audio \
    LIBRARIES_DIR=/data/libraries \
    PYTHONPATH=/app

WORKDIR /app
EXPOSE 8020

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8020/status', timeout=4).status==200 else 1)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8020", "--log-level", "warning"]
