#
# Sound search API (X-CLIP + pgvector).
#
# Shares the PyTorch base image with the other services, so torch and CUDA come
# from a layer that is already present locally.

ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates ffmpeg libsndfile1 libpq5 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# torch/torchvision already match the base image; installing them again would
# only redownload the same wheels, so they are filtered out here.
COPY search/api/requirements.txt /tmp/requirements.txt
RUN grep -vE '^(torch|torchvision)==' /tmp/requirements.txt > /tmp/api-requirements.txt \
    && pip install -r /tmp/api-requirements.txt

COPY search/common /app/common
COPY search/schema.sql /app/common/schema.sql
COPY search/api/main.py /app/main.py
COPY search/api/utils /app/utils

ENV HF_HOME=/cache/huggingface \
    AUDIO_DIR=/data/audio \
    LIBRARIES_DIR=/data/libraries \
    PYTHONPATH=/app

WORKDIR /app
EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8002/health', timeout=4).status==200 else 1)"

CMD ["python", "main.py"]
