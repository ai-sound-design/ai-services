#
# MMAudio generation API — fully self-contained.
#
# Everything the service needs lives inside the image: the MMAudio source, its
# Python dependencies and the API itself. Nothing is mounted from the host except
# the model-weight caches, which are Docker named volumes. The image therefore
# behaves identically on Windows, macOS and Linux.

ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

# Pin the upstream revision so rebuilds stay reproducible.
ARG MMAUDIO_REPO=https://github.com/hkchengrex/MMAudio.git
ARG MMAUDIO_REF=974010a026c731054592d8f777218bd9d85a6c24

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ffmpeg + libsndfile: audio/video I/O. libgl1/libglib2.0-0: required by opencv,
# which MMAudio pulls in. git: to fetch MMAudio.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates ffmpeg libsndfile1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# MMAudio source and dependencies, installed into the image.
RUN git clone "${MMAUDIO_REPO}" /opt/MMAudio \
    && git -C /opt/MMAudio checkout --quiet "${MMAUDIO_REF}" \
    && pip install -e /opt/MMAudio

# torchcodec is declared by MMAudio but unused on the code path this API takes
# (MMAudio decodes with PyAV). Pinned to the build that matches torch 2.5.
RUN pip install "torchcodec==0.2.1"

COPY generation/mmaudio/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# main.py resolves the shared config as ../shared, hence the api/ subdirectory.
COPY shared /app/shared
COPY generation/mmaudio/main.py /app/api/main.py
COPY generation/mmaudio/prefetch.py /app/api/prefetch.py
COPY generation/mmaudio/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN sed -i "s/\r$//" /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

ENV MMAUDIO_PATH=/opt/MMAudio \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    MMAUDIO_MODEL=large_44k_v2 \
    MMAUDIO_PREFETCH=1

# MMAudio resolves its checkpoint paths relative to the working directory, so the
# weights land under /app/api. The compose file mounts the persistent volumes
# there; a mount under /opt/MMAudio is silently ignored.
WORKDIR /app/api

# Optional: put the weights in the image instead of fetching them on first start,
# for machines that have to run without a network. Costs roughly 10 GB of image.
#   docker compose build --build-arg BAKE_WEIGHTS=1 mmaudio-api
# Docker seeds an empty named volume from the image, so the mounts still work.
ARG BAKE_WEIGHTS=0
RUN if [ "${BAKE_WEIGHTS}" = "1" ]; then python /app/api/prefetch.py; fi

EXPOSE 8000

# The start period covers a first start that downloads about 10 GB before the
# server binds its port. Later starts answer /health within seconds.
HEALTHCHECK --interval=30s --timeout=5s --start-period=1800s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "main.py"]
