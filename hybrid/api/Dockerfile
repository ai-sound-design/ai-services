# Hybrid API: composes the spotting and generation services into one sound
# per event. No model of its own, so the image is small and needs no GPU.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

COPY hybrid/api/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY hybrid/api/main.py hybrid/api/matching.py /app/
WORKDIR /app
EXPOSE 8004

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8004/health', timeout=8).status==200 else 1)"

CMD ["python", "main.py"]
