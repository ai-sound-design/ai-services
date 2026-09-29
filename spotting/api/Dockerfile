# Spotting API: samples frames from a video and asks a vision-language model
# for the sound events. The model itself runs elsewhere (Ollama or any
# OpenAI-compatible server) and is reached over VLM_URL, so this image is small
# and needs no GPU.
FROM python:3.12-slim

# fonts-dejavu-core: the font ffmpeg uses to burn the timestamp into each frame.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

COPY spotting/api/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY spotting/api/main.py /app/main.py
WORKDIR /app
EXPOSE 8003

HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8003/health', timeout=12).status==200 else 1)"

CMD ["python", "main.py"]
