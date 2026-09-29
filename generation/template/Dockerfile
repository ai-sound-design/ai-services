# Template image for a generation service of your own. Swap the base image for
# a CUDA-enabled one (for example pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime)
# when your model needs a GPU, and add its dependencies to requirements.txt.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

COPY generation/template/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY generation/template/main.py /app/main.py
WORKDIR /app

ENV API_PORT=8010
EXPOSE 8010

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8010/health', timeout=4).status==200 else 1)"

CMD ["python", "main.py"]
