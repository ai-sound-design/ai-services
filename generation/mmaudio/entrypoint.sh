#!/bin/sh
# Prepare the model, then hand over to the server. A failed prefetch must stop
# the container rather than start a service that fails on its first request.
set -e

# Every model of the stack lives under /models (the MODELS_STORE volume or folder).
# MMAudio resolves its checkpoint paths relative to the working directory, so
# /app/api/weights and /app/api/ext_weights are links into the store. Weights baked
# into the image (BAKE_WEIGHTS=1) move into an empty store once.
for name in weights ext_weights; do
    store="/models/mmaudio/$name"
    local_dir="/app/api/$name"
    mkdir -p "$store"
    if [ -d "$local_dir" ] && [ ! -L "$local_dir" ]; then
        if [ -z "$(ls -A "$store")" ] && [ -n "$(ls -A "$local_dir")" ]; then
            mv "$local_dir"/* "$store"/
        fi
        rm -rf "$local_dir"
    fi
    [ -L "$local_dir" ] || ln -s "$store" "$local_dir"
done
mkdir -p "${HF_HOME:-/models/huggingface}" "${TORCH_HOME:-/models/torch}"

if [ "${MMAUDIO_PREFETCH:-1}" = "1" ]; then
    python /app/api/prefetch.py
fi

exec "$@"
