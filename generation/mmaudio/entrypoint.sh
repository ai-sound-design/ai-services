#!/bin/sh
# Prepare the model, then hand over to the server. A failed prefetch must stop
# the container rather than start a service that fails on its first request.
set -e

if [ "${MMAUDIO_PREFETCH:-1}" = "1" ]; then
    python /app/api/prefetch.py
fi

exec "$@"
