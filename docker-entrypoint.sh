#!/bin/sh
set -eu

# The Proxy owns provider credentials and routing.  It remains private on the
# Docker network; the Flask adapter is the only public listener.
litellm --config /app/litellm_config.yml --port 4000 &
proxy_pid=$!
app_pid=""

cleanup() {
  kill "$proxy_pid" 2>/dev/null || true
  kill "$app_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

gunicorn --bind "0.0.0.0:${PORT:-8000}" --workers "${WEB_CONCURRENCY:-1}" --threads "${GUNICORN_THREADS:-4}" --timeout "${GUNICORN_TIMEOUT:-130}" api.index:app &
app_pid=$!

wait "$app_pid"
