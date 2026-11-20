#!/bin/sh
# Railway / container entrypoint.
# 1. Build (or reuse, if unchanged) the role indexes. Container disks are
#    ephemeral, so without a volume this re-embeds on every deploy.
# 2. Start the API on Railway's $PORT. The server starts even if ingestion
#    failed, so /health and /ingest/status can report the problem.
set -u

python ingest.py || echo "WARNING: ingestion failed or was incomplete; see logs above. /chat may return 503."

# Trust X-Forwarded-* headers from Railway's proxy (read by uvicorn from env).
export FORWARDED_ALLOW_IPS="${FORWARDED_ALLOW_IPS:-*}"

exec python -m uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}" --proxy-headers
