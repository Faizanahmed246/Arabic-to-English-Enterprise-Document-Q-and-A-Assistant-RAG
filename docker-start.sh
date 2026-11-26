#!/bin/sh
set -eu

install -d -m 0750 -o rag -g rag "$RAG_DATA_DIR"
exec gosu rag python -m uvicorn rag.api:app --host 0.0.0.0 --port "${PORT:-10000}" --workers 1 --no-access-log
