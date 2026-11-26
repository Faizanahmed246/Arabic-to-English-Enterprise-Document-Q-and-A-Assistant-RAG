FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    RAG_DATA_DIR=/var/data/rag \
    RAG_MODEL_CACHE=/opt/models \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    HF_HUB_DISABLE_XET=1 \
    TOKIO_WORKER_THREADS=2

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends gosu libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home rag

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY rag ./rag
RUN install -d -o rag -g rag "$RAG_MODEL_CACHE" \
    && gosu rag python -m rag.embeddings "$RAG_MODEL_CACHE"
COPY docker-start.sh .

EXPOSE 10000
ENTRYPOINT ["/bin/sh", "/app/docker-start.sh"]
