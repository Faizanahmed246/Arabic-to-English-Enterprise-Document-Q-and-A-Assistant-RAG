import logging
import os
import secrets
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.concurrency import run_in_threadpool
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, StringConstraints

from .documents import MAX_BYTES
from .embeddings import E5
from .index import PolicyIndex
from .service import Claude, GenerationError, RagService, elapsed_ms

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class SearchRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)]
    top_k: int = Field(default=5, ge=1, le=10)


class AnswerRequest(SearchRequest):
    language: Literal["auto", "ar", "en"] = "auto"


@asynccontextmanager
async def lifespan(app: FastAPI):
    api_key = os.environ.get("RAG_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set RAG_API_KEY before starting the service.")
    data_dir = Path(os.environ.get("RAG_DATA_DIR", "data"))
    cache_dir = Path(os.environ.get("RAG_MODEL_CACHE", str(data_dir / "models")))
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    model = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
    app.state.api_key = api_key.encode("utf-8")
    app.state.service = RagService(
        PolicyIndex(data_dir, E5(cache_dir)),
        Claude(anthropic_key, model) if anthropic_key else None,
    )
    qa_logger = logging.getLogger("rag.service")
    qa_logger.setLevel(logging.INFO)
    if not qa_logger.handlers:
        qa_logger.addHandler(logging.StreamHandler())
    qa_logger.propagate = False
    try:
        yield
    finally:
        service = app.state.service
        service.index.close()
        if service.claude:
            service.claude.client.close()
        del app.state.service


def authorize(request: Request, key: Annotated[str | None, Security(api_key_header)]) -> None:
    if not key or not secrets.compare_digest(key.encode("utf-8"), request.app.state.api_key):
        raise HTTPException(401, "Invalid or missing X-API-Key.")


app = FastAPI(title="Arabic–English Policy Q&A", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.put(
    "/v1/documents/{source}",
    dependencies=[Depends(authorize)],
    openapi_extra={"requestBody": {"required": True, "content": {
        "application/octet-stream": {"schema": {"type": "string", "format": "binary"}},
    }}},
)
async def upload_document(source: str, request: Request) -> dict:
    """Index a policy file. Uploading new contents under the same filename replaces its active version."""
    data = bytearray()
    async for block in request.stream():
        if len(data) + len(block) > MAX_BYTES:
            raise HTTPException(413, "Maximum upload size is 10 MiB.")
        data.extend(block)
    try:
        return await run_in_threadpool(request.app.state.service.index.ingest, source, bytes(data))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/v1/search", dependencies=[Depends(authorize)])
def search(query: SearchRequest, request: Request) -> dict:
    """Retrieve ranked policy passages without calling Claude."""
    start = perf_counter()
    passages = request.app.state.service.index.search(query.question, query.top_k)
    return {"passages": [asdict(passage) for passage in passages], "retrieval_ms": elapsed_ms(start)}


@app.post("/v1/ask", dependencies=[Depends(authorize)])
def ask(query: AnswerRequest, request: Request) -> dict:
    """Answer in Arabic or English and return verified references to retrieved excerpts."""
    service = request.app.state.service
    if service.claude is None:
        raise HTTPException(503, "Configure ANTHROPIC_API_KEY to enable answer generation.")
    try:
        return service.answer(query.question, query.language, query.top_k)
    except GenerationError as exc:
        raise HTTPException(502, str(exc)) from exc
