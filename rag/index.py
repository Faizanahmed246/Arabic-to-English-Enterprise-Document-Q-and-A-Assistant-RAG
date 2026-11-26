import hashlib
import logging
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import chromadb

from .documents import chunk_document
from .embeddings import INDEX_VERSION, E5

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Passage:
    chunk_id: str
    source: str
    location: str
    excerpt: str
    similarity: float


class PolicyIndex:
    def __init__(self, data_dir: Path, embedder: E5):
        data_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.manifest = data_dir / f"{INDEX_VERSION}.sqlite3"
        self.lock = Lock()
        self.client = chromadb.PersistentClient(path=str(data_dir / "chroma"))
        self.collection = self.client.get_or_create_collection(
            name=f"policies-{INDEX_VERSION}",
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine", "num_threads": 2}},
        )
        with closing(sqlite3.connect(self.manifest)) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS sources ("
                "source TEXT PRIMARY KEY, version TEXT NOT NULL, chunk_count INTEGER NOT NULL)"
            )

    def close(self) -> None:
        self.client.close()

    def ingest(self, source: str, data: bytes) -> dict:
        if not source or len(source) > 160 or re.search(r"[/\\\x00-\x1f\x7f]", source):
            raise ValueError("Use a document filename of 1–160 characters without path separators.")
        version = hashlib.sha256(source.encode("utf-8") + b"\x00" + data).hexdigest()
        with self.lock, closing(sqlite3.connect(self.manifest)) as db:
            current = db.execute(
                "SELECT version, chunk_count FROM sources WHERE source = ?", (source,)
            ).fetchone()
            if current and current[0] == version:
                return {"source": source, "chunks": current[1], "status": "already_indexed"}
            chunks = chunk_document(source, data, self.embedder.offsets)
            for start in range(0, len(chunks), 64):
                batch = chunks[start : start + 64]
                texts = [chunk.text for chunk in batch]
                self.collection.upsert(
                    ids=[f"{version}:{i}" for i in range(start, start + len(batch))],
                    documents=texts,
                    embeddings=self.embedder.encode(texts),
                    metadatas=[
                        {"source": source, "location": chunk.location, "version": version}
                        for chunk in batch
                    ],
                )
            with db:
                db.execute(
                    "INSERT INTO sources VALUES (?, ?, ?) ON CONFLICT(source) DO UPDATE SET "
                    "version = excluded.version, chunk_count = excluded.chunk_count",
                    (source, version, len(chunks)),
                )
            if current:
                try:
                    self.collection.delete(where={"version": current[0]})
                except Exception:
                    logger.warning('event=obsolete_chunks_cleanup_failed')
        return {"source": source, "chunks": len(chunks), "status": "indexed"}

    def search(self, question: str, top_k: int) -> list[Passage]:
        with self.lock, closing(sqlite3.connect(self.manifest)) as db:
            versions = [row[0] for row in db.execute("SELECT version FROM sources")]
            if not versions:
                return []
            result = self.collection.query(
                query_embeddings=self.embedder.encode([question], query=True),
                n_results=top_k,
                where={"version": {"$in": versions}},
                include=["documents", "metadatas", "distances"],
            )
        return [
            Passage(chunk_id, metadata["source"], metadata["location"], excerpt, round(1 - distance, 6))
            for chunk_id, metadata, excerpt, distance in zip(
                result["ids"][0], result["metadatas"][0], result["documents"][0], result["distances"][0],
                strict=True,
            )
        ]
