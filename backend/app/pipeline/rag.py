"""Retrieval-модуль: семантический поиск в ChromaDB через BGE-M3."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock

from sentence_transformers import SentenceTransformer

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from app.core.config import settings
from app.models.schemas import KBSource

EMBED_BACKENDS = {
    "bgem3": {"model": "BAAI/bge-m3", "collection": "telecom_kb",
              "q_prefix": "", "d_prefix": ""},
    "frida": {"model": "ai-forever/FRIDA", "collection": "telecom_kb_frida",
              "q_prefix": "search_query: ", "d_prefix": "search_document: "},
}
_EMB = EMBED_BACKENDS.get(settings.embed_backend, EMBED_BACKENDS["bgem3"])
COLLECTION_NAME = _EMB["collection"]
EMBED_MODEL_NAME = _EMB["model"]


@dataclass
class RetrievalResult:
    sources: list[KBSource]
    query_embed_ms: int
    search_ms: int
    total_ms: int


class Retriever:
    """Ленивый singleton-ретривер. Загружает модель и коллекцию при первом обращении."""

    _instance: Retriever | None = None
    _lock = Lock()

    def __new__(cls) -> Retriever:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._model: SentenceTransformer | None = None
        self._collection = None
        self._initialized = True

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._collection is not None:
            return

        logger.info(f"Loading embedder '{settings.embed_backend}': {EMBED_MODEL_NAME}")
        t0 = time.perf_counter()
        self._model = SentenceTransformer(EMBED_MODEL_NAME, trust_remote_code=True)
        logger.info(f"Embedder loaded in {time.perf_counter() - t0:.1f}s")

        chroma_dir = Path(settings.chroma_dir)
        if not chroma_dir.exists():
            raise RuntimeError(
                f"ChromaDB dir not found: {chroma_dir}. "
                f"Run `scripts/index_kb.py` first."
            )

        client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        try:
            self._collection = client.get_collection(COLLECTION_NAME)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Collection '{COLLECTION_NAME}' missing in {chroma_dir}. "
                f"Run `scripts/index_kb.py` to build it."
            ) from exc
        logger.info(f"Collection '{COLLECTION_NAME}' has {self._collection.count()} chunks")

    def search(self, query: str, k: int = 5, company: str | None = None) -> RetrievalResult:
        """Семантический поиск top-k чанков по запросу."""
        self._ensure_loaded()
        assert self._model is not None and self._collection is not None

        t_total = time.perf_counter()
        t_embed = time.perf_counter()
        emb = self._model.encode([_EMB["q_prefix"] + query],
                                  normalize_embeddings=True, convert_to_numpy=True)
        embed_ms = int((time.perf_counter() - t_embed) * 1000)

        t_search = time.perf_counter()
        query_kwargs = {
            "query_embeddings": emb.tolist(),
            "n_results": k * 4,
            "include": ["documents", "metadatas", "distances"],
        }
        if company:
            query_kwargs["where"] = {"company": company}
        res = self._collection.query(**query_kwargs)
        search_ms = int((time.perf_counter() - t_search) * 1000)

        sources: list[KBSource] = []
        seen_docs: set[str] = set()
        for doc_text, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            doc_id = meta.get("doc_id", "")
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            snippet = doc_text.strip().replace("\n", " ")
            if len(snippet) > 220:
                snippet = snippet[:217] + "…"
            sources.append(
                KBSource(
                    doc_id=doc_id,
                    title=meta.get("title", doc_id),
                    snippet=snippet,
                    score=round(max(0.0, 1.0 - float(dist) / 2.0), 3),
                )
            )
            if len(sources) >= k:
                break

        total_ms = int((time.perf_counter() - t_total) * 1000)
        return RetrievalResult(
            sources=sources,
            query_embed_ms=embed_ms,
            search_ms=search_ms,
            total_ms=total_ms,
        )

    def index_chunks(self, chunks) -> int:  # noqa: ANN001
        """Добавить (upsert) чанки KB в коллекцию — для загрузки документов через UI."""
        self._ensure_loaded()
        assert self._model is not None and self._collection is not None
        if not chunks:
            return 0
        texts = [_EMB["d_prefix"] + c.to_embedding_text() for c in chunks]
        emb = self._model.encode(texts, normalize_embeddings=True,
                                 convert_to_numpy=True, batch_size=8)
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=emb.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[{
                "doc_id": c.doc_id, "title": c.title, "section": c.section,
                "intents": ",".join(c.intents), "emotion_context": c.emotion_context,
                "company": c.company,
            } for c in chunks],
        )
        return len(chunks)

    def get_document(self, doc_id: str) -> dict:
        """Собрать полный текст KB-документа по doc_id (из его чанков-секций)."""
        self._ensure_loaded()
        assert self._collection is not None
        res = self._collection.get(
            where={"doc_id": doc_id}, include=["documents", "metadatas"]
        )
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []
        if not docs:
            return {"doc_id": doc_id, "title": doc_id, "company": "", "text": ""}
        title = (metas[0] or {}).get("title", doc_id)
        company = (metas[0] or {}).get("company", "")
        parts: list[str] = []
        for text, meta in zip(docs, metas):
            section = (meta or {}).get("section", "")
            if section and section not in ("all", "intro"):
                parts.append(f"## {section}\n{text}")
            else:
                parts.append(text)
        return {"doc_id": doc_id, "title": title, "company": company,
                "text": "\n\n".join(parts)}

    def list_documents(self) -> list[dict]:
        """Список всех документов (для браузера базы знаний в UI): doc_id/title/company."""
        self._ensure_loaded()
        assert self._collection is not None
        res = self._collection.get(include=["metadatas"])
        seen: dict[str, dict] = {}
        for m in res.get("metadatas") or []:
            did = (m or {}).get("doc_id")
            if did and did not in seen:
                seen[did] = {"doc_id": did, "title": m.get("title", did),
                             "company": m.get("company", "")}
        return sorted(seen.values(), key=lambda d: (d["company"], d["doc_id"]))


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()
