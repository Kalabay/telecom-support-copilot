"""Добавляет (upsert) KB-документы в СУЩЕСТВУЮЩУЮ коллекцию, НЕ пересоздавая её."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from sentence_transformers import SentenceTransformer  # noqa: E402

import chromadb  # noqa: E402
from chromadb.config import Settings as ChromaSettings  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.pipeline.kb_loader import load_kb  # noqa: E402

EMBED = {
    "frida": ("ai-forever/FRIDA", "telecom_kb_frida", "search_document: "),
    "bgem3": ("BAAI/bge-m3", "telecom_kb", ""),
}
MODEL_NAME, COLLECTION_NAME, DOC_PREFIX = EMBED[settings.embed_backend]
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", type=Path, required=True, help="папка с .md (рекурсивно)")
    args = ap.parse_args()

    kb_dir = args.kb if args.kb.is_absolute() else PROJECT_ROOT / args.kb
    chunks = load_kb(kb_dir, recursive=True)
    docs = {c.doc_id for c in chunks}
    companies = sorted({c.company or "(нет)" for c in chunks})
    print(f"loaded {len(chunks)} chunks / {len(docs)} docs from {kb_dir}")
    print(f"companies: {companies}")

    print(f"loading embedder {MODEL_NAME} (backend={settings.embed_backend})…")
    t0 = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    texts = [DOC_PREFIX + c.to_embedding_text() for c in chunks]
    print(f"encoding {len(texts)} chunks…")
    t0 = time.perf_counter()
    emb = model.encode(
        texts, batch_size=8, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )
    print(f"  encoded in {time.perf_counter() - t0:.1f}s, dim={emb.shape[1]}")

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR), settings=ChromaSettings(anonymized_telemetry=False)
    )
    col = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    before = col.count()

    metadatas = [
        {
            "doc_id": c.doc_id, "title": c.title, "section": c.section,
            "intents": ",".join(c.intents), "emotion_context": c.emotion_context,
            "company": c.company,
        }
        for c in chunks
    ]
    col.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=emb.tolist(),
        documents=[c.text for c in chunks],
        metadatas=metadatas,
    )
    print(f"collection '{COLLECTION_NAME}': {before} -> {col.count()} chunks "
          f"(+{col.count() - before})")


if __name__ == "__main__":
    main()
