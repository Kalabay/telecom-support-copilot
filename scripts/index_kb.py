"""Индексирует data/kb/*.md в ChromaDB через BGE-M3."""

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
from rich.console import Console  # noqa: E402
from rich.progress import (  # noqa: E402
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.pipeline.kb_loader import load_kb  # noqa: E402

EMBED_BACKENDS = {
    "bgem3": {"model": "BAAI/bge-m3", "collection": "telecom_kb",
              "q_prefix": "", "d_prefix": ""},
    "frida": {"model": "ai-forever/FRIDA", "collection": "telecom_kb_frida",
              "q_prefix": "search_query: ", "d_prefix": "search_document: "},
}
_EMB = EMBED_BACKENDS[settings.embed_backend]
COLLECTION_NAME = _EMB["collection"]
EMBED_MODEL_NAME = _EMB["model"]
DOC_PREFIX = _EMB["d_prefix"]

DEFAULT_KB_DIR = PROJECT_ROOT / "data" / "kb"
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"


def build_index(kb_dir: Path, chroma_dir: Path, console: Console) -> None:
    console.rule(f"[bold cyan]Indexing KB[/]  {kb_dir}")

    recursive = any(p.is_dir() for p in kb_dir.iterdir())
    chunks = load_kb(kb_dir, recursive=recursive)
    console.log(f"Loaded [bold]{len(chunks)}[/] chunks from {kb_dir} (recursive={recursive})")

    chunks_by_doc: dict[str, int] = {}
    for c in chunks:
        chunks_by_doc[c.doc_id] = chunks_by_doc.get(c.doc_id, 0) + 1

    table = Table(title="Chunks per document", show_lines=False)
    table.add_column("doc_id", style="cyan")
    table.add_column("chunks", justify="right")
    table.add_column("intents", style="magenta")
    seen_docs: set[str] = set()
    for c in chunks:
        if c.doc_id in seen_docs:
            continue
        seen_docs.add(c.doc_id)
        table.add_row(c.doc_id, str(chunks_by_doc[c.doc_id]), ", ".join(c.intents))
    console.print(table)

    console.log(f"Loading embedding model: [bold]{EMBED_MODEL_NAME}[/] "
                f"(backend={settings.embed_backend}, collection={COLLECTION_NAME})")
    t0 = time.perf_counter()
    model = SentenceTransformer(EMBED_MODEL_NAME, trust_remote_code=True)
    console.log(f"Model loaded in {time.perf_counter() - t0:.1f}s")

    texts = [DOC_PREFIX + c.to_embedding_text() for c in chunks]

    console.log(f"Encoding {len(texts)} chunks…")
    t0 = time.perf_counter()
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("encoding", total=None)
        embeddings = model.encode(
            texts,
            batch_size=8,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        progress.update(task, completed=1, total=1)
    console.log(f"Encoded in {time.perf_counter() - t0:.1f}s, dim={embeddings.shape[1]}")

    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        console.log(f"Dropping existing collection '{COLLECTION_NAME}'")
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    metadatas = []
    for c in chunks:
        metadatas.append(
            {
                "doc_id": c.doc_id,
                "title": c.title,
                "section": c.section,
                "intents": ",".join(c.intents),
                "emotion_context": c.emotion_context,
                "company": c.company,
            }
        )

    collection.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings.tolist(),
        documents=[c.text for c in chunks],
        metadatas=metadatas,
    )

    console.log(
        f"[bold green]Indexed[/] {collection.count()} chunks "
        f"into collection '{COLLECTION_NAME}' at {chroma_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Index KB markdown into ChromaDB")
    parser.add_argument("--kb", type=Path, default=DEFAULT_KB_DIR, help="path to data/kb")
    parser.add_argument(
        "--chroma", type=Path, default=DEFAULT_CHROMA_DIR, help="path to chroma persistent store"
    )
    args = parser.parse_args()
    build_index(args.kb, args.chroma, Console())


if __name__ == "__main__":
    main()
