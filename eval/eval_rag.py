"""RAG A/B: метрики retrieval для разных эмбеддеров (+ опц. reranker)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import app  # noqa: F401,E402

import numpy as np  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from app.pipeline.kb_loader import load_kb  # noqa: E402

KB_DIR = PROJECT_ROOT / "data" / "kb"
QUERIES = PROJECT_ROOT / "eval" / "rag_queries.json"


def embed(model, texts, prefix, pooling, bs=16):
    if prefix:
        texts = [prefix + t for t in texts]
    return model.encode(texts, batch_size=bs, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prefix-query", default="")
    ap.add_argument("--prefix-doc", default="")
    ap.add_argument("--pooling", default="auto")
    ap.add_argument("--reranker", default="")
    ap.add_argument("--rerank-pool", type=int, default=20, help="сколько кандидатов до реранка")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "eval" / "results" / "rag_eval.json")
    args = ap.parse_args()

    chunks = load_kb(KB_DIR)
    doc_ids = [c.doc_id for c in chunks]
    texts = [c.to_embedding_text() for c in chunks]
    qs = json.loads(QUERIES.read_text(encoding="utf-8"))["queries"]
    print(f"KB chunks={len(chunks)}  queries={len(qs)}  model={args.model}", flush=True)

    t0 = time.perf_counter()
    model = SentenceTransformer(args.model, trust_remote_code=True)
    print(f"embedder loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    doc_emb = embed(model, texts, args.prefix_doc, args.pooling)

    reranker = None
    if args.reranker:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(args.reranker, trust_remote_code=True)
        print(f"reranker: {args.reranker}", flush=True)

    K = [1, 3, 5]
    recall = {k: 0 for k in K}
    mrr = 0.0
    lat = []
    for item in qs:
        q, rel = item["q"], set(item["relevant"])
        tq = time.perf_counter()
        qe = embed(model, [q], args.prefix_query, args.pooling)[0]
        sims = doc_emb @ qe
        order = np.argsort(-sims)
        ranked, seen = [], set()
        for idx in order:
            d = doc_ids[idx]
            if d not in seen:
                seen.add(d); ranked.append((d, idx))
            if len(ranked) >= max(args.rerank_pool, max(K)):
                break
        if reranker is not None:
            pairs = [[q, texts[idx]] for _, idx in ranked]
            scores = reranker.predict(pairs)
            reord = np.argsort(-np.asarray(scores))
            ranked = [ranked[i] for i in reord]
        lat.append((time.perf_counter() - tq) * 1000)
        ranked_docs = [d for d, _ in ranked]
        rank1 = next((i + 1 for i, d in enumerate(ranked_docs) if d in rel), 0)
        if rank1:
            mrr += 1.0 / rank1
        for k in K:
            if rel & set(ranked_docs[:k]):
                recall[k] += 1

    n = len(qs)
    res = {
        "model": args.model,
        "reranker": args.reranker or None,
        "recall@1": round(recall[1] / n, 4),
        "recall@3": round(recall[3] / n, 4),
        "recall@5": round(recall[5] / n, 4),
        "mrr": round(mrr / n, 4),
        "latency_ms_p50": round(float(np.percentile(lat, 50)), 1),
        "latency_ms_p95": round(float(np.percentile(lat, 95)), 1),
        "n_queries": n,
    }
    print("\n=== RAG eval ===")
    for k, v in res.items():
        print(f"  {k}: {v}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    runs = []
    if args.out.exists():
        try:
            runs = json.loads(args.out.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            runs = []
    runs.append(res)
    args.out.write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
