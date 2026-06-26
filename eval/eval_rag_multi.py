"""RAG A/B на мульти-тенант корпусе (kb_multi) — несатурированный бенчмарк."""

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

KB_DIR = PROJECT_ROOT / "data" / "kb_multi"
COMPANIES = ["mts", "beeline", "megafon", "tele2", "rostelecom"]


def embed(model, texts, prefix, bs=32):
    if prefix:
        texts = [prefix + t for t in texts]
    return model.encode(texts, batch_size=bs, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prefix-query", default="")
    ap.add_argument("--prefix-doc", default="")
    ap.add_argument("--reranker", default="")
    ap.add_argument("--rerank-pool", type=int, default=20)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "eval" / "results" / "rag_multi_eval.json")
    args = ap.parse_args()

    chunks = load_kb(KB_DIR, recursive=True)
    print(f"KB chunks={len(chunks)} docs={len(set(c.doc_id for c in chunks))} model={args.model}", flush=True)
    t0 = time.perf_counter()
    model = SentenceTransformer(args.model, trust_remote_code=True)
    print(f"embedder loaded {time.perf_counter()-t0:.1f}s", flush=True)

    reranker = None
    if args.reranker:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(args.reranker, trust_remote_code=True)

    per_company = {}
    for comp in COMPANIES:
        cch = [c for c in chunks if c.company == comp]
        texts = [c.to_embedding_text() for c in cch]
        emb = embed(model, texts, args.prefix_doc)
        per_company[comp] = {
            "doc_ids": [c.doc_id for c in cch],
            "texts": texts,
            "emb": emb,
        }

    K = [1, 3, 5]
    agg = {k: 0 for k in K}
    agg_mrr = 0.0
    total = 0
    lat = []
    by_comp = {}
    for comp in COMPANIES:
        qs = json.loads((PROJECT_ROOT / "eval" / f"rag_queries_{comp}.json").read_text(
            encoding="utf-8"))["queries"]
        idx = per_company[comp]
        doc_ids, texts, demb = idx["doc_ids"], idx["texts"], idx["emb"]
        c_recall = {k: 0 for k in K}
        c_mrr = 0.0
        for item in qs:
            q, rel = item["q"], set(item["relevant"])
            tq = time.perf_counter()
            qe = embed(model, [q], args.prefix_query)[0]
            sims = demb @ qe
            order = np.argsort(-sims)
            ranked, seen = [], set()
            for i in order:
                d = doc_ids[i]
                if d not in seen:
                    seen.add(d); ranked.append((d, i))
                if len(ranked) >= max(args.rerank_pool, max(K)):
                    break
            if reranker is not None:
                pairs = [[q, texts[i]] for _, i in ranked]
                sc = reranker.predict(pairs)
                ranked = [ranked[j] for j in np.argsort(-np.asarray(sc))]
            lat.append((time.perf_counter() - tq) * 1000)
            rd = [d for d, _ in ranked]
            r1 = next((i + 1 for i, d in enumerate(rd) if d in rel), 0)
            if r1:
                c_mrr += 1.0 / r1
            for k in K:
                if rel & set(rd[:k]):
                    c_recall[k] += 1
        n = len(qs)
        by_comp[comp] = {"recall@1": round(c_recall[1]/n, 3), "recall@3": round(c_recall[3]/n, 3),
                         "recall@5": round(c_recall[5]/n, 3), "mrr": round(c_mrr/n, 3), "n": n}
        for k in K:
            agg[k] += c_recall[k]
        agg_mrr += c_mrr
        total += n

    res = {
        "model": args.model,
        "reranker": args.reranker or None,
        "recall@1": round(agg[1]/total, 4),
        "recall@3": round(agg[3]/total, 4),
        "recall@5": round(agg[5]/total, 4),
        "mrr": round(agg_mrr/total, 4),
        "latency_ms_p50": round(float(np.percentile(lat, 50)), 1),
        "latency_ms_p95": round(float(np.percentile(lat, 95)), 1),
        "n_queries": total,
        "by_company": by_comp,
    }
    print(f"\n=== RAG multi-tenant eval ({total} запросов, 5 компаний) ===")
    for k in ["recall@1", "recall@3", "recall@5", "mrr", "latency_ms_p50", "latency_ms_p95"]:
        print(f"  {k}: {res[k]}")
    print("  по компаниям:")
    for c, m in by_comp.items():
        print(f"    {c:11s} R@1={m['recall@1']} R@5={m['recall@5']} MRR={m['mrr']}")

    runs = []
    if args.out.exists():
        try:
            runs = json.loads(args.out.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            runs = []
    runs.append(res)
    args.out.write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
