"""Эксперимент по улучшению RAG на разговорных «Вектор»-запросах из E2E-прогона."""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from sentence_transformers import CrossEncoder, SentenceTransformer  # noqa: E402
import torch  # noqa: F401,E402

import chromadb  # noqa: E402
from chromadb.config import Settings as ChromaSettings  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402
import json  # noqa: E402

CHROMA = PROJECT_ROOT / "data" / "chroma"
RES = PROJECT_ROOT / "eval" / "results" / "e2e_dialogues_result.json"
COMPANY = "vektor"
TOPN = 20


def toks(t: str):
    return re.findall(r"[а-яёa-z0-9]+", t.lower())


def docs_from_chunks(chunk_ids, metas):
    """Список doc_id в порядке появления (дедуп)."""
    seen, out = set(), []
    for cid, m in zip(chunk_ids, metas):
        d = (m or {}).get("doc_id")
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return out


def metrics(ranklists, golds):
    r1 = r3 = r5 = 0
    mrr = 0.0
    for docs, gold in zip(ranklists, golds):
        gset = set(gold)
        if docs[:1] and docs[0] in gset:
            r1 += 1
        if any(d in gset for d in docs[:3]):
            r3 += 1
        if any(d in gset for d in docs[:5]):
            r5 += 1
        rank = next((i + 1 for i, d in enumerate(docs) if d in gset), 0)
        if rank:
            mrr += 1 / rank
    n = len(golds)
    return r1 / n, r3 / n, r5 / n, mrr / n


def main() -> None:
    data = json.loads(RES.read_text(encoding="utf-8"))
    rows = [r for r in data["turns"] if r.get("wer") is not None and r["company"] == COMPANY]
    by_dlg = defaultdict(list)
    for r in rows:
        by_dlg[r["dialogue_id"]].append(r)
    queries = []
    for r in rows:
        seq = by_dlg[r["dialogue_id"]]
        i = seq.index(r)
        prev = seq[i - 1]["asr_text"] if i > 0 else ""
        queries.append({
            "last": r["asr_text"],
            "ctx": (prev + " " + r["asr_text"]).strip(),
            "gold": r.get("gold_doc_ids", []),
        })
    golds = [q["gold"] for q in queries]
    print(f"запросов (vektor): {len(queries)}", flush=True)

    print("loading FRIDA + chroma + reranker…", flush=True)
    emb = SentenceTransformer("ai-forever/FRIDA", trust_remote_code=True)
    client = chromadb.PersistentClient(path=str(CHROMA),
                                       settings=ChromaSettings(anonymized_telemetry=False))
    col = client.get_collection("telecom_kb_frida")
    got = col.get(where={"company": COMPANY}, include=["documents", "metadatas"])
    chunk_texts = got["documents"]
    chunk_meta = got["metadatas"]
    chunk_doc = [m["doc_id"] for m in chunk_meta]
    bm25 = BM25Okapi([toks(t) for t in chunk_texts])
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)

    dense_ctx, dense_last, bm25_rank, hybrid, rr_dense, rr_hybrid = ([] for _ in range(6))

    q_ctx_emb = emb.encode(["search_query: " + q["ctx"] for q in queries],
                           normalize_embeddings=True, convert_to_numpy=True, batch_size=16)
    q_last_emb = emb.encode(["search_query: " + q["last"] for q in queries],
                            normalize_embeddings=True, convert_to_numpy=True, batch_size=16)

    for qi, q in enumerate(queries):
        rc = col.query(query_embeddings=[q_ctx_emb[qi].tolist()], n_results=TOPN,
                       where={"company": COMPANY}, include=["metadatas"])
        dctx_docs = docs_from_chunks(rc["ids"][0], rc["metadatas"][0])
        dense_ctx.append(dctx_docs)
        rl = col.query(query_embeddings=[q_last_emb[qi].tolist()], n_results=TOPN,
                       where={"company": COMPANY}, include=["metadatas"])
        dense_last.append(docs_from_chunks(rl["ids"][0], rl["metadatas"][0]))
        sc = bm25.get_scores(toks(q["ctx"]))
        order = sorted(range(len(sc)), key=lambda i: -sc[i])[:TOPN]
        bm_doc_order = []
        seen = set()
        for i in order:
            d = chunk_doc[i]
            if d not in seen:
                seen.add(d); bm_doc_order.append(d)
        bm25_rank.append(bm_doc_order)
        rrf = defaultdict(float)
        for rank, d in enumerate(dctx_docs):
            rrf[d] += 1 / (60 + rank)
        for rank, d in enumerate(bm_doc_order):
            rrf[d] += 1 / (60 + rank)
        hyb = [d for d, _ in sorted(rrf.items(), key=lambda x: -x[1])]
        hybrid.append(hyb)
        cand_idx = [i for i in range(len(chunk_texts))
                    if chunk_doc[i] in set(dctx_docs[:15])]
        if cand_idx:
            pairs = [[q["ctx"], chunk_texts[i]] for i in cand_idx]
            scores = reranker.predict(pairs)
            rr_order = sorted(range(len(cand_idx)), key=lambda j: -scores[j])
            rr_docs, seen = [], set()
            for j in rr_order:
                d = chunk_doc[cand_idx[j]]
                if d not in seen:
                    seen.add(d); rr_docs.append(d)
            rr_dense.append(rr_docs)
        else:
            rr_dense.append(dctx_docs)
        cand_idx2 = [i for i in range(len(chunk_texts)) if chunk_doc[i] in set(hyb[:8])]
        if cand_idx2:
            pairs = [[q["ctx"], chunk_texts[i]] for i in cand_idx2]
            scores = reranker.predict(pairs)
            o = sorted(range(len(cand_idx2)), key=lambda j: -scores[j])
            d2, seen = [], set()
            for j in o:
                d = chunk_doc[cand_idx2[j]]
                if d not in seen:
                    seen.add(d); d2.append(d)
            rr_hybrid.append(d2)
        else:
            rr_hybrid.append(hyb)
        if (qi + 1) % 50 == 0:
            print(f"  {qi+1}/{len(queries)}", flush=True)

    variants = {
        "dense (ctx) — текущий": dense_ctx,
        "dense (только послед.)": dense_last,
        "BM25": bm25_rank,
        "гибрид BM25+dense (RRF)": hybrid,
        "реранк поверх dense": rr_dense,
        "реранк поверх гибрида": rr_hybrid,
    }
    print(f"\n{'вариант':30s}  R@1    R@3    R@5    MRR")
    out = {}
    for name, rl in variants.items():
        r1, r3, r5, mrr = metrics(rl, golds)
        out[name] = {"r1": r1, "r3": r3, "r5": r5, "mrr": mrr}
        print(f"{name:30s}  {r1:.3f}  {r3:.3f}  {r5:.3f}  {mrr:.3f}")
    (PROJECT_ROOT / "eval" / "results" / "rag_experiment.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
