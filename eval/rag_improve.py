"""Методы улучшения RAG по отдельности на E2E-запросах (real + Вектор), FRIDA + chroma."""
from __future__ import annotations

import json
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

CHROMA = PROJECT_ROOT / "data" / "chroma"
R = PROJECT_ROOT / "eval" / "results"
TOPN = 20
DATASETS = [("real", R / "e2e_real_result.json"), ("Вектор", R / "e2e_dialogues_result.json")]


def toks(t):
    return re.findall(r"[а-яёa-z0-9]+", t.lower())


def docs_from(ids, metas):
    seen, out = set(), []
    for cid, m in zip(ids, metas):
        d = (m or {}).get("doc_id")
        if d and d not in seen:
            seen.add(d); out.append(d)
    return out


def metrics(ranklists, golds):
    r1 = r3 = r5 = 0; mrr = 0.0
    for docs, gold in zip(ranklists, golds):
        g = set(gold)
        if docs[:1] and docs[0] in g: r1 += 1
        if any(d in g for d in docs[:3]): r3 += 1
        if any(d in g for d in docs[:5]): r5 += 1
        rank = next((i + 1 for i, d in enumerate(docs) if d in g), 0)
        if rank: mrr += 1 / rank
    n = len(golds) or 1
    return r1/n, r3/n, r5/n, mrr/n


def main() -> None:
    reform = json.loads((R / "reformulated_queries.json").read_text(encoding="utf-8"))
    print("loading FRIDA + chroma + reranker…", flush=True)
    emb = SentenceTransformer("ai-forever/FRIDA", trust_remote_code=True)
    client = chromadb.PersistentClient(path=str(CHROMA),
                                       settings=ChromaSettings(anonymized_telemetry=False))
    col = client.get_collection("telecom_kb_frida")
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)

    comp_cache = {}

    def get_comp(company):
        if company not in comp_cache:
            got = col.get(where={"company": company}, include=["documents", "metadatas"])
            texts = got["documents"]; doc = [m["doc_id"] for m in got["metadatas"]]
            comp_cache[company] = (texts, doc, BM25Okapi([toks(t) for t in texts]))
        return comp_cache[company]

    all_out = {}
    for tag, path in DATASETS:
        rows = [r for r in json.loads(path.read_text(encoding="utf-8"))["turns"]
                if r.get("gold_doc_ids")]
        by_dlg = defaultdict(list)
        for r in rows:
            by_dlg[r["dialogue_id"]].append(r)
        Q = []
        for r in rows:
            seq = by_dlg[r["dialogue_id"]]; i = seq.index(r)
            prev = seq[i-1]["asr_text"] if i > 0 else ""
            Q.append({"company": r["company"], "gold": r["gold_doc_ids"],
                      "ctx": (prev + " " + r["asr_text"]).strip(),
                      "reform": reform.get(f"{r['company']}|{r['dialogue_id']}|{r['turn_idx']}", r["asr_text"])})
        golds = [q["gold"] for q in Q]
        print(f"\n[{tag}] запросов: {len(Q)}", flush=True)

        qc = emb.encode(["search_query: " + q["ctx"] for q in Q],
                        normalize_embeddings=True, convert_to_numpy=True, batch_size=16)
        qr = emb.encode(["search_query: " + q["reform"] for q in Q],
                        normalize_embeddings=True, convert_to_numpy=True, batch_size=16)

        dense_ctx, dense_ref, bm25r, hybrid, rr = ([] for _ in range(5))
        for qi, q in enumerate(Q):
            comp = q["company"]; texts, cdoc, bm = get_comp(comp)
            rc = col.query(query_embeddings=[qc[qi].tolist()], n_results=TOPN,
                           where={"company": comp}, include=["metadatas"])
            dc = docs_from(rc["ids"][0], rc["metadatas"][0]); dense_ctx.append(dc)
            rr2 = col.query(query_embeddings=[qr[qi].tolist()], n_results=TOPN,
                            where={"company": comp}, include=["metadatas"])
            dense_ref.append(docs_from(rr2["ids"][0], rr2["metadatas"][0]))
            sc = bm.get_scores(toks(q["ctx"]))
            order = sorted(range(len(sc)), key=lambda i: -sc[i])[:TOPN]
            bdoc, seen = [], set()
            for i in order:
                d = cdoc[i]
                if d not in seen: seen.add(d); bdoc.append(d)
            bm25r.append(bdoc)
            rrf = defaultdict(float)
            for rk, d in enumerate(dc): rrf[d] += 1/(60+rk)
            for rk, d in enumerate(bdoc): rrf[d] += 1/(60+rk)
            hybrid.append([d for d, _ in sorted(rrf.items(), key=lambda x: -x[1])])
            cand = [i for i in range(len(texts)) if cdoc[i] in set(dc[:15])]
            if cand:
                scs = reranker.predict([[q["ctx"], texts[i]] for i in cand])
                o = sorted(range(len(cand)), key=lambda j: -scs[j])
                rd, seen = [], set()
                for j in o:
                    d = cdoc[cand[j]]
                    if d not in seen: seen.add(d); rd.append(d)
                rr.append(rd)
            else:
                rr.append(dc)
            if (qi+1) % 60 == 0:
                print(f"  {qi+1}/{len(Q)}", flush=True)

        variants = {"dense (ctx) — ТЕКУЩИЙ": dense_ctx, "dense (переформулировка)": dense_ref,
                    "BM25": bm25r, "гибрид BM25+dense (RRF)": hybrid, "реранк поверх dense": rr}
        print(f"\n  {'метод':28s} R@1    R@3    R@5    MRR")
        out = {}
        for name, rl in variants.items():
            m = metrics(rl, golds); out[name] = dict(zip(["R@1", "R@3", "R@5", "MRR"], [round(x, 3) for x in m]))
            print(f"  {name:28s} {m[0]:.3f}  {m[1]:.3f}  {m[2]:.3f}  {m[3]:.3f}")
        all_out[tag] = out

    (R / "rag_improve.json").write_text(json.dumps(all_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nsaved -> eval/results/rag_improve.json")


if __name__ == "__main__":
    main()
