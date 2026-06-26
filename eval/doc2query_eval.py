"""Замер doc2query: поиск по базовому индексу (чанки) vs обогащённому (чанки + реплики клиента)."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
from sentence_transformers import SentenceTransformer  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: F401,E402
import chromadb  # noqa: E402
from chromadb.config import Settings as ChromaSettings  # noqa: E402

R = PROJECT_ROOT / "eval" / "results"
DATASETS = [("real", R / "e2e_real_result.json"), ("Вектор", R / "e2e_dialogues_result.json")]


def metrics(rls, golds):
    r1 = r3 = r5 = 0; mrr = 0.0
    for docs, gold in zip(rls, golds):
        g = set(gold)
        if docs[:1] and docs[0] in g: r1 += 1
        if any(d in g for d in docs[:3]): r3 += 1
        if any(d in g for d in docs[:5]): r5 += 1
        rk = next((i+1 for i, d in enumerate(docs) if d in g), 0)
        if rk: mrr += 1/rk
    n = len(golds) or 1
    return r1/n, r3/n, r5/n, mrr/n


def rank(qvec, emb, idx, ent_doc):
    order = np.argsort(-(emb[idx] @ qvec))
    docs, seen = [], set()
    for j in order:
        d = ent_doc[idx[j]]
        if d not in seen:
            seen.add(d); docs.append(d)
    return docs


def main() -> None:
    m = SentenceTransformer("ai-forever/FRIDA", trust_remote_code=True)
    client = chromadb.PersistentClient(path=str(PROJECT_ROOT / "data" / "chroma"),
                                       settings=ChromaSettings(anonymized_telemetry=False))
    col = client.get_collection("telecom_kb_frida")
    got = col.get(include=["documents", "metadatas"])
    ent_text = list(got["documents"])
    ent_doc = [x["doc_id"] for x in got["metadatas"]]
    ent_comp = [x["company"] for x in got["metadatas"]]
    n_base = len(ent_text)
    phr = json.loads((R / "doc2query_phrasings.json").read_text(encoding="utf-8"))
    for o in phr:
        for p in o.get("phrasings", []):
            if p and len(p) > 5:
                ent_text.append(p); ent_doc.append(o["doc_id"]); ent_comp.append(o["company"])
    print(f"записей: база {n_base} чанков + {len(ent_text)-n_base} реплик = {len(ent_text)}", flush=True)

    emb = m.encode(["search_document: " + t for t in ent_text], normalize_embeddings=True,
                   convert_to_numpy=True, batch_size=32)
    base_idx = defaultdict(list); enr_idx = defaultdict(list)
    for i, c in enumerate(ent_comp):
        enr_idx[c].append(i)
        if i < n_base:
            base_idx[c].append(i)

    for tag, path in DATASETS:
        rows = [r for r in json.loads(path.read_text(encoding="utf-8"))["turns"] if r.get("gold_doc_ids")]
        by = defaultdict(list)
        for r in rows:
            by[r["dialogue_id"]].append(r)
        Q = []
        for r in rows:
            seq = by[r["dialogue_id"]]; i = seq.index(r)
            prev = seq[i-1]["asr_text"] if i > 0 else ""
            Q.append({"company": r["company"], "gold": r["gold_doc_ids"],
                      "q": (prev + " " + r["asr_text"]).strip()})
        golds = [q["gold"] for q in Q]
        qemb = m.encode(["search_query: " + q["q"] for q in Q], normalize_embeddings=True,
                        convert_to_numpy=True, batch_size=16)
        print(f"\n[{tag}] запросов: {len(Q)}")
        print(f"  {'индекс':24s} R@1    R@3    R@5    MRR")
        for name, idxmap in [("базовый (чанки)", base_idx), ("обогащённый (+реплики)", enr_idx)]:
            rls = [rank(qemb[qi], emb, np.array(idxmap[q["company"]]), ent_doc) for qi, q in enumerate(Q)]
            r1, r3, r5, mrr = metrics(rls, golds)
            print(f"  {name:24s} {r1:.3f}  {r3:.3f}  {r5:.3f}  {mrr:.3f}")


if __name__ == "__main__":
    main()
