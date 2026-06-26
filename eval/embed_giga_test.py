"""Тест Giga-Embeddings (ai-sage, ruMTEB SOTA) на разговорных запросах vs FRIDA."""
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


def main() -> None:
    client = chromadb.PersistentClient(path=str(PROJECT_ROOT / "data" / "chroma"),
                                       settings=ChromaSettings(anonymized_telemetry=False))
    got = client.get_collection("telecom_kb_frida").get(include=["documents", "metadatas"])
    texts = got["documents"]; cdoc = [x["doc_id"] for x in got["metadatas"]]
    comp_idx = defaultdict(list)
    for i, x in enumerate(got["metadatas"]):
        comp_idx[x["company"]].append(i)
    qsets = {}
    for tag, path in DATASETS:
        rows = [r for r in json.loads(path.read_text(encoding="utf-8"))["turns"] if r.get("gold_doc_ids")]
        by = defaultdict(list)
        for r in rows:
            by[r["dialogue_id"]].append(r)
        Q = []
        for r in rows:
            seq = by[r["dialogue_id"]]; i = seq.index(r)
            prev = seq[i-1]["asr_text"] if i > 0 else ""
            Q.append({"company": r["company"], "gold": r["gold_doc_ids"], "ctx": (prev + " " + r["asr_text"]).strip()})
        qsets[tag] = Q

    print("грузим Giga-Embeddings…", flush=True)
    try:
        m = SentenceTransformer("ai-sage/Giga-Embeddings-instruct", trust_remote_code=True)
    except Exception as e:  # noqa: BLE001
        print(f"НЕ ЗАГРУЗИЛСЯ через SentenceTransformer: {type(e).__name__}: {str(e)[:200]}")
        print("Giga-Embeddings требует кастомной загрузки (decoder + latent-attn) — пропускаю.")
        return
    demb = m.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=8)
    for tag, Q in qsets.items():
        qemb = m.encode([q["ctx"] for q in Q], normalize_embeddings=True, convert_to_numpy=True, batch_size=8,
                        prompt="Дан вопрос, найди релевантный пассаж: ")
        rls = []
        for qi, q in enumerate(Q):
            idx = comp_idx[q["company"]]
            order = np.argsort(-(demb[idx] @ qemb[qi]))
            docs, seen = [], set()
            for j in order:
                d = cdoc[idx[j]]
                if d not in seen:
                    seen.add(d); docs.append(d)
            rls.append(docs)
        r1, r3, r5, mrr = metrics(rls, [q["gold"] for q in Q])
        print(f"  [{tag:7s}] R@1={r1:.3f}  R@3={r3:.3f}  R@5={r5:.3f}  MRR={mrr:.3f}  (FRIDA real был 0.508/Вектор 0.500)")


if __name__ == "__main__":
    main()
