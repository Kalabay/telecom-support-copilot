"""Сравнение эмбеддеров на E2E-разговорных запросах (real + Вектор)."""
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

CHROMA = PROJECT_ROOT / "data" / "chroma"
R = PROJECT_ROOT / "eval" / "results"
DATASETS = [("real", R / "e2e_real_result.json"), ("Вектор", R / "e2e_dialogues_result.json")]
EMBEDDERS = [
    ("FRIDA (текущий)", "ai-forever/FRIDA", "search_query: ", "search_document: "),
    ("USER2-base", "deepvk/USER2-base", "search_query: ", "search_document: "),
    ("USER-bge-m3", "deepvk/USER-bge-m3", "", ""),
    ("Qwen3-Emb-0.6B", "Qwen/Qwen3-Embedding-0.6B",
     "Instruct: Найди статью базы знаний оператора по реплике клиента\nQuery:", ""),
]


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
    client = chromadb.PersistentClient(path=str(CHROMA),
                                       settings=ChromaSettings(anonymized_telemetry=False))
    col = client.get_collection("telecom_kb_frida")
    got = col.get(include=["documents", "metadatas"])
    texts = got["documents"]
    cdoc = [m["doc_id"] for m in got["metadatas"]]
    ccomp = [m["company"] for m in got["metadatas"]]
    comp_idx = defaultdict(list)
    for i, c in enumerate(ccomp):
        comp_idx[c].append(i)
    print(f"чанков всего: {len(texts)}", flush=True)

    qsets = {}
    for tag, path in DATASETS:
        rows = [r for r in json.loads(path.read_text(encoding="utf-8"))["turns"] if r.get("gold_doc_ids")]
        by_dlg = defaultdict(list)
        for r in rows:
            by_dlg[r["dialogue_id"]].append(r)
        Q = []
        for r in rows:
            seq = by_dlg[r["dialogue_id"]]; i = seq.index(r)
            prev = seq[i-1]["asr_text"] if i > 0 else ""
            Q.append({"company": r["company"], "gold": r["gold_doc_ids"],
                      "ctx": (prev + " " + r["asr_text"]).strip()})
        qsets[tag] = Q

    results = {}
    for label, model_name, qp, dp in EMBEDDERS:
        print(f"\n=== {label} ({model_name}) ===", flush=True)
        try:
            m = SentenceTransformer(model_name, trust_remote_code=True)
        except Exception as e:  # noqa: BLE001
            print(f"  не загрузился: {e}"); continue
        demb = m.encode([dp + t for t in texts], normalize_embeddings=True,
                        convert_to_numpy=True, batch_size=32)
        results[label] = {}
        for tag, Q in qsets.items():
            qemb = m.encode([qp + q["ctx"] for q in Q], normalize_embeddings=True,
                            convert_to_numpy=True, batch_size=16)
            ranklists = []
            for qi, q in enumerate(Q):
                idx = comp_idx[q["company"]]
                sims = demb[idx] @ qemb[qi]
                order = np.argsort(-sims)
                docs, seen = [], set()
                for j in order:
                    d = cdoc[idx[j]]
                    if d not in seen:
                        seen.add(d); docs.append(d)
                ranklists.append(docs)
            r1, r3, r5, mrr = metrics(ranklists, [q["gold"] for q in Q])
            results[label][tag] = {"R@1": round(r1, 3), "R@3": round(r3, 3),
                                   "R@5": round(r5, 3), "MRR": round(mrr, 3)}
            print(f"  [{tag:7s}] R@1={r1:.3f}  R@3={r3:.3f}  R@5={r5:.3f}  MRR={mrr:.3f}", flush=True)
        del m, demb
        torch.cuda.empty_cache()

    (R / "embed_compare.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nsaved -> eval/results/embed_compare.json")


if __name__ == "__main__":
    main()
