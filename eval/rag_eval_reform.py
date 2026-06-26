"""Сравнение качества ретрива (FRIDA dense) при разных запросах:."""
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
    tlite = json.loads((R / "reformulated_queries.json").read_text(encoding="utf-8"))
    qwen3 = json.loads((R / "reformulated_qwen3_14b.json").read_text(encoding="utf-8"))
    claude = json.loads((R / "claude_reformulated.json").read_text(encoding="utf-8"))
    claude_cl = json.loads((R / "claude_reformulated_clean.json").read_text(encoding="utf-8"))
    m = SentenceTransformer("ai-forever/FRIDA", trust_remote_code=True)
    client = chromadb.PersistentClient(path=str(CHROMA),
                                       settings=ChromaSettings(anonymized_telemetry=False))
    col = client.get_collection("telecom_kb_frida")
    got = col.get(include=["documents", "metadatas"])
    texts = got["documents"]; cdoc = [x["doc_id"] for x in got["metadatas"]]
    comp_idx = defaultdict(list)
    for i, x in enumerate(got["metadatas"]):
        comp_idx[x["company"]].append(i)
    demb = m.encode(["search_document: " + t for t in texts], normalize_embeddings=True,
                    convert_to_numpy=True, batch_size=32)

    for tag, path in DATASETS:
        rows = [r for r in json.loads(path.read_text(encoding="utf-8"))["turns"] if r.get("gold_doc_ids")]
        by = defaultdict(list)
        for r in rows:
            by[r["dialogue_id"]].append(r)
        Q = []
        for r in rows:
            seq = by[r["dialogue_id"]]; i = seq.index(r)
            prev = seq[i-1]["asr_text"] if i > 0 else ""
            key = f"{r['company']}|{r['dialogue_id']}|{r['turn_idx']}"
            Q.append({"company": r["company"], "gold": r["gold_doc_ids"],
                      "baseline": (prev + " " + r["asr_text"]).strip(),
                      "tlite": tlite.get(key, r["asr_text"]),
                      "qwen3": qwen3.get(key, r["asr_text"]),
                      "claude_clean": claude_cl.get(key, r["asr_text"]),
                      "claude": claude.get(key, r["asr_text"])})
        golds = [q["gold"] for q in Q]

        print(f"\n[{tag}] запросов: {len(Q)}")
        print(f"  {'тип запроса':26s} R@1    R@3    R@5    MRR")
        for qt in ["baseline", "tlite", "qwen3", "claude_clean", "claude"]:
            qemb = m.encode(["search_query: " + q[qt] for q in Q], normalize_embeddings=True,
                            convert_to_numpy=True, batch_size=16)
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
            r1, r3, r5, mrr = metrics(rls, golds)
            label = {"baseline": "baseline (сырая реплика)", "tlite": "T-lite переформ.",
                     "qwen3": "Qwen3 переформ.", "claude_clean": "Claude переформ. (без gold)",
                     "claude": "Claude переформ. (с файлами)"}[qt]
            print(f"  {label:26s} {r1:.3f}  {r3:.3f}  {r5:.3f}  {mrr:.3f}")


if __name__ == "__main__":
    main()
