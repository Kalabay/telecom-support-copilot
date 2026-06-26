"""Наша ЛУЧШАЯ текущая версия end-to-end: doc2query-обогащённый RAG -> Qwen3 -> tight-промпт."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import app  # noqa: F401,E402
from sentence_transformers import SentenceTransformer  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: F401,E402
import chromadb  # noqa: E402
from chromadb.config import Settings as ChromaSettings  # noqa: E402

from app.models.schemas import Emotion, EmotionState, KBSource  # noqa: E402

R = PROJECT_ROOT / "eval" / "results"


def main() -> None:
    from app.pipeline.llm import get_generator
    m = SentenceTransformer("ai-forever/FRIDA", trust_remote_code=True)
    client = chromadb.PersistentClient(path=str(PROJECT_ROOT / "data" / "chroma"),
                                       settings=ChromaSettings(anonymized_telemetry=False))
    got = client.get_collection("telecom_kb_frida").get(include=["documents", "metadatas"])
    ent_text = list(got["documents"]); ent_doc = [x["doc_id"] for x in got["metadatas"]]
    ent_comp = [x["company"] for x in got["metadatas"]]
    dtext = defaultdict(str); dtitle = {}
    for txt, x in zip(got["documents"], got["metadatas"]):
        if len(dtext[x["doc_id"]]) < 1400:
            dtext[x["doc_id"]] = (dtext[x["doc_id"]] + " " + txt).strip()
        dtitle[x["doc_id"]] = x.get("title", x["doc_id"])
    for o in json.loads((R / "doc2query_phrasings.json").read_text(encoding="utf-8")):
        for p in o.get("phrasings", []):
            if p and len(p) > 5:
                ent_text.append(p); ent_doc.append(o["doc_id"]); ent_comp.append(o["company"])
    emb = m.encode(["search_document: " + t for t in ent_text], normalize_embeddings=True,
                   convert_to_numpy=True, batch_size=32)
    comp_idx = defaultdict(list)
    for i, c in enumerate(ent_comp):
        comp_idx[c].append(i)
    print("индекс обогащён, FRIDA готов", flush=True)

    gen = get_generator(); gen._ensure_loaded()
    emo = EmotionState(label=Emotion.ANGRY, confidence=0.85, arousal=0.85, valence=-0.7, escalation_risk=True)
    angry = json.loads((R / "compare_data.json").read_text(encoding="utf-8"))
    out = []
    for i, d in enumerate(angry):
        cust = [t["text"] for t in d["transcript"] if t["speaker"] == "customer"]
        query = (cust[-2] + " " + cust[-1]).strip() if len(cust) >= 2 else (cust[-1] if cust else d["asr_text"])
        qe = m.encode(["search_query: " + query], normalize_embeddings=True, convert_to_numpy=True)[0]
        idx = comp_idx[d["company"]]
        order = np.argsort(-(emb[idx] @ qe))
        top, seen = [], set()
        for j in order:
            dd = ent_doc[idx[j]]
            if dd not in seen:
                seen.add(dd); top.append(dd)
            if len(top) >= 3:
                break
        sources = [KBSource(doc_id=dd, title=dtitle[dd], snippet=dtext[dd], score=1.0) for dd in top]
        ans = gen.generate(d["transcript"], emo, sources, safe=False).suggestions
        out.append({"dialogue_id": d["dialogue_id"], "turn_idx": d["turn_idx"], "dataset": d["dataset"],
                    "asr_text": d["asr_text"], "ideal_text": d["ideal_text"],
                    "retrieved_docs": top, "answer": ans[0] if ans else ""})
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(angry)}", flush=True)
    (R / "current_best.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(out)} -> current_best.json")


if __name__ == "__main__":
    main()
