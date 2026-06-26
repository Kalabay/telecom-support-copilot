"""Answer-bank ретрив: индексируем ГОТОВЫЕ ответы плейбука и ищем по реплике клиента."""
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


def main() -> None:
    m = SentenceTransformer("ai-forever/FRIDA", trust_remote_code=True)
    client = chromadb.PersistentClient(path=str(PROJECT_ROOT / "data" / "chroma"),
                                       settings=ChromaSettings(anonymized_telemetry=False))
    got = client.get_collection("telecom_kb_frida").get(include=["documents", "metadatas"])
    dtext = defaultdict(str); dtitle = {}; dcomp = {}
    for txt, x in zip(got["documents"], got["metadatas"]):
        if len(dtext[x["doc_id"]]) < 1200:
            dtext[x["doc_id"]] = (dtext[x["doc_id"]] + " " + txt).strip()
        dtitle[x["doc_id"]] = x.get("title", x["doc_id"]); dcomp[x["doc_id"]] = x["company"]

    bank_text, bank_doc, bank_comp = [], [], []
    for o in json.loads((R / "doc_answers.json").read_text(encoding="utf-8")):
        for a in o.get("answers", []):
            if a and len(a) > 8:
                bank_text.append(a); bank_doc.append(o["doc_id"]); bank_comp.append(o["company"])
    print(f"банк ответов: {len(bank_text)} реплик", flush=True)
    bemb = m.encode(["search_document: " + t for t in bank_text], normalize_embeddings=True,
                    convert_to_numpy=True, batch_size=32)
    comp_idx = defaultdict(list)
    for i, c in enumerate(bank_comp):
        comp_idx[c].append(i)

    angry = json.loads((R / "compare_data.json").read_text(encoding="utf-8"))
    out = []
    for d in angry:
        cust = [t["text"] for t in d["transcript"] if t["speaker"] == "customer"]
        query = (cust[-2] + " " + cust[-1]).strip() if len(cust) >= 2 else (cust[-1] if cust else d["asr_text"])
        qe = m.encode(["search_query: " + query], normalize_embeddings=True, convert_to_numpy=True)[0]
        idx = comp_idx[d["company"]]
        order = np.argsort(-(bemb[idx] @ qe))[:8]
        cand = [bank_text[idx[j]] for j in order]
        seen, docs = set(), []
        for j in order:
            dd = bank_doc[idx[j]]
            if dd not in seen:
                seen.add(dd); docs.append(dd)
            if len(docs) >= 3:
                break
        out.append({"dialogue_id": d["dialogue_id"], "turn_idx": d["turn_idx"], "dataset": d["dataset"],
                    "asr_text": d["asr_text"], "ideal_text": d["ideal_text"], "transcript": d["transcript"],
                    "doc_texts": [dtext[x] for x in docs], "candidate_answers": cand})
    (R / "playbook_ctx.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(out)} -> playbook_ctx.json")


if __name__ == "__main__":
    main()
