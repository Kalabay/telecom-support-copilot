"""doc2query: для каждой статьи базы LLM генерит разговорные реплики клиента по теме."""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import app  # noqa: F401,E402
import sentence_transformers  # noqa: F401,E402
import torch  # noqa: F401,E402
import chromadb  # noqa: E402
from chromadb.config import Settings as ChromaSettings  # noqa: E402

R = PROJECT_ROOT / "eval" / "results"
CHROMA = PROJECT_ROOT / "data" / "chroma"

SYS = (
    "Ты — помощник по созданию обучающих данных для поиска в базе знаний телеком-поддержки. "
    "По статье базы придумай реплики, с которыми РЕАЛЬНЫЙ клиент позвонил бы оператору по этой "
    "теме: разговорно, эмоционально, своими словами, как живой человек (не формально, без терминов "
    "из статьи). Разные формулировки и интонации. Каждую — с новой строки, без нумерации и кавычек."
)


def main() -> None:
    from app.pipeline.llm import get_generator
    gen = get_generator(); gen._ensure_loaded()
    nothink = "\n/no_think" if os.environ.get("LLM_BACKEND", "").startswith("qwen") else ""
    print("LLM загружен", flush=True)

    client = chromadb.PersistentClient(path=str(CHROMA),
                                       settings=ChromaSettings(anonymized_telemetry=False))
    col = client.get_collection("telecom_kb_frida")
    got = col.get(include=["documents", "metadatas"])
    docs = defaultdict(lambda: {"text": "", "company": "", "title": ""})
    for txt, m in zip(got["documents"], got["metadatas"]):
        d = docs[m["doc_id"]]
        d["company"] = m["company"]; d["title"] = m.get("title", m["doc_id"])
        if len(d["text"]) < 1500:
            d["text"] += " " + txt
    print(f"документов: {len(docs)}", flush=True)

    out = []
    for i, (doc_id, d) in enumerate(docs.items()):
        user = (f"Статья (тема: {d['title']}):\n{d['text'][:1500]}\n\n"
                "Напиши 4 разные реплики клиента по этой теме (разговорно, эмоционально):")
        res = gen._llm.create_chat_completion(
            messages=[{"role": "system", "content": SYS + nothink},
                      {"role": "user", "content": user}],
            max_tokens=220, temperature=0.7, top_p=0.9,
            stop=["<|im_end|>", "<|eot_id|>", "<|message_sep|>"],
        )
        raw = re.sub(r"<think>.*?</think>", "", res["choices"][0]["message"]["content"], flags=re.DOTALL)
        raw = raw.replace("<think>", "").replace("</think>", "")
        phr = [re.sub(r"^[\d\-\.\)\s«\"]+", "", ln).strip().strip('"«».').strip()
               for ln in raw.splitlines() if len(ln.strip()) > 8]
        out.append({"doc_id": doc_id, "company": d["company"], "title": d["title"],
                    "phrasings": phr[:4]})
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(docs)}", flush=True)

    (R / "doc2query_phrasings.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    print(f"\n{len(out)} статей -> doc2query_phrasings.json")
    for o in out[:2]:
        print(f"  [{o['doc_id']}] {o['phrasings']}")


if __name__ == "__main__":
    main()
