"""Условие C: gold-документ + SPECIFIC-промпт (назови ответ из документа, не хеджируй)."""
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

V_SPECIFIC = (
    "Ты — AI-ассистент оператора телеком-поддержки. Подскажи короткую реплику (1-2 предложения, "
    "по-русски, разговорно) клиенту.\n"
    "ГЛАВНОЕ: НАЗОВИ конкретный ответ/следующий шаг ИЗ приведённых документов — причину, факт, "
    "опцию, как опытный оператор, а не общую фразу 'я проверю' или 'оформлю обращение'. Если в "
    "документе есть прямой ответ на вопрос клиента (причина сбоя, бесплатно/платно, что делать) — "
    "СКАЖИ его прямо.\n"
    "Отвечай ИМЕННО на реплику клиента, учитывай что он уже сказал и пробовал.\n"
    "Если раздражён — короткое признание и извинение, затем сразу конкретный шаг.\n"
    "Не выдумывай того, чего НЕТ в документах (числа, сроки, статусы 'уже сделал'); но всё, что в "
    "документах есть, — используй уверенно."
)


def build_user(docs, transcript):
    kb = "\n".join(f"[{i+1}] {d['title']}: {d['text']}" for i, d in enumerate(docs))
    hist = "\n".join(f"{'Клиент' if t['speaker']=='customer' else 'Оператор'}: {t['text']}"
                     for t in transcript)
    return (f"## Документы базы знаний\n{kb}\n\n## Диалог (клиент раздражён)\n{hist}\n\n"
            "Подскажи ОДНУ короткую реплику оператора — только реплику.")


def clean(raw):
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = raw.replace("<think>", "").replace("</think>", "").strip()
    raw = re.sub(r"^(Подсказка|Реплика|Оператор)\s*:\s*", "", raw, flags=re.I).strip()
    return (raw.split("\n\n")[0]).strip().strip('"«»').strip()


def main() -> None:
    from app.pipeline.llm import get_generator
    gold = {}
    for f in ["e2e_real_result.json", "e2e_dialogues_result.json"]:
        for t in json.loads((R / f).read_text(encoding="utf-8"))["turns"]:
            if t.get("gold_doc_ids"):
                gold[(t["dialogue_id"], t["turn_idx"])] = t["gold_doc_ids"]
    client = chromadb.PersistentClient(path=str(PROJECT_ROOT / "data" / "chroma"),
                                       settings=ChromaSettings(anonymized_telemetry=False))
    got = client.get_collection("telecom_kb_frida").get(include=["documents", "metadatas"])
    dtext = defaultdict(str); dtitle = {}
    for txt, m in zip(got["documents"], got["metadatas"]):
        if len(dtext[m["doc_id"]]) < 1400:
            dtext[m["doc_id"]] = (dtext[m["doc_id"]] + " " + txt).strip()
        dtitle[m["doc_id"]] = m.get("title", m["doc_id"])

    gen = get_generator(); gen._ensure_loaded()
    nothink = "\n/no_think" if os.environ.get("LLM_BACKEND", "").startswith("qwen") else ""
    angry = json.loads((R / "compare_data.json").read_text(encoding="utf-8"))
    out = []
    for i, d in enumerate(angry):
        gd = [g for g in gold.get((d["dialogue_id"], d["turn_idx"]), []) if g in dtext]
        if not gd:
            continue
        docs = [{"title": dtitle[g], "text": dtext[g]} for g in gd]
        res = gen._llm.create_chat_completion(
            messages=[{"role": "system", "content": V_SPECIFIC + nothink},
                      {"role": "user", "content": build_user(docs, d["transcript"])}],
            max_tokens=220, temperature=0.5, top_p=0.8,
            stop=["\n\n\n", "<|im_end|>", "<|message_sep|>"],
        )
        out.append({"dialogue_id": d["dialogue_id"], "turn_idx": d["turn_idx"],
                    "asr_text": d["asr_text"], "ideal_text": d["ideal_text"],
                    "answer_C_gold_specific": clean(res["choices"][0]["message"]["content"])})
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(angry)}", flush=True)
    (R / "perfect_specific.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(out)} -> perfect_specific.json")


if __name__ == "__main__":
    main()
