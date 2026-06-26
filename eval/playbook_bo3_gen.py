"""Best-of-3 на лучшей версии: Mistral-24B + gold-документ + плейбук, ПО 3 варианта на реплику."""
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

SYS = (
    "Ты — AI-ассистент оператора телеком-поддержки. Тебе дают документ базы знаний, несколько "
    "ГОТОВЫХ образцовых ответов оператора по этой теме и диалог с раздражённым клиентом.\n"
    "Твоя ГЛАВНАЯ задача: выбери наиболее подходящий под реплику клиента готовый ответ и "
    "адаптируй его — подгони тон и детали под этого клиента, ответь именно на его вопрос. "
    "Если ни один готовый ответ точно не подходит — ответь по документу.\n"
    "Не выдумывай чисел, сроков, сумм и фактов, которых нет в материалах. Реплика короткая "
    "(1-2 предложения), по-русски, разговорно."
)


def build_user(doc_text, answers, transcript):
    ans = "\n".join(f"- {a}" for a in answers) or "(нет)"
    hist = "\n".join(f"{'Клиент' if t['speaker']=='customer' else 'Оператор'}: {t['text']}"
                     for t in transcript)
    return (f"## Документ базы знаний\n{doc_text}\n\n## Готовые образцовые ответы по теме\n{ans}\n\n"
            f"## Диалог\n{hist}\n\nДай ОДНУ короткую реплику оператора — только реплику.")


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
    dtext = defaultdict(str)
    for txt, m in zip(got["documents"], got["metadatas"]):
        if len(dtext[m["doc_id"]]) < 1200:
            dtext[m["doc_id"]] = (dtext[m["doc_id"]] + " " + txt).strip()
    answers = {o["doc_id"]: o.get("answers", []) for o in
               json.loads((R / "doc_answers.json").read_text(encoding="utf-8"))}

    gen = get_generator(); gen._ensure_loaded()
    angry = json.loads((R / "compare_data.json").read_text(encoding="utf-8"))
    out = []
    for i, d in enumerate(angry):
        gd = [g for g in gold.get((d["dialogue_id"], d["turn_idx"]), []) if g in dtext]
        if not gd:
            continue
        doc_text = "\n---\n".join(dtext[g] for g in gd)
        ans = [a for g in gd for a in answers.get(g, [])]
        user = build_user(doc_text, ans, d["transcript"])
        variants = []
        for _ in range(3):
            res = gen._llm.create_chat_completion(
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": user}],
                max_tokens=220, temperature=0.75, top_p=0.95,
                stop=["\n\n\n", "<|im_end|>", "<|message_sep|>"],
            )
            variants.append(clean(res["choices"][0]["message"]["content"]))
        out.append({"dialogue_id": d["dialogue_id"], "turn_idx": d["turn_idx"],
                    "asr_text": d["asr_text"], "ideal_text": d["ideal_text"],
                    "v1": variants[0], "v2": variants[1], "v3": variants[2]})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(angry)}", flush=True)
    backend = os.environ.get("LLM_BACKEND", "mistral24")
    dest = R / ("bo3_blind.json" if backend == "mistral24" else f"bo3_pb_{backend}.json")
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(out)} реплик × 3 варианта -> {dest.name}")


if __name__ == "__main__":
    main()
