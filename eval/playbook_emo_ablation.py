"""Абляция эмоции на лучшем пайплайне: плейбук с эмоц-блоком и без него (одна модель, gold)."""
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

SYS_WITH = (
    "Ты — AI-ассистент оператора телеком-поддержки. Тебе дают документ базы знаний, готовые "
    "образцовые ответы по теме и диалог с РАЗДРАЖЁННЫМ клиентом.\n"
    "Выбери подходящий готовый ответ и адаптируй его. Клиент злой — поэтому сначала коротко "
    "признай ситуацию и извинись, и только потом давай решение; подгони тон под раздражение. "
    "Не выдумывай чисел и фактов вне материалов. 1-2 предложения, по-русски, разговорно."
)
SYS_WITHOUT = (
    "Ты — AI-ассистент оператора телеком-поддержки. Тебе дают документ базы знаний, готовые "
    "образцовые ответы по теме и диалог с клиентом.\n"
    "Выбери подходящий готовый ответ и адаптируй его — ответь по сути вопроса клиента. "
    "Не выдумывай чисел и фактов вне материалов. 1-2 предложения, по-русски, разговорно."
)


def build_user(doc_text, answers, transcript):
    ans = "\n".join(f"- {a}" for a in answers) or "(нет)"
    hist = "\n".join(f"{'Клиент' if t['speaker']=='customer' else 'Оператор'}: {t['text']}"
                     for t in transcript)
    return (f"## Документ\n{doc_text}\n\n## Готовые ответы\n{ans}\n\n## Диалог\n{hist}\n\n"
            "Дай ОДНУ короткую реплику оператора.")


def clean(raw):
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return (raw.split("\n\n")[0]).strip().strip('"«»').strip()


def gen(g, sysp, user):
    res = g._llm.create_chat_completion(
        messages=[{"role": "system", "content": sysp}, {"role": "user", "content": user}],
        max_tokens=220, temperature=0.4, top_p=0.9, stop=["\n\n\n", "<|im_end|>"])
    return clean(res["choices"][0]["message"]["content"])


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

    g = get_generator(); g._ensure_loaded()
    out = []
    for i, d in enumerate(json.loads((R / "compare_data.json").read_text(encoding="utf-8"))):
        gd = [x for x in gold.get((d["dialogue_id"], d["turn_idx"]), []) if x in dtext]
        if not gd:
            continue
        doc = "\n---\n".join(dtext[x] for x in gd)
        ans = [a for x in gd for a in answers.get(x, [])]
        user = build_user(doc, ans, d["transcript"])
        out.append({"dialogue_id": d["dialogue_id"], "turn_idx": d["turn_idx"],
                    "asr_text": d["asr_text"], "ideal_text": d["ideal_text"],
                    "with_emotion": gen(g, SYS_WITH, user), "without_emotion": gen(g, SYS_WITHOUT, user)})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}", flush=True)
    be = os.environ.get("LLM_BACKEND", "mistral24")
    (R / f"emo_ablation_{be}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(out)} -> emo_ablation_{be}.json")


if __name__ == "__main__":
    main()
