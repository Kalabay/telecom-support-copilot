"""Шаг 1 улучшения RAG: LLM переформулирует реплику клиента в чистый поисковый запрос."""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import app  # noqa: F401,E402
import sentence_transformers  # noqa: F401,E402
import torch  # noqa: F401,E402

R = PROJECT_ROOT / "eval" / "results"
FILES = [("real", R / "e2e_real_result.json"), ("vektor", R / "e2e_dialogues_result.json")]

SYS = (
    "Ты помогаешь искать статью в базе знаний телеком-поддержки. По реплике клиента "
    "сформулируй КОРОТКИЙ поисковый запрос (3-8 слов) — суть проблемы или темы обращения, "
    "нормальными словами, без эмоций и лишнего. Только запрос, без пояснений и кавычек."
)


def main() -> None:
    from app.pipeline.llm import get_generator
    gen = get_generator(); gen._ensure_loaded()
    print("LLM загружен", flush=True)

    out = {}
    for tag, path in FILES:
        turns = [t for t in json.loads(path.read_text(encoding="utf-8"))["turns"]
                 if t.get("gold_doc_ids") and t.get("asr_text")]
        nothink = "\n/no_think" if os.environ.get("LLM_BACKEND", "").startswith("qwen") else ""
        for i, t in enumerate(turns):
            res = gen._llm.create_chat_completion(
                messages=[{"role": "system", "content": SYS + nothink},
                          {"role": "user", "content": f"Реплика клиента: «{t['asr_text']}»\n\nПоисковый запрос:"}],
                max_tokens=64, temperature=0.2, top_p=0.9,
                stop=["<|im_end|>", "<|eot_id|>", "<|message_sep|>"],
            )
            raw = re.sub(r"<think>.*?</think>", "", res["choices"][0]["message"]["content"], flags=re.DOTALL)
            raw = raw.replace("<think>", "").replace("</think>", "").strip()
            q = next((ln.strip().strip('"«».').strip() for ln in raw.splitlines() if ln.strip()), "")
            out[f"{t['company']}|{t['dialogue_id']}|{t['turn_idx']}"] = q
            if (i + 1) % 50 == 0:
                print(f"  {tag} {i+1}/{len(turns)}", flush=True)
        print(f"  {tag}: готово", flush=True)

    backend = os.environ.get("LLM_BACKEND", "tlite21")
    name = "reformulated_queries.json" if backend == "tlite21" else f"reformulated_{backend}.json"
    (R / name).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(out)} запросов -> {name}")
    for k in list(out)[:3]:
        print(f"  {k} -> {out[k]}")


if __name__ == "__main__":
    main()
