"""Способ C: быстрый LLM-верификатор (MoE) проверяет ответ плейбука против документа."""
from __future__ import annotations

import json
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
SYS = ("Ты — контролёр качества. Тебе дают документ базы знаний и ответ оператора клиенту. "
       "Твоя задача — проверить, не утверждает ли ответ ничего, чего нет в документе.\n/no_think")


def main() -> None:
    from app.pipeline.llm import get_generator
    pb = json.loads((R / "bench_playbook.json").read_text(encoding="utf-8"))
    moe = {(r["dialogue_id"], r["turn_idx"]): r["raw"]
           for r in json.loads((R / "bench_qwen3moe.json").read_text(encoding="utf-8"))}
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
        if len(dtext[m["doc_id"]]) < 1400:
            dtext[m["doc_id"]] = (dtext[m["doc_id"]] + " " + txt).strip()

    gen = get_generator(); gen._ensure_loaded()
    out = []; swapped = 0
    for i, r in enumerate(pb):
        k = (r["dialogue_id"], r["turn_idx"])
        kb = "\n".join(dtext[g] for g in gold.get(k, []))
        user = (f"## Документ\n{kb[:1400]}\n\n## Ответ оператора\n«{r['raw']}»\n\n"
                "Есть ли в ответе выдуманные факты, причины, сроки, суммы, статусы «уже сделал» "
                "или утверждения не по теме, которых НЕТ в документе? Ответь ОДНИМ словом: "
                "ДА (есть выдумка) или НЕТ (всё по документу).")
        res = gen._llm.create_chat_completion(
            messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}],
            max_tokens=8, temperature=0.0, top_p=0.9, stop=["<|im_end|>", "\n\n"])
        ans = re.sub(r"<think>.*?</think>", "", res["choices"][0]["message"]["content"], flags=re.DOTALL)
        flagged = "да" in ans.lower()
        if flagged and k in moe:
            use = moe[k]; swapped += 1
        else:
            use = r["raw"]
        out.append({**{x: r[x] for x in ("dialogue_id", "turn_idx", "asr_text", "ideal_text")}, "raw": use})
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(pb)}", flush=True)
    (R / "playbook_fixC.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"C (LLM-верификатор+MoE): подменено {swapped}/{len(out)} -> playbook_fixC.json")


if __name__ == "__main__":
    main()
