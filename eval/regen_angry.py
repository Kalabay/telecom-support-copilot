"""Быстрая перегенерация подсказок ТОЛЬКО для злых реплик с новым промптом."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import app  # noqa: F401,E402
import sentence_transformers  # noqa: F401,E402
import torch  # noqa: F401,E402

from app.models.schemas import Emotion, EmotionState  # noqa: E402

DATASETS = [
    ("real", PROJECT_ROOT / "eval" / "e2e_dialogues_real.json",
     PROJECT_ROOT / "eval" / "results" / "e2e_real_result.json"),
    ("vektor", PROJECT_ROOT / "eval" / "e2e_dialogues.json",
     PROJECT_ROOT / "eval" / "results" / "e2e_dialogues_result.json"),
]
OUT = PROJECT_ROOT / "eval" / "results" / "angry_regen.json"


def main() -> None:
    from app.pipeline.rag import get_retriever
    from app.pipeline.llm import get_generator
    rag = get_retriever()
    gen = get_generator(); gen._ensure_loaded()
    print("RAG + LLM загружены", flush=True)

    emo = EmotionState(label=Emotion.ANGRY, confidence=0.85, arousal=0.85,
                       valence=-0.7, escalation_risk=True)
    out = []
    for tag, dlg_path, res_path in DATASETS:
        dialogues = json.loads(dlg_path.read_text(encoding="utf-8"))
        res = json.loads(res_path.read_text(encoding="utf-8"))
        asr = {(r["dialogue_id"], r["turn_idx"]): r.get("asr_text", "")
               for r in res["turns"]}
        for d in dialogues:
            did, company = d["dialogue_id"], d["company"]
            turns = d["turns"]
            transcript = []
            for t in turns:
                if t["role"] == "client":
                    txt = asr.get((did, t["idx"])) or t["text"]
                    transcript.append({"speaker": "customer", "text": txt})
                    if t.get("emotion") == "angry":
                        prev = next((x["text"] for x in reversed(transcript[:-1])
                                     if x["speaker"] == "customer"), "")
                        query = f"{prev} {txt}".strip()
                        sources = rag.search(query, k=3, company=company).sources
                        sug = gen.generate(list(transcript), emo, sources,
                                           max_tokens=200).suggestions
                        out.append({
                            "dataset": tag, "dialogue_id": did, "turn_idx": t["idx"],
                            "company": company, "emotion": "angry",
                            "asr_text": txt,
                            "ideal_text": next((tt["ideal_text"] for tt in turns
                                                if tt["idx"] == t["idx"] + 1
                                                and tt["role"] == "operator"), ""),
                            "suggestion": sug[0] if sug else "",
                        })
                else:
                    transcript.append({"speaker": "operator", "text": t["ideal_text"]})
        print(f"  {tag}: всего злых обработано -> {len(out)}", flush=True)

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nзлых реплик перегенерировано: {len(out)}\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
