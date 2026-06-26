"""Прогон одной LLM-модели на 87 злых репликах (RAG-источники взяты из compare_data.json)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import app  # noqa: F401,E402
import sentence_transformers  # noqa: F401,E402
import torch  # noqa: F401,E402

from app.models.schemas import Emotion, EmotionState, KBSource  # noqa: E402

BACKEND = os.environ.get("LLM_BACKEND", "tlite21")
R = PROJECT_ROOT / "eval" / "results"


def main() -> None:
    from app.pipeline.llm import get_generator, _safety_rank
    gen = get_generator(); gen._ensure_loaded()
    print(f"модель '{BACKEND}' загружена", flush=True)

    emo = EmotionState(label=Emotion.ANGRY, confidence=0.85, arousal=0.85,
                       valence=-0.7, escalation_risk=True)
    inp = os.environ.get("BENCH_INPUT", "compare_data.json")
    rows = json.loads((R / inp).read_text(encoding="utf-8"))
    out = []
    for i, d in enumerate(rows):
        src = [KBSource(doc_id=s.get("doc_id", "kb"), title=s["title"],
                        snippet=s["snippet"], score=1.0) for s in d["sources"]]
        gres = gen.generate(d["transcript"], emo, src, max_tokens=200, safe=False)
        variants = gres.suggestions
        raw = variants[0] if variants else ""
        filt = _safety_rank(variants, src, emo)[0] if variants else ""
        out.append({"dialogue_id": d["dialogue_id"], "turn_idx": d["turn_idx"],
                    "dataset": d["dataset"], "asr_text": d["asr_text"],
                    "ideal_text": d["ideal_text"], "raw": raw, "filtered": filt,
                    "ms": gres.total_ms, "completion_tokens": gres.completion_tokens})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    dest = R / f"bench_{BACKEND}.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nготово: {len(out)} реплик -> {dest.name}")


if __name__ == "__main__":
    main()
