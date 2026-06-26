"""Best-of-3 для одной модели: сохраняет 3 подсказки продукта (что видит оператор) на реплику."""
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
    from app.pipeline.llm import get_generator
    gen = get_generator(); gen._ensure_loaded()
    print(f"модель '{BACKEND}' загружена", flush=True)

    emo = EmotionState(label=Emotion.ANGRY, confidence=0.85, arousal=0.85,
                       valence=-0.7, escalation_risk=True)
    rows = json.loads((R / "compare_data.json").read_text(encoding="utf-8"))
    out, few = [], 0
    for i, d in enumerate(rows):
        src = [KBSource(doc_id=s.get("doc_id", "kb"), title=s["title"],
                        snippet=s["snippet"], score=1.0) for s in d["sources"]]
        gres = gen.generate(d["transcript"], emo, src, max_tokens=200, safe=False)
        variants = [s for s in gres.suggestions if s and s.strip()] or [""]
        if len(variants) < 3:
            few += 1
        v = (variants * 3)[:3]
        out.append({"dialogue_id": d["dialogue_id"], "turn_idx": d["turn_idx"],
                    "dataset": d["dataset"], "asr_text": d["asr_text"],
                    "ideal_text": d["ideal_text"], "n_variants": len(variants),
                    "v1": v[0], "v2": v[1], "v3": v[2], "ms": gres.total_ms})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    dest = R / f"bo3_{BACKEND}.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nготово: {len(out)} реплик -> {dest.name} (с <3 вариантами: {few})")


if __name__ == "__main__":
    main()
