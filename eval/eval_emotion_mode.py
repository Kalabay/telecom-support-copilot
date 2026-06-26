"""A/B: emotion-в-промпт словарь (dict) vs cause-aware (cot). Генерит обе подсказки."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import app  # noqa: F401,E402

from app.models.schemas import Emotion, EmotionState, KBSource  # noqa: E402,F401

R = PROJECT_ROOT / "eval" / "results"
CASES = PROJECT_ROOT / "eval" / "e2e_cases.json"

EMO = {
    "angry": EmotionState(label=Emotion.ANGRY, confidence=0.82, arousal=0.80,
                          valence=-0.7, escalation_risk=True),
    "sad": EmotionState(label=Emotion.SAD, confidence=0.76, arousal=0.30,
                        valence=-0.55, escalation_risk=False),
    "neutral": EmotionState(label=Emotion.NEUTRAL, confidence=0.92, arousal=0.30,
                            valence=0.0, escalation_risk=False),
}


def main():
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    neg = [c for c in cases if c["emotion"] in ("angry", "sad")]
    pick = neg[::max(1, len(neg) // 24)][:24]

    from app.pipeline.rag import get_retriever
    from app.pipeline.llm import get_generator
    rag = get_retriever()
    gen = get_generator(); gen._ensure_loaded()

    out = []
    for i, c in enumerate(pick, 1):
        emo = EMO[c["emotion"]]
        src = rag.search(c["text"], k=3, company=c["company"]).sources
        d = gen.generate([{"speaker": "customer", "text": c["text"]}], emo, src,
                         max_tokens=200, use_emotion=True, emotion_mode="dict").suggestions
        t = gen.generate([{"speaker": "customer", "text": c["text"]}], emo, src,
                         max_tokens=200, use_emotion=True, emotion_mode="cot").suggestions
        out.append({"id": c["id"], "company": c["company"], "emotion": c["emotion"],
                    "text": c["text"], "dict": d, "cot": t})
        print(f"  {i}/{len(pick)} {c['id']}", flush=True)

    (R / "emotion_mode_pairs.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> emotion_mode_pairs.json ({len(out)})", flush=True)


if __name__ == "__main__":
    main()
