"""End-to-end оценка ВСЕГО решения (RAG+LLM) в разных конфигурациях (абляции)."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import app  # noqa: F401,E402

from app.models.schemas import Emotion, EmotionState, KBSource  # noqa: E402

R = PROJECT_ROOT / "eval" / "results"
CASES = PROJECT_ROOT / "eval" / "e2e_cases.json"

EMPATHY = ["извин", "прощени", "понима", "сожале", "к сожалению", "неприят",
           "сочувств", "приношу", "жаль", "слышу вас"]
ACTION = ["компенс", "приоритет", "провер", "оформ", "верн", "перезагру",
          "техник", "сейчас", "сразу", "подключ", "настро", "заблокир"]
EMO_PRESETS = {
    "angry": EmotionState(label=Emotion.ANGRY, confidence=0.82, arousal=0.85,
                          valence=-0.7, escalation_risk=True),
    "sad": EmotionState(label=Emotion.SAD, confidence=0.76, arousal=0.30,
                        valence=-0.55, escalation_risk=False),
    "neutral": EmotionState(label=Emotion.NEUTRAL, confidence=0.92, arousal=0.30,
                            valence=0.0, escalation_risk=False),
}


def _norm(t: str) -> set[str]:
    t = t.lower().replace("ё", "е")
    return {w for w in re.sub(r"[^\wа-я ]+", " ", t).split() if len(w) > 3}


def grounded_score(suggestion: str, kb_text: str) -> float:
    """Прокси faithfulness: доля содержательных слов подсказки, есть в KB."""
    sug = _norm(suggestion)
    if not sug:
        return 0.0
    kb = _norm(kb_text)
    return round(len(sug & kb) / len(sug), 3)


def has(markers: list[str], text: str) -> bool:
    t = text.lower()
    return any(m in t for m in markers)


def fake_numbers(suggestion: str, kb_text: str) -> bool:
    """True если в подсказке есть число (цена/срок), которого НЕТ в KB → галлюцинация."""
    sug_nums = set(re.findall(r"\d+", suggestion))
    kb_nums = set(re.findall(r"\d+", kb_text))
    invented = sug_nums - kb_nums
    return any(int(n) > 5 for n in invented)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    choices=["full", "no_rag", "no_emotion", "naive"])
    args = ap.parse_args()

    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    from app.pipeline.rag import get_retriever
    retr = get_retriever()

    use_llm = args.config != "naive"
    if use_llm:
        from app.pipeline.llm import get_generator
        gen = get_generator()
        gen._ensure_loaded()

    rows = []
    agg = {"grounded": 0.0, "format_ok": 0, "action": 0, "empathy": 0,
           "no_fake_num": 0, "len": 0, "ms": 0.0}
    neg_n = 0
    for c in cases:
        emo = EMO_PRESETS[c["emotion"]]
        if args.config == "no_rag":
            sources: list[KBSource] = []
            kb_text = ""
        else:
            res = retr.search(c["text"], k=3, company=c["company"])
            sources = res.sources
            kb_text = " ".join(s.snippet for s in sources)

        t0 = time.perf_counter()
        if args.config == "naive":
            top = sources[0].snippet if sources else "(нет данных)"
            suggestions = [top]
        else:
            use_emotion = args.config != "no_emotion"
            g = gen.generate([{"speaker": "customer", "text": c["text"]}], emo,
                             sources, max_tokens=220, use_emotion=use_emotion)
            suggestions = g.suggestions
        ms = (time.perf_counter() - t0) * 1000

        joined = " ".join(suggestions)
        m = {
            "grounded": grounded_score(joined, kb_text) if kb_text else 0.0,
            "format_ok": int(len(suggestions) >= 2) if use_llm else 1,
            "action": int(has(ACTION, joined)),
            "empathy": int(has(EMPATHY, joined)),
            "no_fake_num": int(not fake_numbers(joined, kb_text)),
            "len": len(joined),
            "ms": int(ms),
        }
        rows.append({**c, "suggestions": suggestions,
                     "sources": [s.doc_id for s in sources], "metrics": m})
        for k in ("grounded", "format_ok", "action", "no_fake_num", "len", "ms"):
            agg[k] += m[k]
        if c["emotion"] in ("angry", "sad"):
            agg["empathy"] += m["empathy"]
            neg_n += 1

    n = len(cases)
    summary = {
        "config": args.config,
        "grounded": round(agg["grounded"] / n, 3),
        "format_ok": round(agg["format_ok"] / n, 3),
        "action_rate": round(agg["action"] / n, 3),
        "empathy_rate_neg": round(agg["empathy"] / max(neg_n, 1), 3),
        "no_fake_num_rate": round(agg["no_fake_num"] / n, 3),
        "avg_len": int(agg["len"] / n),
        "p_mean_ms": int(agg["ms"] / n),
        "n": n,
    }
    print(f"\n=== E2E [{args.config}] (n={n}) ===")
    for k, v in summary.items():
        if k != "config":
            print(f"  {k}: {v}")

    out = R / f"e2e_{args.config}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {out}")

    jr = R / "e2e_summary.json"
    runs = []
    if jr.exists():
        try:
            runs = json.loads(jr.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            runs = []
    runs = [r for r in runs if r["config"] != args.config] + [summary]
    jr.write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
