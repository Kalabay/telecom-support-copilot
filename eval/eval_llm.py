"""LLM A/B: прямой прогон генератора (без сервера) на тест-репликах."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import app  # noqa: F401,E402

from app.core.config import settings  # noqa: E402
from app.models.schemas import Emotion, EmotionState, KBSource  # noqa: E402

R = PROJECT_ROOT / "eval" / "results"
EMPATHY = ["извин", "прощени", "понимаю", "сожале", "к сожалению", "неприят",
           "сочувств", "приношу", "жаль"]
ACTION = ["компенс", "приоритет", "проверю", "проверим", "оформлю", "оформим",
          "верн", "перезагруз", "техник", "сейчас", "сразу"]

TESTSET = [
    ("angry", "Я плачу каждый месяц, а интернета нет третий день!"),
    ("angry", "Сколько можно?! Это безобразие, я в бешенстве!"),
    ("angry", "Если не почините сегодня — расторгаю договор!"),
    ("angry", "Звоню четвёртый раз, никто не помогает, верните деньги!"),
    ("sad", "Так обидно, я с вами десять лет, а тут такие проблемы..."),
    ("sad", "Дома ребёнок на удалёнке, а интернета нет, я в отчаянии."),
    ("sad", "Я уже не знаю что делать, ничего не помогает."),
    ("neutral", "Здравствуйте, не работает интернет, подскажите что делать."),
    ("neutral", "На роутере горит красная лампочка, это нормально?"),
    ("neutral", "Уточните, нет ли в моём районе плановых работ."),
    ("neutral", "Как настроить интернет на новом роутере?"),
    ("neutral", "Можно перенести интернет при переезде?"),
]
EMO = {
    "angry": EmotionState(label=Emotion.ANGRY, confidence=0.82, arousal=0.85,
                          valence=-0.7, escalation_risk=True),
    "sad": EmotionState(label=Emotion.SAD, confidence=0.76, arousal=0.30,
                        valence=-0.55, escalation_risk=False),
    "neutral": EmotionState(label=Emotion.NEUTRAL, confidence=0.92, arousal=0.30,
                            valence=0.0, escalation_risk=False),
}
SRC = [KBSource(doc_id="diagnose_general", title="Общая диагностика",
                snippet="Проверьте индикаторы роутера, перезагрузите устройство, "
                        "при аварии в районе сообщите сроки.", score=0.9)]


def _has(markers, text):
    t = text.lower()
    return any(m in t for m in markers)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=settings.llm_backend)
    args = ap.parse_args()

    from app.pipeline.llm import LLM_GGUF_FILE, LLM_REPO_ID, get_generator
    print(f"LLM backend='{settings.llm_backend}'  repo={LLM_REPO_ID}\n  gguf={LLM_GGUF_FILE}", flush=True)
    gen = get_generator()
    gen._ensure_loaded()

    rows, lats, toks = [], [], []
    fmt_ok = emp = act = 0
    total_len = 0
    for emo_label, text in TESTSET:
        t0 = time.perf_counter()
        res = gen.generate([{"speaker": "customer", "text": text}], EMO[emo_label],
                           SRC, max_tokens=220)
        ms = (time.perf_counter() - t0) * 1000
        lats.append(ms)
        sugg = res.suggestions
        if res.completion_tokens and ms > 0:
            toks.append(res.completion_tokens / (ms / 1000))
        ok = len(sugg) >= 2
        fmt_ok += ok
        joined = " ".join(sugg)
        emp += _has(EMPATHY, joined)
        act += _has(ACTION, joined)
        total_len += len(joined)
        rows.append({"emotion": emo_label, "text": text, "suggestions": sugg,
                     "format_ok": ok, "ms": int(ms)})
        print(f"  [{emo_label}] ok={ok} n={len(sugg)} :: {sugg[0][:60] if sugg else '—'}", flush=True)

    n = len(TESTSET)
    import numpy as np
    res = {
        "tag": args.tag,
        "backend": settings.llm_backend,
        "format_ok_rate": round(fmt_ok / n, 3),
        "empathy_rate": round(emp / n, 3),
        "action_rate": round(act / n, 3),
        "avg_len": int(total_len / n),
        "tok_per_s": round(float(np.mean(toks)) if toks else 0, 1),
        "p95_ms": round(float(np.percentile(lats, 95)), 0),
        "n": n,
    }
    print("\n=== LLM eval ===")
    for k, v in res.items():
        print(f"  {k}: {v}")

    out = R / "llm_eval.json"
    runs = []
    if out.exists():
        try:
            runs = json.loads(out.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            runs = []
    runs.append({**res, "rows": rows})
    out.write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
