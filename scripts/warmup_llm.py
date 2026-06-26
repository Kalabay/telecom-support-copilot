"""Прогрев T-lite GGUF: скачивает модель в HF_HOME и делает один тестовый промпт."""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import app  # noqa: F401,E402  (запускает .env-bootstrap для HF_HOME)

from app.models.schemas import Emotion, EmotionState, KBSource  # noqa: E402
from app.pipeline.llm import get_generator  # noqa: E402


def main() -> None:
    print("=== T-lite warmup ===")
    gen = get_generator()

    transcript = [
        "Алло, добрый день. У меня уже третий день не работает интернет.",
    ]
    emotion = EmotionState(
        label=Emotion.NEUTRAL,
        confidence=0.92,
        arousal=0.30,
        valence=-0.20,
        escalation_risk=False,
    )
    sources = [
        KBSource(
            doc_id="diagnose_general",
            title="Общая диагностика отсутствия интернета",
            snippet=(
                "Уточнить какое подключение не работает (домашний/мобильный). "
                "Проверить адрес на наличие массовой аварии. Базовая диагностика: "
                "перезагрузка роутера, проверка индикаторов, проверка кабеля."
            ),
            score=0.83,
        ),
    ]

    print("Generating (first call also loads + GPU-offloads the model)…")
    t0 = time.perf_counter()
    result = gen.generate(transcript, emotion, sources)
    elapsed = time.perf_counter() - t0

    print(f"\n--- raw completion ---\n{result.raw_completion}\n")
    print(f"--- parsed suggestions ({len(result.suggestions)}) ---")
    for i, s in enumerate(result.suggestions, 1):
        print(f"  {i}. {s}")

    print(f"\nprompt_tokens     : {result.prompt_tokens}")
    print(f"completion_tokens : {result.completion_tokens}")
    print(f"llm_total_ms      : {result.total_ms}")
    print(f"wall elapsed      : {elapsed:.1f}s")


if __name__ == "__main__":
    main()
