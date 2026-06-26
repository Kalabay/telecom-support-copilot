"""Проверка, что реплики оператора попадают в промпт LLM с метками говорящего."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import app  # noqa: F401,E402

from app.models.schemas import Emotion, EmotionState, KBSource  # noqa: E402
from app.pipeline.llm import LLMGenerator  # noqa: E402


def main() -> None:
    g = LLMGenerator.__new__(LLMGenerator)
    turns = [
        {"speaker": "customer", "text": "Не работает интернет третий день"},
        {"speaker": "operator", "text": "Давайте перезагрузим роутер — выньте питание на 30 секунд"},
        {"speaker": "customer", "text": "Я же только что это сделал, не помогло!"},
    ]
    emo = EmotionState(
        label=Emotion.ANGRY, confidence=0.8, arousal=0.85, valence=-0.7, escalation_risk=True
    )
    src = [KBSource(doc_id="router_reboot", title="Перезагрузка роутера", snippet="...", score=0.8)]
    prompt = g._build_user_prompt(turns, emo, src)
    print(prompt)
    print("\n--- checks ---")
    print("operator turn in prompt:", "Оператор:" in prompt)
    print("customer turns in prompt:", "Клиент:" in prompt)
    print("last customer is the complaint:", "только что это сделал" in prompt)


if __name__ == "__main__":
    main()
