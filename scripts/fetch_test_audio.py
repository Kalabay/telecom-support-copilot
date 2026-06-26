"""Получает тестовые аудио для eval SER."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import app  # noqa: F401,E402

if "TORCH_HOME" not in os.environ:
    os.environ["TORCH_HOME"] = str(PROJECT_ROOT / ".caches" / "torch")

import soundfile as sf  # noqa: E402

AUDIO_DIR = PROJECT_ROOT / "data" / "audio_samples"
MANIFEST = AUDIO_DIR / "manifest.json"

DEMO_PHRASES = [
    ("demo_01_greeting.wav",     "neutral", "Добрый день. У меня третий день не работает интернет, разберитесь, пожалуйста."),
    ("demo_02_diagnostic.wav",   "neutral", "Я уже два раза перезагружал роутер, индикаторы все горят, но интернета нет."),
    ("demo_03_complaint.wav",    "angry",   "Я плачу вам каждый месяц, а услугу не получаю! Сколько можно?"),
    ("demo_04_threat.wav",       "angry",   "Если сегодня не починят, я расторгну договор и уйду к Билайну!"),
]


def synth_with_silero() -> None:
    """Скачивает Silero TTS и генерирует демо-аудио."""
    import torch

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Silero TTS (ru, v4)…")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language="ru",
        speaker="v4_ru",
        trust_repo=True,
    )
    model.to(device)

    manifest: list[dict] = []
    speakers = ["aidar", "baya", "kseniya", "xenia"]
    for (fname, label, text), spk in zip(DEMO_PHRASES, speakers):
        print(f"  synth -> {fname}  ({label})  speaker={spk}")
        audio = model.apply_tts(text=text, speaker=spk, sample_rate=24000, put_accent=True, put_yo=True)
        sf.write(AUDIO_DIR / fname, audio.cpu().numpy(), 24000, subtype="PCM_16")
        manifest.append(
            {
                "file": fname,
                "label": label,
                "text": text,
                "source": "silero-tts",
                "speaker": spk,
                "note": "TTS-generated; label is the *intended* emotion (TTS не передаёт affect)",
            }
        )

    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSaved {len(manifest)} clips -> {AUDIO_DIR}")
    print(f"Manifest -> {MANIFEST}")


def main() -> None:
    synth_with_silero()


if __name__ == "__main__":
    main()
