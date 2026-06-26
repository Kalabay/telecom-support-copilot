"""Прогрев SER-модели: скачивает HuBERT-Dusha в HF_HOME и делает один тестовый инференс на синусоиде."""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import numpy as np  # noqa: E402

from app.pipeline.ser import TARGET_SR, get_recognizer  # noqa: E402


def main() -> None:
    print("=== SER warmup ===")
    rec = get_recognizer()

    duration = 3.0
    t = np.linspace(0, duration, int(TARGET_SR * duration), endpoint=False)
    wav = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    print("Running inference on a 3-sec test tone…")
    t0 = time.perf_counter()
    result = rec.predict(wav)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    print(f"\nResult:")
    print(f"  label        : {result.state.label.value}")
    print(f"  confidence   : {result.state.confidence}")
    print(f"  arousal      : {result.state.arousal}")
    print(f"  valence      : {result.state.valence}")
    print(f"  escalation   : {result.state.escalation_risk}")
    print(f"  probs        : {result.probs}")
    print(f"  duration_ms  : {result.duration_ms}")
    print(f"  inference_ms : {result.inference_ms}")
    print(f"  total elapsed: {elapsed_ms} ms")


if __name__ == "__main__":
    main()
