"""Быстрый тест: грузится ли medium-ASR, его latency и качество на 5 Dusha-сэмплах."""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import app  # noqa: F401,E402

from datasets import Audio, load_dataset  # noqa: E402

from app.pipeline.asr import MODEL_SIZE, get_asr  # noqa: E402
from app.pipeline.ser import get_recognizer  # noqa: E402

print(f"MODEL_SIZE = {MODEL_SIZE}")
asr = get_asr()
t0 = time.perf_counter()
asr._ensure_loaded()  # noqa: SLF001
print(f"loaded in {time.perf_counter()-t0:.1f}s")

rec = get_recognizer()
ds = load_dataset("xbgoose/dusha", split="test", streaming=True).cast_column(
    "audio", Audio(decode=False))
n = 0
for item in ds:
    if n >= 6:
        break
    raw = item["audio"].get("bytes")
    if not raw:
        continue
    wav, _ = rec.load_audio(raw)
    dur = wav.size / 16000
    if dur < 0.5:
        continue
    r = asr.transcribe(wav, "ru")
    rtf = (r.inference_ms / 1000) / dur
    print(f"\n[{n}] emotion={item['emotion']}  dur={dur:.1f}s  asr={r.inference_ms}ms  RTF={rtf:.3f}")
    print(f"    «{r.text}»")
    n += 1
