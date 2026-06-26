"""ASR-транскрипция локальных train-wav (для текстового канала fusion)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import app  # noqa: F401,E402

import soundfile as sf  # noqa: E402

CLASSES = {"neutral", "angry", "positive", "sad"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--manifest", type=Path, default=Path(r"K:\.caches\dusha_train\manifest.jsonl"))
    ap.add_argument("--out", type=Path, default=Path(r"K:\.caches\dusha_train\transcripts.jsonl"))
    args = ap.parse_args()
    MANIFEST = args.manifest

    from app.pipeline.asr import get_asr
    from app.pipeline.ser import get_recognizer
    asr = get_asr(); asr._ensure_loaded()
    rec = get_recognizer()

    rows = [json.loads(l) for l in MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    done = set()
    if args.out.exists():
        for l in args.out.read_text(encoding="utf-8").splitlines():
            if l.strip():
                done.add(json.loads(l)["id"])
        print(f"уже есть {len(done)}, докачиваю", flush=True)

    written, t0 = 0, time.perf_counter()
    with args.out.open("a", encoding="utf-8") as f:
        for r in rows:
            if written + len(done) >= args.n:
                break
            if r["id"] in done or r["emotion"].lower() not in CLASSES:
                continue
            try:
                data, sr = sf.read(r["path"], dtype="float32")
                if data.ndim > 1:
                    data = data.mean(axis=1)
                if sr != 16000:
                    import librosa
                    data = librosa.resample(data, orig_sr=sr, target_sr=16000)
                text = asr.transcribe(data.astype("float32"), "ru").text.strip()
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {r['id']}: {exc}", flush=True)
                continue
            f.write(json.dumps({"id": r["id"], "text": text, "emotion": r["emotion"].lower()},
                                ensure_ascii=False) + "\n")
            f.flush()
            written += 1
            if written % 500 == 0:
                print(f"  {written}  {time.perf_counter()-t0:.0f}s", flush=True)
    print(f"готово: +{written} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
