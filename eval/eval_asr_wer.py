"""ASR A/B по WER на FLEURS ru_ru (золотые транскрипты)."""

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

import numpy as np  # noqa: E402

CACHE = PROJECT_ROOT / ".hf_cache"
R = PROJECT_ROOT / "eval" / "results"


def normalize(t: str) -> list[str]:
    t = t.lower().replace("ё", "е")
    t = re.sub(r"[^\wа-я ]+", " ", t)
    return [w for w in t.split() if w]


def wer(ref: list[str], hyp: list[str]) -> tuple[int, int]:
    """Levenshtein по словам → (errors, ref_len)."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return m, 0
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m], n


def resolve_model(name: str) -> str:
    """medium -> локальный путь .hf_cache/whisper-medium; turbo -> имя для HF."""
    if name == "medium":
        p = CACHE / "whisper-medium"
        if (p / "model.bin").exists():
            return str(p)
        return "medium"
    return name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="medium | large-v3-turbo | large-v3")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", type=Path, default=R / "asr_wer.json")
    args = ap.parse_args()

    import librosa
    from datasets import Audio, load_dataset
    from faster_whisper import WhisperModel

    print("loading FLEURS ru_ru test (stream)…", flush=True)
    ds = load_dataset("google/fleurs", "ru_ru", split="test", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    mpath = resolve_model(args.model)
    print(f"loading faster-whisper '{args.model}' ({mpath})…", flush=True)
    model = WhisperModel(mpath, device="cuda", compute_type="int8_float16")

    tot_err = tot_ref = 0
    audio_sec = infer_sec = 0.0
    n = 0
    t0 = time.perf_counter()
    for item in ds:
        if n >= args.n:
            break
        ref_text = item.get("transcription") or item.get("sentence", "")
        raw = item["audio"].get("bytes")
        if not ref_text or not raw:
            continue
        try:
            import io
            wav, sr = librosa.load(io.BytesIO(raw), sr=16000, mono=True)
        except Exception:  # noqa: BLE001
            continue
        dur = len(wav) / 16000
        ti = time.perf_counter()
        segs, _ = model.transcribe(wav, language="ru", beam_size=5,
                                   condition_on_previous_text=False)
        hyp = " ".join(s.text.strip() for s in segs)
        infer_sec += time.perf_counter() - ti
        audio_sec += dur
        e, r = wer(normalize(ref_text), normalize(hyp))
        tot_err += e
        tot_ref += r
        n += 1
        if n % 20 == 0:
            print(f"  {n}/{args.n}  WER={tot_err/max(tot_ref,1):.3f}  "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)

    res = {
        "model": args.model,
        "wer": round(tot_err / max(tot_ref, 1), 4),
        "rtf": round(infer_sec / max(audio_sec, 1e-9), 4),
        "n": n,
        "audio_sec": round(audio_sec, 1),
    }
    print("\n=== ASR WER eval ===")
    for k, v in res.items():
        print(f"  {k}: {v}")

    runs = []
    if args.out.exists():
        try:
            runs = json.loads(args.out.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            runs = []
    runs.append(res)
    args.out.write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
