"""WER нескольких ASR на двух наборах: Dusha (реальная речь) и наша синтетика."""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import librosa
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
R = PROJECT_ROOT / "eval" / "results"
AUDIO_E2E = PROJECT_ROOT / "eval" / "audio" / "e2e"
MODEL = os.environ.get("ASR_MODEL", "large-v3-turbo")
DSET = os.environ.get("ASR_SET", "synth")
N = int(os.environ.get("ASR_N", "300"))


def norm(t: str) -> str:
    t = (t or "").lower().replace("ё", "е")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t)).strip()


def wer(ref: str, hyp: str) -> tuple[int, int]:
    r, h = norm(ref).split(), norm(hyp).split()
    n, m = len(r), len(h)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (0 if r[i - 1] == h[j - 1] else 1))
            prev = cur
    return dp[m], n


def synth_items():
    man = json.loads((AUDIO_E2E / "manifest.json").read_text(encoding="utf-8"))
    for m in man:
        if m.get("ok") and (AUDIO_E2E / m["file"]).exists() and m.get("clean_text"):
            wav, _ = librosa.load(str(AUDIO_E2E / m["file"]), sr=16000, mono=True)
            yield wav, m["clean_text"]


def dusha_items():
    import io
    from datasets import Audio, load_dataset
    tj = (PROJECT_ROOT / "data" / "dusha_text" / "test.jsonl").read_text(encoding="utf-8").splitlines()
    txt = [json.loads(x) for x in tj if x.strip()]
    ds = load_dataset("xbgoose/dusha", split="test", streaming=True).cast_column("audio", Audio(decode=False))
    mism = 0
    for i, it in enumerate(ds):
        if i >= min(N, len(txt)):
            break
        ref = txt[i].get("text", "")
        if it.get("emotion") != txt[i].get("emotion"):
            mism += 1
        raw = (it.get("audio") or {}).get("bytes")
        if ref and raw:
            wav, _ = librosa.load(io.BytesIO(raw), sr=16000, mono=True)
            yield wav, ref
    if mism:
        print(f"  emotion-рассинхрон: {mism} строк (выравнивание под вопросом)", flush=True)


def main() -> None:
    from faster_whisper import WhisperModel
    model = WhisperModel(MODEL, device="cuda", compute_type="int8_float16")
    items = dusha_items() if DSET == "dusha" else synth_items()
    print(f"{MODEL} / {DSET}: считаю...", flush=True)
    tot_err = tot_ref = n = 0
    for wav, ref in items:
        segs, _ = model.transcribe(wav, language="ru", beam_size=5)
        hyp = " ".join(s.text for s in segs)
        e, r = wer(ref, hyp)
        tot_err += e; tot_ref += r; n += 1
        if n % 100 == 0:
            print(f"  {n}  WER={tot_err/max(1,tot_ref):.4f}", flush=True)
    out = {"model": MODEL, "set": DSET, "n": n, "wer": round(tot_err / max(1, tot_ref), 4)}
    (R / f"asr_{MODEL.replace('-', '')}_{DSET}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{MODEL} / {DSET}: WER = {out['wer']}  (n={n})")


if __name__ == "__main__":
    main()
