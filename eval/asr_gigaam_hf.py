"""WER GigaAM-v2-CTC (HF-обёртка) на Dusha и синтетике для сравнения с Whisper."""
from __future__ import annotations

import io
import json
import os
import re
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
R = PROJECT_ROOT / "eval" / "results"
AUDIO_E2E = PROJECT_ROOT / "eval" / "audio" / "e2e"
DSET = os.environ.get("ASR_SET", "synth")
N = int(os.environ.get("ASR_N", "300"))
NAME = "waveletdeboshir/gigaam-ctc"


def norm(t):
    t = (t or "").lower().replace("ё", "е")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t)).strip()


def wer(ref, hyp):
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
    for m in json.loads((AUDIO_E2E / "manifest.json").read_text(encoding="utf-8")):
        if m.get("ok") and (AUDIO_E2E / m["file"]).exists() and m.get("clean_text"):
            wav, _ = librosa.load(str(AUDIO_E2E / m["file"]), sr=16000, mono=True)
            yield wav, m["clean_text"]


def dusha_items():
    from datasets import Audio, load_dataset
    txt = [json.loads(x) for x in (PROJECT_ROOT / "data" / "dusha_text" / "test.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    ds = load_dataset("xbgoose/dusha", split="test", streaming=True).cast_column("audio", Audio(decode=False))
    for i, it in enumerate(ds):
        if i >= min(N, len(txt)):
            break
        ref = txt[i].get("text", "")
        raw = (it.get("audio") or {}).get("bytes")
        if ref and raw:
            wav, _ = librosa.load(io.BytesIO(raw), sr=16000, mono=True)
            yield wav, ref


def main():
    from transformers import AutoModelForCTC, AutoProcessor
    proc = AutoProcessor.from_pretrained(NAME, trust_remote_code=True)
    model = AutoModelForCTC.from_pretrained(NAME, trust_remote_code=True).eval()
    print("GigaAM-CTC загружен", flush=True)

    def transcribe(wav):
        inp = proc(wav, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inp).logits
        ids = torch.argmax(logits, dim=-1)
        return proc.batch_decode(ids)[0]

    items = dusha_items() if DSET == "dusha" else synth_items()
    te = tr = n = 0
    for wav, ref in items:
        try:
            hyp = transcribe(wav)
        except Exception as exc:
            print("err:", exc, flush=True); continue
        e, r = wer(ref, hyp); te += e; tr += r; n += 1
        if n % 100 == 0:
            print(f"  {n}  WER={te/max(1,tr):.4f}", flush=True)
    out = {"model": "gigaam-ctc", "set": DSET, "n": n, "wer": round(te / max(1, tr), 4)}
    (R / f"asr_gigaam_{DSET}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ngigaam-ctc / {DSET}: WER = {out['wer']}  (n={n})")


if __name__ == "__main__":
    main()
