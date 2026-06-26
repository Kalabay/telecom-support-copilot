"""Сравнение ASR-движков по WER на наших данных. Движок из ASR_ENGINE (whisper|gigaam)."""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
R = PROJECT_ROOT / "eval" / "results"
AUDIO_E2E = PROJECT_ROOT / "eval" / "audio" / "e2e"
ENGINE = os.environ.get("ASR_ENGINE", "whisper")
SCRATCH = Path(os.environ.get("TMP", "/tmp")) / "asr_cmp"
SCRATCH.mkdir(parents=True, exist_ok=True)


def norm(t: str) -> str:
    t = t.lower().replace("ё", "е")
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


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


def load16k(path: Path) -> np.ndarray:
    wav, _ = librosa.load(str(path), sr=16000, mono=True)
    return wav


def main() -> None:
    man = json.loads((AUDIO_E2E / "manifest.json").read_text(encoding="utf-8"))
    rows = [m for m in man if m.get("ok") and (AUDIO_E2E / m["file"]).exists() and m.get("clean_text")]
    print(f"{ENGINE}: {len(rows)} файлов синтетики", flush=True)

    if ENGINE == "whisper":
        from faster_whisper import WhisperModel
        model = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8_float16")

        def transcribe(path: Path) -> str:
            segs, _ = model.transcribe(load16k(path), language="ru", beam_size=5)
            return " ".join(s.text for s in segs).strip()
    elif ENGINE == "gigaam":
        import gigaam
        model = gigaam.load_model("v2_rnnt")

        def transcribe(path: Path) -> str:
            wav = load16k(path)
            tmp = SCRATCH / "cur.wav"
            sf.write(str(tmp), wav, 16000)
            return model.transcribe(str(tmp)).strip()
    else:
        raise SystemExit(f"unknown engine {ENGINE}")

    tot_err = tot_ref = 0
    items = []
    for i, m in enumerate(rows):
        try:
            hyp = transcribe(AUDIO_E2E / m["file"])
        except Exception as exc:  # noqa: BLE001
            print(f"  err {m['file']}: {exc}", flush=True); continue
        e, n = wer(m["clean_text"], hyp)
        tot_err += e; tot_ref += n
        items.append({"file": m["file"], "ref": m["clean_text"], "hyp": hyp})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(rows)}  WER={tot_err/max(1,tot_ref):.4f}", flush=True)

    out = {"engine": ENGINE, "set": "synth", "n": len(items),
           "wer": round(tot_err / max(1, tot_ref), 4), "items": items}
    (R / f"asr_cmp_{ENGINE}_synth.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{ENGINE} synth WER = {out['wer']}  (n={out['n']}) -> asr_cmp_{ENGINE}_synth.json")


if __name__ == "__main__":
    main()
