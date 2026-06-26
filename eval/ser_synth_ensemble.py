"""Ансамбль vs одиночные на синтетике: macro-F1 нативно (HuBERT, GigaAM, наша FT, стекинг)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
from app.pipeline.ensemble import CLASSES, get_ensemble_recognizer  # noqa: E402

AUDIO = PROJECT_ROOT / "eval" / "audio" / "e2e"
RU = {"neutral": "спок", "angry": "злость", "positive": "позит", "sad": "грусть"}


def macro_f1(y, p):
    f1 = []
    for c in range(4):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        pr = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0
        f1.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
    return float(np.mean(f1))


def main():
    e = get_ensemble_recognizer()
    e._load_stack()
    coef, intercept = e._stack
    man = json.loads((AUDIO / "manifest.json").read_text(encoding="utf-8"))
    rows = [m for m in man if m.get("ok") and (AUDIO / m["file"]).exists() and m.get("emotion") in CLASSES]
    y = np.array([CLASSES.index(m["emotion"]) for m in rows])
    P = {"HuBERT": [], "GigaAM": [], "Наша FT": [], "Ансамбль": []}
    for i, m in enumerate(rows):
        path = str(AUDIO / m["file"])
        wav, _ = librosa.load(path, sr=16000, mono=True)
        hu = e._probs_hubert(path); ga = e._probs_gigaam(wav); ou = e._probs_ours(wav)
        ens = coef @ np.concatenate([ga, hu, ou]) + intercept
        P["HuBERT"].append(hu.argmax()); P["GigaAM"].append(ga.argmax())
        P["Наша FT"].append(ou.argmax()); P["Ансамбль"].append(int(ens.argmax()))
        if (i + 1) % 60 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)
    print(f"\n=== Синтетика (n={len(rows)}), macro-F1 ===")
    out = {}
    for name, preds in P.items():
        f = macro_f1(y, np.array(preds))
        out[name] = round(f, 3)
        print(f"  {name:10s}  macro-F1 = {f:.3f}")
    (PROJECT_ROOT / "eval" / "results" / "ser_synth_ensemble.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
