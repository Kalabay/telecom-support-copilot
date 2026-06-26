"""SER (HuBERT-Dusha) на наших 244 синтетических e2e-репликах -> per-class F1 + confusion."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO = PROJECT_ROOT / "eval" / "audio" / "e2e"
R = PROJECT_ROOT / "eval" / "results"
NAME = "xbgoose/hubert-large-speech-emotion-recognition-russian-dusha-finetuned"
RU = {"neutral": "спокойное", "angry": "раздражение", "positive": "позитив", "sad": "грусть"}


def main() -> None:
    man = json.loads((AUDIO / "manifest.json").read_text(encoding="utf-8"))
    rows = [m for m in man if m.get("ok") and (AUDIO / m["file"]).exists() and m.get("emotion")]
    print(f"{len(rows)} синтетических файлов", flush=True)

    fe = AutoFeatureExtractor.from_pretrained(NAME)
    model = AutoModelForAudioClassification.from_pretrained(NAME).eval()
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    print("классы модели:", id2label, flush=True)

    trues, preds = [], []
    for i, m in enumerate(rows):
        wav, _ = librosa.load(str(AUDIO / m["file"]), sr=16000, mono=True)
        inp = fe(wav, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inp).logits
        preds.append(id2label[int(logits.argmax())])
        trues.append(m["emotion"])
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    classes = ["neutral", "angry", "positive", "sad"]
    acc = np.mean([t == p for t, p in zip(trues, preds)])
    perclass, f1s = {}, []
    for c in classes:
        tp = sum(1 for t, p in zip(trues, preds) if t == c and p == c)
        fp = sum(1 for t, p in zip(trues, preds) if t != c and p == c)
        fn = sum(1 for t, p in zip(trues, preds) if t == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        n = sum(1 for t in trues if t == c)
        perclass[c] = {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3), "n": n}
        f1s.append(f1)
    tp = sum(1 for t, p in zip(trues, preds) if t == "angry" and p == "angry")
    fp = sum(1 for t, p in zip(trues, preds) if t != "angry" and p == "angry")
    fn = sum(1 for t, p in zip(trues, preds) if t == "angry" and p != "angry")
    aprec = tp / (tp + fp) if tp + fp else 0.0
    arec = tp / (tp + fn) if tp + fn else 0.0
    af1 = 2 * aprec * arec / (aprec + arec) if aprec + arec else 0.0
    mg = {"neutral": "спок", "positive": "спок", "angry": "злость", "sad": "грусть"}
    tm = [mg[t] for t in trues]; pm = [mg[p] for p in preds]
    mf1 = []
    for c in ["спок", "злость", "грусть"]:
        tp2 = sum(1 for t, p in zip(tm, pm) if t == c and p == c)
        fp2 = sum(1 for t, p in zip(tm, pm) if t != c and p == c)
        fn2 = sum(1 for t, p in zip(tm, pm) if t == c and p != c)
        pr = tp2 / (tp2 + fp2) if tp2 + fp2 else 0; rc = tp2 / (tp2 + fn2) if tp2 + fn2 else 0
        mf1.append((c, 2 * pr * rc / (pr + rc) if pr + rc else 0))

    out = {"n": len(rows), "accuracy": round(float(acc), 3),
           "f1_macro": round(float(np.mean(f1s)), 3), "per_class": perclass,
           "angry_detect": {"precision": round(aprec, 3), "recall": round(arec, 3), "f1": round(af1, 3)},
           "merged3_macro_f1": round(float(np.mean([f for _, f in mf1])), 3),
           "merged3": {c: round(f, 3) for c, f in mf1},
           "per_clip": [{"file": m["file"], "true": t, "pred": p} for m, t, p in zip(rows, trues, preds)]}
    (R / "ser_synth_e2e.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSER на синтетике (n={len(rows)}): acc={out['accuracy']}, macro-F1(4)={out['f1_macro']}\n")
    print(f"{'класс':12s} {'n':>4s} {'precision':>10s} {'recall':>8s} {'F1':>7s}")
    for c in classes:
        pc = perclass[c]
        print(f"{RU[c]:12s} {pc['n']:4d} {pc['precision']:10.3f} {pc['recall']:8.3f} {pc['f1']:7.3f}")
    print(f"\n>>> ДЕТЕКЦИЯ ЗЛОСТИ (бинарно): precision {aprec:.3f}  recall {arec:.3f}  F1 {af1:.3f}")
    print(f">>> Слитый нейтрал+позитив (3 класса) macro-F1 = {out['merged3_macro_f1']:.3f}  "
          f"[{', '.join(f'{c} {f:.2f}' for c, f in mf1)}]")


if __name__ == "__main__":
    main()
