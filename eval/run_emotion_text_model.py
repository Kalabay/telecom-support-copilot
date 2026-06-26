"""Готовая русская text-emotion модель как текстовый канал ансамбля."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import app  # noqa: F401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

CLASSES = ["neutral", "angry", "positive", "sad"]
C2I = {c: i for i, c in enumerate(CLASSES)}

NAME_MAP = {
    "no_emotion": "neutral", "neutral": "neutral", "no emotion": "neutral",
    "anger": "angry", "angry": "angry", "anger/disgust": "angry",
    "joy": "positive", "happiness": "positive", "positive": "positive",
    "enthusiasm": "positive", "interest": "positive",
    "sadness": "sad", "sad": "sad", "grief": "sad",
}


def build_map(id2label: dict[int, str]) -> dict[int, int]:
    out = {}
    for i, name in id2label.items():
        key = str(name).lower().strip()
        if key in NAME_MAP:
            out[int(i)] = C2I[NAME_MAP[key]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--multilabel", action="store_true",
                    help="sigmoid вместо softmax (для CEDR multi-label)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"гружу {args.model} на {dev}…", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(dev).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    print("id2label:", id2label, flush=True)
    lmap = build_map(id2label)
    mapped_names = {id2label[i]: CLASSES[j] for i, j in lmap.items()}
    print("маппинг на Dusha-4:", mapped_names, flush=True)
    missing = set(CLASSES) - {CLASSES[j] for j in lmap.values()}
    if missing:
        print(f"ВНИМАНИЕ: нет источника для классов {missing} -> будут 0", flush=True)

    src = json.loads(args.src.read_text(encoding="utf-8"))["preds"]
    print(f"транскриптов: {len(src)}", flush=True)

    out, t0, bs = [], time.perf_counter(), 64
    for i in range(0, len(src), bs):
        chunk = src[i:i + bs]
        texts = [(r.get("text") or "").strip() or "." for r in chunk]
        enc = tok(texts, padding=True, truncation=True, max_length=128,
                  return_tensors="pt").to(dev)
        with torch.inference_mode():
            logits = model(**enc).logits
            scores = (torch.sigmoid(logits) if args.multilabel
                      else torch.softmax(logits, -1)).cpu().numpy()
        for r, sc in zip(chunk, scores):
            vec = np.zeros(4)
            for src_i, dst_i in lmap.items():
                vec[dst_i] += sc[src_i]
            s = vec.sum()
            vec = vec / s if s > 0 else np.full(4, 0.25)
            out.append({"id": r["id"], "true": int(r["true"]),
                         "pred": int(vec.argmax()),
                         "probs": [round(float(x), 6) for x in vec]})
    print(f"готово за {time.perf_counter()-t0:.1f}s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"classes": CLASSES, "source": "crowd", "model": f"emo:{args.model}",
         "preds": out}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {len(out)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
