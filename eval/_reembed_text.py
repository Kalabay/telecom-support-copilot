"""Пересчитать text-probs на уже готовых транскриптах (без ASR) другой головой."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))
import app  # noqa: F401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

from train_text_head import TextHead  # noqa: E402

CLASSES = ["neutral", "angry", "positive", "sad"]
C2I = {c: i for i, c in enumerate(CLASSES)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.head, map_location="cpu", weights_only=False)
    text_classes = ckpt["classes"]
    d_in = ckpt.get("d_in", 312)
    t_map = {i: C2I[c] for i, c in enumerate(text_classes) if c in C2I}
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"энкодер: {ckpt['text_model']}  d_in={d_in}  on {dev}", flush=True)
    tok = AutoTokenizer.from_pretrained(ckpt["text_model"])
    base = AutoModel.from_pretrained(ckpt["text_model"]).to(dev).eval()
    head = TextHead(d_in=d_in).to(dev).eval()
    head.load_state_dict(ckpt["state_dict"])

    src = json.loads(args.src.read_text(encoding="utf-8"))["preds"]
    print(f"транскриптов: {len(src)}", flush=True)

    def embed_batch(texts: list[str]) -> np.ndarray:
        enc = tok(texts, padding=True, truncation=True, max_length=128,
                  return_tensors="pt").to(dev)
        with torch.inference_mode():
            out = base(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1).clamp_min(1)
            logits = head(pooled)
            return torch.softmax(logits, -1).cpu().numpy()

    out, t0, bs = [], time.perf_counter(), 64
    for i in range(0, len(src), bs):
        chunk = src[i:i + bs]
        texts = [(r.get("text") or "").strip() or "." for r in chunk]
        probs5 = embed_batch(texts)
        for r, p5 in zip(chunk, probs5):
            vec = np.zeros(4)
            for i5, i4 in t_map.items():
                vec[i4] = p5[i5]
            s = vec.sum()
            vec = vec / s if s > 0 else np.full(4, 0.25)
            out.append({"id": r["id"], "true": int(r["true"]),
                         "pred": int(vec.argmax()),
                         "probs": [round(float(x), 6) for x in vec]})
    print(f"готово за {time.perf_counter()-t0:.1f}s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"classes": CLASSES, "source": "crowd", "model": f"text:{ckpt['text_model']}",
         "preds": out}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {len(out)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
