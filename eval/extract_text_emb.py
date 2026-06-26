"""Текстовые эмбеддинги CEDR (pre-classifier, hook) для fusion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import app  # noqa: F401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

CLASSES = ["neutral", "angry", "positive", "sad"]
C2I = {c: i for i, c in enumerate(CLASSES)}
MODEL = "seara/rubert-base-cased-russian-emotion-detection-cedr"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, help="jsonl {id,text,emotion}")
    ap.add_argument("--preds", type=Path, help="preds json с id,text,true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    if args.src:
        for l in args.src.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            emo = str(r.get("emotion", "")).lower()
            if emo not in C2I:
                continue
            rows.append((r["id"], r.get("text") or "", C2I[emo]))
    else:
        for r in json.loads(args.preds.read_text(encoding="utf-8"))["preds"]:
            rows.append((r["id"], r.get("text") or "", int(r["true"])))
    print(f"строк: {len(rows)}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(dev).eval()
    store = {}
    model.classifier.register_forward_pre_hook(
        lambda m, inp: store.__setitem__("e", inp[0].detach().cpu().numpy()))

    ids, embs, labs = [], [], []
    bs = 64
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        texts = [(t or "").strip() or "." for _, t, _ in chunk]
        enc = tok(texts, padding=True, truncation=True, max_length=128,
                  return_tensors="pt").to(dev)
        with torch.inference_mode():
            model(**enc)
        e = store["e"]
        for j, (fid, _, lab) in enumerate(chunk):
            ids.append(fid); embs.append(e[j].astype("float32")); labs.append(lab)
    embs = np.array(embs)
    print(f"эмбеддингов: {len(ids)} dim={embs.shape[1]}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, ids=np.array(ids), emb=embs, labels=np.array(labs))
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
