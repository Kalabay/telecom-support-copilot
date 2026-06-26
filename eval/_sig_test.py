"""Парный bootstrap: значима ли разница our vs GigaAM на тех же 2000 crowd test."""

import json
import sys
from pathlib import Path

import numpy as np

CLASSES = ["neutral", "angry", "positive", "sad"]


def load(p):
    return {str(r["id"]): r for r in json.loads(Path(p).read_text(encoding="utf-8"))["preds"]}


def macro_f1(yt, yp):
    f1s = []
    for c in range(4):
        tp = int(((yt == c) & (yp == c)).sum())
        fp = int(((yt != c) & (yp == c)).sum())
        fn = int(((yt == c) & (yp != c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return np.mean(f1s)


ga = load(sys.argv[1])
hu = load(sys.argv[2])
ours = load(sys.argv[3])
ids = sorted(set(ga) & set(hu) & set(ours))
y = np.array([ga[i]["true"] for i in ids])
pred_ga = np.array([int(np.argmax(ga[i]["probs"])) for i in ids])
pred_hu = np.array([int(np.argmax(hu[i]["probs"])) for i in ids])
pred_ours = np.array([int(np.argmax(ours[i]["probs"])) for i in ids])
n = len(ids)
print(f"n={n}")
print(f"full-set macro-F1:  GigaAM={macro_f1(y,pred_ga):.4f}  HuBERT={macro_f1(y,pred_hu):.4f}  "
      f"ours={macro_f1(y,pred_ours):.4f}")

rng = np.random.RandomState(0)
diffs_og, diffs_oh = [], []
for _ in range(2000):
    idx = rng.randint(0, n, n)
    f_o = macro_f1(y[idx], pred_ours[idx])
    f_g = macro_f1(y[idx], pred_ga[idx])
    f_h = macro_f1(y[idx], pred_hu[idx])
    diffs_og.append(f_o - f_g)
    diffs_oh.append(f_o - f_h)
diffs_og = np.array(diffs_og)
diffs_oh = np.array(diffs_oh)

def ci(d):
    return np.percentile(d, 2.5), np.percentile(d, 97.5)

lo, hi = ci(diffs_og)
print(f"\nours - GigaAM: mean={diffs_og.mean():+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]  "
      f"P(ours>GigaAM)={(diffs_og>0).mean():.2f}")
print(f"  -> {'ЗНАЧИМО' if lo>0 else 'НЕ значимо (CI включает 0) = ничья'}")
lo2, hi2 = ci(diffs_oh)
print(f"ours - HuBERT: mean={diffs_oh.mean():+.4f}  95% CI=[{lo2:+.4f}, {hi2:+.4f}]  "
      f"P(ours>HuBERT)={(diffs_oh>0).mean():.2f}")
print(f"  -> {'ЗНАЧИМО' if lo2>0 else 'НЕ значимо'}")
