"""Значимо ли [ga+hu+ours] лучше [ga+hu]? Bootstrap на held-out половине."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


def load(p):
    return {str(r["id"]): r for r in json.loads(Path(p).read_text(encoding="utf-8"))["preds"]}


def macro_f1(yt, yp):
    f1s = []
    for c in range(4):
        tp = int(((yt == c) & (yp == c)).sum()); fp = int(((yt != c) & (yp == c)).sum())
        fn = int(((yt == c) & (yp != c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return np.mean(f1s)


ga, hu, ours = load(sys.argv[1]), load(sys.argv[2]), load(sys.argv[3])
ids = sorted(set(ga) & set(hu) & set(ours))
y = np.array([ga[i]["true"] for i in ids])
Pga = np.array([ga[i]["probs"] for i in ids])
Phu = np.array([hu[i]["probs"] for i in ids])
Pou = np.array([ours[i]["probs"] for i in ids])
n = len(ids)

rng = np.random.RandomState(0)
perm = rng.permutation(n)
tr, te = perm[: n // 2], perm[n // 2:]

X_pair = np.hstack([Pga, Phu])
X_tri = np.hstack([Pga, Phu, Pou])
clf_p = LogisticRegression(max_iter=2000).fit(X_pair[tr], y[tr])
clf_t = LogisticRegression(max_iter=2000).fit(X_tri[tr], y[tr])
pred_p = clf_p.predict(X_pair[te])
pred_t = clf_t.predict(X_tri[te])
yte = y[te]
print(f"test n={len(te)}")
print(f"pair  [ga+hu]      macro-F1={macro_f1(yte,pred_p):.4f}")
print(f"triple[ga+hu+ours] macro-F1={macro_f1(yte,pred_t):.4f}")

diffs = []
m = len(te)
for _ in range(2000):
    b = rng.randint(0, m, m)
    diffs.append(macro_f1(yte[b], pred_t[b]) - macro_f1(yte[b], pred_p[b]))
diffs = np.array(diffs)
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"\ntriple - pair: mean={diffs.mean():+.4f}  95% CI=[{lo:+.4f},{hi:+.4f}]  "
      f"P(triple>pair)={(diffs>0).mean():.2f}")
print(f"  -> {'ЗНАЧИМО' if lo>0 else 'НЕ значимо = ансамбль с нашей не лучше пары'}")
