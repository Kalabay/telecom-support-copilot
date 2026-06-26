"""emotion2vec: одиночно + декорреляция с GA/HU + в ансамбль. Помогает или нет?"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

R = Path(r"K:\dev\coursework\eval\results")


def macro_f1(yt, yp):
    f = []
    for c in range(4):
        tp = int(((yt == c) & (yp == c)).sum()); fp = int(((yt != c) & (yp == c)).sum())
        fn = int(((yt == c) & (yp != c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        f.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f))


def load_pred(name):
    return {str(p["id"]): p for p in json.loads((R / name).read_text(encoding="utf-8"))["preds"]}


def boot_ci(yt, yp, it=2000, seed=0):
    rng = np.random.RandomState(seed); n = len(yt); s = []
    for _ in range(it):
        b = rng.randint(0, n, n); s.append(macro_f1(yt[b], yp[b]))
    return np.percentile(s, 2.5), np.percentile(s, 97.5)


preds = {k: load_pred(f"_pred_{k}_crowd.json") for k in ["ga", "hu", "ours", "e2v"]}
ids = sorted(set.intersection(*[set(p) for p in preds.values()]))
y = np.array([preds["ga"][i]["true"] for i in ids])
P = {k: np.array([preds[k][i]["probs"] for i in ids]) for k in preds}
pred = {k: P[k].argmax(1) for k in P}
print(f"общих id: {len(ids)}\n")

print("=== 1. emotion2vec САМ ПО СЕБЕ ===")
f1 = macro_f1(y, pred["e2v"]); lo, hi = boot_ci(y, pred["e2v"])
print(f"  emotion2vec macro-F1 = {f1:.4f}  CI [{lo:.3f}, {hi:.3f}]")
print(f"  (для сравнения: HuBERT 0.796, GigaAM 0.812, наша 0.819)\n")

print("=== 2. ДЕКОРРЕЛЯЦИЯ ОШИБОК (ключ: помогает только некоррелированный) ===")
err = {k: (pred[k] != y) for k in ["ga", "hu", "e2v"]}
for a in ["ga", "hu"]:
    both = (err[a] & err["e2v"]).sum()
    only_e = (~err[a] & err["e2v"]).sum()
    only_a = (err[a] & ~err["e2v"]).sum()
    union = (err[a] | err["e2v"]).sum()
    jacc = both / union if union else 0
    print(f"  e2v vs {a}: общие ошибки={both}, только {a}={only_a}, только e2v={only_e}, "
          f"Jaccard={jacc:.2f}")
print("  (низкий Jaccard = ошибки РАЗНЫЕ = e2v комплементарен = поможет в ансамбле)\n")

print("=== 3. В АНСАМБЛЬ (logreg stacking, 5-fold OOF) ===")


def stack(keys):
    X = np.hstack([P[k] for k in keys])
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    pr = np.zeros(len(y), dtype=int)
    for tr, te in skf.split(X, y):
        pr[te] = LogisticRegression(max_iter=2000).fit(X[tr], y[tr]).predict(X[te])
    return pr


combos = [
    ("GA+HU (база)", ["ga", "hu"]),
    ("GA+HU+наша", ["ga", "hu", "ours"]),
    ("GA+HU+e2v", ["ga", "hu", "e2v"]),
    ("GA+HU+наша+e2v", ["ga", "hu", "ours", "e2v"]),
]
base = None
for name, keys in combos:
    pr = stack(keys); f1 = macro_f1(y, pr); lo, hi = boot_ci(y, pr)
    if name.startswith("GA+HU (база)"):
        base = f1
    delta = f"  delta={f1-base:+.4f}" if base is not None and not name.startswith("GA+HU (база)") else ""
    print(f"  {name:18s} F1={f1:.4f}  CI [{lo:.3f}, {hi:.3f}]{delta}")
print("\n  Вердикт: если GA+HU+e2v значимо > GA+HU → emotion2vec пробил потолок.")
