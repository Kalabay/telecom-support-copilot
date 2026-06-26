"""Полное SER-сравнение на СИНТЕТИКЕ из готовых эмбеддингов (без GPU, без сети)."""
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression

sys.stdout.reconfigure(encoding="utf-8")
R = "eval/results/"
CLASSES = ["neutral", "angry", "positive", "sad"]
ANGRY = CLASSES.index("angry")
RU = {"neutral": "спокойн", "angry": "раздраж", "positive": "позитив", "sad": "грусть"}


def load(m, split):
    d = np.load(f"{R}emb_{m}_{split}.npz", allow_pickle=True)
    return d["emb"], d["labels"].astype(int), d["ids"]


def f1_macro_perclass(y, pred, k):
    f1s = []
    for c in range(k):
        tp = np.sum((pred == c) & (y == c)); fp = np.sum((pred == c) & (y != c)); fn = np.sum((pred != c) & (y == c))
        p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
        f1s.append(2 * p * r / (p + r) if p + r else 0)
    return np.mean(f1s), f1s


def fit_predict(Xtr, ytr, Xte):
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, ytr)
    return clf.predict(Xte), clf.predict_proba(Xte)


def main():
    ga_tr, y_tr, _ = load("gigaam", "train"); hu_tr, _, _ = load("hubert", "train")
    ga_s, y_s, _ = load("gigaam", "synth"); hu_s, _, _ = load("hubert", "synth")
    print("синтетика n =", len(y_s), "распределение:", {RU[CLASSES[c]]: int(np.sum(y_s == c)) for c in range(4)})

    preds = {}
    preds["GigaAM"], pg = fit_predict(ga_tr, y_tr, ga_s)
    preds["HuBERT"], ph = fit_predict(hu_tr, y_tr, hu_s)
    preds["Ансамбль(GA+HU)"], _ = fit_predict(np.hstack([ga_tr, hu_tr]), y_tr, np.hstack([ga_s, hu_s]))

    print(f"\n=== СИНТЕТИКА: 4 класса, per-class F1 ===")
    print(f"{'класс':9s} {'n':>4s} | " + " | ".join(f"{m:>16s}" for m in preds))
    for c in range(4):
        line = f"{RU[CLASSES[c]]:9s} {int(np.sum(y_s==c)):4d} | "
        line += " | ".join(f"{f1_macro_perclass(y_s,p,4)[1][c]:15.3f} " for p in preds.values())
        print(line)
    print("-" * 60)
    for m, p in preds.items():
        print(f"{m:18s} macro-F1(4) = {f1_macro_perclass(y_s,p,4)[0]:.3f}  acc = {np.mean(p==y_s):.3f}")

    print(f"\n=== ЗАДАЧНО-важное: детекция ЗЛОСТИ (бинарно: злость vs остальное) ===")
    for m, p in preds.items():
        ya = (y_s == ANGRY).astype(int); pa = (p == ANGRY).astype(int)
        tp = np.sum(pa & ya); fp = np.sum(pa & ~ya.astype(bool)); fn = np.sum((~pa.astype(bool)) & ya.astype(bool))
        prec = tp / (tp + fp) if tp + fp else 0; rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        print(f"{m:18s} злость: precision {prec:.3f}  recall {rec:.3f}  F1 {f1:.3f}")

    print(f"\n=== Слитый класс: нейтральное+позитив = 'спокойный' (3 класса) ===")
    merge = {0: 0, 1: 1, 2: 0, 3: 2}
    ym = np.array([merge[c] for c in y_s])
    for m, p in preds.items():
        pm = np.array([merge[c] for c in p])
        f1, per = f1_macro_perclass(ym, pm, 3)
        print(f"{m:18s} macro-F1(3, спок/злость/грусть) = {f1:.3f}  "
              f"[спок {per[0]:.2f}, злость {per[1]:.2f}, грусть {per[2]:.2f}]")


if __name__ == "__main__":
    main()
