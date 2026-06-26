"""Умный комбайнер: нелинейный мета-классификатор с фичами уверенности/согласия."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier


def load(p: str):
    obj = json.loads(Path(p).read_text(encoding="utf-8"))
    return {str(r["id"]): r for r in obj["preds"]}


def entropy(P):
    return -(P * np.log(np.clip(P, 1e-9, 1))).sum(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ga", required=True)
    ap.add_argument("--hu", required=True)
    ap.add_argument("--text", nargs="*", default=[])
    args = ap.parse_args()

    ga, hu = load(args.ga), load(args.hu)
    texts = [load(t) for t in args.text]
    sets = [set(ga), set(hu)] + [set(t) for t in texts]
    ids = sorted(set.intersection(*sets))
    y = np.array([ga[i]["true"] for i in ids])
    Pga = np.array([ga[i]["probs"] for i in ids])
    Phu = np.array([hu[i]["probs"] for i in ids])
    Pts = [np.array([t[i]["probs"] for i in ids]) for t in texts]
    print(f"n={len(ids)}, текст-каналов={len(texts)}")

    Paud = (Pga + Phu) / 2
    aud_conf = Paud.max(1, keepdims=True)
    disagree = (Pga.argmax(1) != Phu.argmax(1)).astype(float)[:, None]
    aud_ent = entropy(Paud)[:, None]

    X_probs_A = np.hstack([Pga, Phu])
    X_probs_AT = np.hstack([Pga, Phu] + Pts)
    meta = np.hstack([aud_conf, disagree, aud_ent] +
                     [Pt.max(1, keepdims=True) for Pt in Pts])
    X_smart = np.hstack([X_probs_AT, meta])

    ga_pred, hu_pred = Pga.argmax(1), Phu.argmax(1)
    mean_A = (0.5 * Pga + 0.5 * Phu).argmax(1)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = {
        "GigaAM (one)": ("static", ga_pred),
        "HuBERT (one)": ("static", hu_pred),
        "mean(A)": ("static", mean_A),
        "logreg stack(A)": ("clf", LogisticRegression(max_iter=2000), X_probs_A),
        "logreg stack(A+T)": ("clf", LogisticRegression(max_iter=2000), X_probs_AT),
        "HGB stack(A+T)": ("clf", HistGradientBoostingClassifier(
            max_depth=3, max_iter=300, learning_rate=0.05, l2_regularization=1.0),
            X_probs_AT),
        "HGB smart(A+T+meta)": ("clf", HistGradientBoostingClassifier(
            max_depth=3, max_iter=300, learning_rate=0.05, l2_regularization=1.0),
            X_smart),
        "MLP smart(A+T+meta)": ("clf", MLPClassifier(
            hidden_layer_sizes=(32,), max_iter=1000, alpha=1e-2, random_state=0),
            X_smart),
    }

    res = {k: [] for k in rows}
    for tr, te in skf.split(X_smart, y):
        for name, spec in rows.items():
            if spec[0] == "static":
                pred = spec[1][te]
            else:
                _, clf, X = spec
                from sklearn.base import clone
                c = clone(clf).fit(X[tr], y[tr])
                pred = c.predict(X[te])
            res[name].append(f1_score(y[te], pred, average="macro", zero_division=0))

    print("\n5-fold CV (mean +/- std):")
    base = None
    for k in rows:
        a = np.array(res[k])
        if k == "logreg stack(A)":
            base = a.mean()
        tag = ""
        if base is not None and "A+T" in k:
            tag = f"  (vs stack(A): {a.mean()-base:+.4f})"
        print(f"  {k:24s} {a.mean():.4f} +/- {a.std():.4f}{tag}")


if __name__ == "__main__":
    main()
