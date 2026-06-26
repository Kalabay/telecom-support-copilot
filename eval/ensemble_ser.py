"""Ансамбль SER на crowd Dusha: GigaAM + HuBERT (+ опционально text)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


def load(p: Path) -> tuple[dict[str, dict], list[str]]:
    obj = json.loads(p.read_text(encoding="utf-8"))
    return {str(r["id"]): r for r in obj["preds"]}, obj["classes"]


def mf1(yt, yp) -> float:
    return f1_score(yt, yp, average="macro", zero_division=0)


def main() -> None:
    ga, classes = load(Path(sys.argv[1]))
    hu, _ = load(Path(sys.argv[2]))
    tx = load(Path(sys.argv[3]))[0] if len(sys.argv) > 3 else None

    sets = [set(ga), set(hu)] + ([set(tx)] if tx else [])
    ids = sorted(set.intersection(*sets))
    print(f"общих id: {len(ids)}  (text-канал: {'да' if tx else 'нет'})")

    y = np.array([ga[i]["true"] for i in ids])
    Pga = np.array([ga[i]["probs"] for i in ids])
    Phu = np.array([hu[i]["probs"] for i in ids])
    Ptx = np.array([tx[i]["probs"] for i in ids]) if tx else None

    ga_pred, hu_pred = Pga.argmax(1), Phu.argmax(1)
    methods = {
        "GigaAM": lambda I: ga_pred[I],
        "HuBERT": lambda I: hu_pred[I],
    }
    if tx is not None:
        tx_pred = Ptx.argmax(1)
        methods["Text(ASR)"] = lambda I: tx_pred[I]

    X_av = np.hstack([Pga, Phu])
    X_all = np.hstack([Pga, Phu, Ptx]) if tx is not None else X_av

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    order = (["HuBERT", "GigaAM"] + (["Text(ASR)"] if tx else [])
             + ["mean(A)", "soft-vote(A)", "stack(A)"]
             + (["mean(A+T)", "stack(A+T)"] if tx else []))
    res = {k: [] for k in order}
    tx_weight_share = []

    for tr, te in skf.split(X_all, y):
        for name, fn in methods.items():
            res[name].append(mf1(y[te], fn(te)))

        res["mean(A)"].append(mf1(y[te], (0.5 * Pga + 0.5 * Phu).argmax(1)[te]))
        ba, bv = 0.5, -1.0
        for a in np.linspace(0, 1, 51):
            v = mf1(y[tr], (a * Pga + (1 - a) * Phu).argmax(1)[tr])
            if v > bv:
                bv, ba = v, a
        res["soft-vote(A)"].append(mf1(y[te], (ba * Pga + (1 - ba) * Phu).argmax(1)[te]))
        c_a = LogisticRegression(max_iter=2000, C=1.0).fit(X_av[tr], y[tr])
        res["stack(A)"].append(mf1(y[te], c_a.predict(X_av[te])))

        if tx is not None:
            res["mean(A+T)"].append(
                mf1(y[te], ((Pga + Phu + Ptx) / 3).argmax(1)[te]))
            c_all = LogisticRegression(max_iter=2000, C=1.0).fit(X_all[tr], y[tr])
            res["stack(A+T)"].append(mf1(y[te], c_all.predict(X_all[te])))
            W = np.abs(c_all.coef_)
            tx_weight_share.append(W[:, 8:12].sum() / W.sum())

    print("\n5-fold CV (mean +/- std), n=%d:" % len(ids))
    for k in order:
        a = np.array(res[k])
        print(f"  {k:14s} {a.mean():.4f} +/- {a.std():.4f}")

    if tx is not None:
        share = np.mean(tx_weight_share)
        print(f"\nstacking(A+T): доля |веса| на text-фичах = {share*100:.1f}%  "
              f"(нейтрально было бы 33%)")


if __name__ == "__main__":
    main()
