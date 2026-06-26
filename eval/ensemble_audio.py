"""Ансамбль произвольного числа аудио-каналов + prior-коррекция под macro-F1."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

CLASSES = ["neutral", "angry", "positive", "sad"]


def load(p: str):
    obj = json.loads(Path(p).read_text(encoding="utf-8"))
    return {str(r["id"]): r for r in obj["preds"]}


def mf1(yt, yp):
    return f1_score(yt, yp, average="macro", zero_division=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", nargs="+", required=True)
    args = ap.parse_args()

    chans = [load(a) for a in args.audio]
    names = [Path(a).stem.replace("_pred_", "").replace("_crowd", "") for a in args.audio]
    ids = sorted(set.intersection(*[set(c) for c in chans]))
    y = np.array([chans[0][i]["true"] for i in ids])
    Ps = [np.array([c[i]["probs"] for i in ids]) for c in chans]
    print(f"n={len(ids)}, аудио-каналов={len(chans)}: {names}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def eval_method(fn) -> tuple[float, float]:
        scores = []
        for tr, te in skf.split(Ps[0], y):
            scores.append(fn(tr, te))
        a = np.array(scores)
        return a.mean(), a.std()

    print("\nОдиночные модели (5-fold CV):")
    for nm, P in zip(names, Ps):
        pred = P.argmax(1)
        m, s = eval_method(lambda tr, te, p=pred: mf1(y[te], p[te]))
        print(f"  {nm:18s} {m:.4f} +/- {s:.4f}")

    print("\nАнсамбли (5-fold CV):")
    idx_sets = []
    if len(Ps) >= 2:
        idx_sets += [list(c) for c in combinations(range(len(Ps)), 2)]
    if len(Ps) >= 3:
        idx_sets += [list(range(len(Ps)))]

    for idxs in idx_sets:
        tag = "+".join(names[i] for i in idxs)
        stacked = np.hstack([Ps[i] for i in idxs])
        meanP = sum(Ps[i] for i in idxs) / len(idxs)

        m_mean, s_mean = eval_method(lambda tr, te, mp=meanP: mf1(y[te], mp.argmax(1)[te]))

        def stack_fn(tr, te, X=stacked):
            c = clone(LogisticRegression(max_iter=2000)).fit(X[tr], y[tr])
            return mf1(y[te], c.predict(X[te]))
        m_st, s_st = eval_method(stack_fn)

        def stack_prior_fn(tr, te, X=stacked):
            c = clone(LogisticRegression(max_iter=2000)).fit(X[tr], y[tr])
            proba_tr = c.predict_proba(X[tr])
            prior = np.bincount(y[tr], minlength=4) / len(tr)
            logp = np.log(np.clip(prior, 1e-9, 1))
            best_t, best_f = 0.0, -1.0
            for t in np.linspace(0, 1.5, 31):
                pr = (np.log(np.clip(proba_tr, 1e-9, 1)) - t * logp).argmax(1)
                f = mf1(y[tr], pr)
                if f > best_f:
                    best_f, best_t = f, t
            proba_te = c.predict_proba(X[te])
            pr_te = (np.log(np.clip(proba_te, 1e-9, 1)) - best_t * logp).argmax(1)
            return mf1(y[te], pr_te)
        m_sp, s_sp = eval_method(stack_prior_fn)

        print(f"  [{tag}]")
        print(f"     mean            {m_mean:.4f} +/- {s_mean:.4f}")
        print(f"     stack           {m_st:.4f} +/- {s_st:.4f}")
        print(f"     stack+prior     {m_sp:.4f} +/- {s_sp:.4f}")


if __name__ == "__main__":
    main()
