"""Есть ли у текста КОМПЛЕМЕНТАРНЫЙ сигнал к аудио?."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


def load(p: Path):
    obj = json.loads(Path(p).read_text(encoding="utf-8"))
    return {str(r["id"]): r for r in obj["preds"]}, obj["classes"]


def mf1(yt, yp):
    return f1_score(yt, yp, average="macro", zero_division=0)


def main() -> None:
    ga, classes = load(sys.argv[1])
    hu, _ = load(sys.argv[2])
    tx, _ = load(sys.argv[3])
    ids = sorted(set(ga) & set(hu) & set(tx))
    n = len(ids)
    print(f"n={n}, классы={classes}\n")

    y = np.array([ga[i]["true"] for i in ids])
    Pga = np.array([ga[i]["probs"] for i in ids])
    Phu = np.array([hu[i]["probs"] for i in ids])
    Ptx = np.array([tx[i]["probs"] for i in ids])

    Paud = (Pga + Phu) / 2
    aud_pred = Paud.argmax(1)
    tx_pred = Ptx.argmax(1)

    aud_ok = aud_pred == y
    tx_ok = tx_pred == y

    print(f"аудио-ансамбль   acc={aud_ok.mean():.4f}  F1={mf1(y, aud_pred):.4f}")
    print(f"текст            acc={tx_ok.mean():.4f}  F1={mf1(y, tx_pred):.4f}\n")

    aud_wrong = ~aud_ok
    recoverable = aud_wrong & tx_ok
    at_risk = aud_ok & ~tx_ok
    print(f"аудио неправо:           {aud_wrong.sum():4d} ({aud_wrong.mean()*100:.1f}%)")
    print(f"  из них текст прав:      {recoverable.sum():4d}  <- потенциально восстановимо")
    print(f"  P(текст прав | аудио неправо) = {recoverable.sum()/max(aud_wrong.sum(),1):.3f}")
    print(f"     (если бы текст был СЛУЧАЕН ~ {1/len(classes):.3f})")
    print(f"аудио право, текст неправо: {at_risk.sum():4d}  <- риск испортить\n")

    oracle = np.where(aud_ok, aud_pred, np.where(tx_ok, tx_pred, aud_pred))
    print(f"ORACLE (идеальный выбор)  acc={(oracle==y).mean():.4f}  F1={mf1(y, oracle):.4f}"
          f"   <- верхняя граница выигрыша от текста\n")

    aud_conf = Paud.max(1)
    ga_pred, hu_pred = Pga.argmax(1), Phu.argmax(1)
    disagree = ga_pred != hu_pred
    for thr in (0.5, 0.6, 0.7):
        gate = aud_conf < thr
        gated = np.where(gate, tx_pred, aud_pred)
        print(f"gate (aud_conf<{thr}): срабатывает {gate.mean()*100:4.1f}%  "
              f"F1={mf1(y, gated):.4f}")
    gated_dis = np.where(disagree, tx_pred, aud_pred)
    print(f"gate (audio disagree):  срабатывает {disagree.mean()*100:4.1f}%  "
          f"F1={mf1(y, gated_dis):.4f}")
    soft = Paud.copy()
    w = 0.5
    lowconf = aud_conf < 0.6
    soft[lowconf] = (1 - w) * Paud[lowconf] + w * Ptx[lowconf]
    print(f"soft-gate (lowconf+={w}*text): F1={mf1(y, soft.argmax(1)):.4f}")


if __name__ == "__main__":
    main()
