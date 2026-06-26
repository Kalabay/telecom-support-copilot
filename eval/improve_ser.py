"""Улучшение SER без обучения: logit adjustment по приорам классов."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import app  # noqa: F401,E402

import numpy as np  # noqa: E402
from datasets import Audio, load_dataset  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from app.pipeline.ser import get_recognizer  # noqa: E402

CLASSES = ["neutral", "angry", "positive", "sad", "other"]
C2I = {c: i for i, c in enumerate(CLASSES)}
OUT = PROJECT_ROOT / "eval" / "results" / "ser_improved.json"


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, dict]:
    per, f1s = {}, []
    for ci, c in enumerate(CLASSES):
        tp = int(np.sum((y_true == ci) & (y_pred == ci)))
        fp = int(np.sum((y_true != ci) & (y_pred == ci)))
        fn = int(np.sum((y_true == ci) & (y_pred != ci)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per[c] = {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}
        f1s.append(f1)
    return float(np.mean(f1s)), per


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    args = ap.parse_args()
    console = Console()
    console.rule(f"[bold cyan]SER improve — logit adjustment[/]  (стрим {args.n})")

    rec = get_recognizer()
    rec._ensure_loaded()  # noqa: SLF001

    ds = load_dataset("xbgoose/dusha", split="test", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    probs_all, y_all = [], []
    t0 = time.perf_counter()
    for item in ds:
        if len(probs_all) >= args.n:
            break
        true = str(item["emotion"]).lower().strip()
        if true not in C2I:
            continue
        raw = item["audio"].get("bytes")
        if not raw:
            continue
        res = rec.predict(raw)
        p = np.array([res.probs.get(c, 0.0) for c in CLASSES], dtype=np.float64)
        s = p.sum()
        if s <= 0:
            continue
        probs_all.append(p / s)
        y_all.append(C2I[true])
        if len(probs_all) % 100 == 0:
            console.log(f"  {len(probs_all)}/{args.n}")

    P = np.stack(probs_all)
    y = np.array(y_all)
    console.log(f"Собрано {len(y)} сэмплов за {time.perf_counter()-t0:.0f}s")

    half = len(y) // 2
    P_test, y_test = P[:half], y[:half]
    P_val, y_val = P[half:], y[half:]

    logP = np.log(np.clip(P, 1e-9, 1.0))
    logP_test, logP_val = logP[:half], logP[half:]

    val_counts = np.array([(y_val == i).sum() for i in range(5)], dtype=np.float64)
    prior = val_counts / val_counts.sum()
    log_prior = np.log(np.clip(prior, 1e-9, 1.0))

    base_test_f1, base_per = macro_f1(y_test, logP_test.argmax(1))

    best_tau, best_val_f1 = 0.0, -1.0
    grid = [round(t, 2) for t in np.arange(0.0, 2.01, 0.1)]
    val_curve = {}
    for tau in grid:
        pred_val = (logP_val - tau * log_prior).argmax(1)
        f1, _ = macro_f1(y_val, pred_val)
        val_curve[tau] = round(f1, 4)
        if f1 > best_val_f1:
            best_val_f1, best_tau = f1, tau

    imp_pred = (logP_test - best_tau * log_prior).argmax(1)
    imp_test_f1, imp_per = macro_f1(y_test, imp_pred)

    cmp = Table(title="macro-F1 на TEST (n={})".format(half))
    cmp.add_column("вариант", style="cyan")
    cmp.add_column("macro-F1", justify="right")
    cmp.add_column("delta", justify="right")
    cmp.add_row("baseline (argmax, τ=0)", f"{base_test_f1:.4f}", "—")
    d = imp_test_f1 - base_test_f1
    col = "green" if d > 0 else "red"
    cmp.add_row(f"logit-adjust (τ={best_tau})", f"{imp_test_f1:.4f}", f"[{col}]{d:+.4f}[/]")
    console.print(cmp)
    console.log(f"τ подобран на VAL (val macro-F1 {best_val_f1:.4f} при τ={best_tau})")

    per_t = Table(title="Per-class F1: baseline -> improved (TEST)")
    per_t.add_column("класс", style="cyan")
    per_t.add_column("base F1", justify="right")
    per_t.add_column("impr F1", justify="right")
    per_t.add_column("base rec", justify="right")
    per_t.add_column("impr rec", justify="right")
    for c in CLASSES:
        per_t.add_row(c, f"{base_per[c]['f1']:.3f}", f"{imp_per[c]['f1']:.3f}",
                      f"{base_per[c]['recall']:.3f}", f"{imp_per[c]['recall']:.3f}")
    console.print(per_t)

    verdict = "ПОМОГЛО" if d > 0.003 else ("без изменений" if abs(d) <= 0.003 else "ХУЖЕ")
    console.print(f"\n[bold]Вывод:[/] logit adjustment {verdict} (delta {d:+.4f}, τ={best_tau})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "method": "logit_adjustment (Menon 2021)",
        "n_test": int(half), "n_val": int(len(y) - half),
        "best_tau": best_tau,
        "baseline_test_macro_f1": round(base_test_f1, 4),
        "improved_test_macro_f1": round(imp_test_f1, 4),
        "delta": round(d, 4),
        "val_curve": val_curve,
        "baseline_per_class": base_per,
        "improved_per_class": imp_per,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    console.log(f"[bold green]Saved[/] → {OUT}")


if __name__ == "__main__":
    main()
