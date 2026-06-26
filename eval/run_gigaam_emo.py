"""Сравнение GigaAM-Emo (Sber, SOTA-кандидат) против нашего HuBERT+logit-adjust."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import app  # noqa: F401,E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from datasets import Audio, load_dataset  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

import gigaam  # noqa: E402

CLASSES = ["neutral", "angry", "positive", "sad", "other"]
C2I = {c: i for i, c in enumerate(CLASSES)}
OUT = PROJECT_ROOT / "eval" / "results" / "ser_gigaam_emo.json"


def macro_f1(y_true, y_pred):  # noqa: ANN001
    yt, yp = np.array(y_true), np.array(y_pred)
    per, f1s = {}, []
    for ci, c in enumerate(CLASSES):
        tp = int(((yt == ci) & (yp == ci)).sum())
        fp = int(((yt != ci) & (yp == ci)).sum())
        fn = int(((yt == ci) & (yp != ci)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per[c] = {"precision": round(prec, 3), "recall": round(rec, 3),
                  "f1": round(f1, 3), "support": int(((yt == ci)).sum())}
        f1s.append(f1)
    return float(np.mean(f1s)), per


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    args = ap.parse_args()
    console = Console()
    console.rule(f"[bold cyan]GigaAM-Emo eval[/]  Dusha test (стрим, n={args.n})")

    console.log("Загружаю GigaAM-Emo (первый раз — скачка с Sber CDN)…")
    t0 = time.perf_counter()
    model = gigaam.load_model("emo")
    console.log(f"Модель готова за {time.perf_counter()-t0:.1f}s")
    console.log(f"GigaAM id2name: {getattr(model, 'id2name', '?')}")
    name2idx = {n.lower(): i for i, n in getattr(model, "id2name", {}).items()}
    console.log(f"маппинг GigaAM->Dusha порядок: {name2idx}")

    ds = load_dataset("xbgoose/dusha", split="test", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    y_true, y_pred = [], []
    confusion = Counter()
    skipped = 0
    latencies = []
    seen = 0
    t_start = time.perf_counter()
    tmp_dir = tempfile.mkdtemp(prefix="gigaam_eval_")
    tmp_path = Path(tmp_dir) / "in.wav"

    for item in ds:
        if seen >= args.n:
            break
        true_label = str(item["emotion"]).lower().strip()
        if true_label not in C2I:
            skipped += 1
            continue
        raw = item["audio"].get("bytes")
        if not raw:
            skipped += 1
            continue

        try:
            import io
            data, sr = sf.read(io.BytesIO(raw), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            sf.write(tmp_path, data, sr, subtype="PCM_16")
        except Exception as exc:  # noqa: BLE001
            console.log(f"[yellow]skip decode {exc}[/]")
            skipped += 1
            continue

        try:
            t_inf = time.perf_counter()
            probs = model.get_probs(str(tmp_path))
            latencies.append(int((time.perf_counter() - t_inf) * 1000))
        except Exception as exc:  # noqa: BLE001
            console.log(f"[yellow]skip predict {exc}[/]")
            skipped += 1
            continue

        pred_name = max(probs, key=probs.get).lower()
        if pred_name not in C2I:
            filtered = {k: v for k, v in probs.items() if k.lower() in C2I}
            if not filtered:
                skipped += 1
                continue
            pred_name = max(filtered, key=filtered.get).lower()

        y_true.append(C2I[true_label])
        y_pred.append(C2I[pred_name])
        confusion[(true_label, pred_name)] += 1
        seen += 1
        if seen % 50 == 0:
            run_f1, _ = macro_f1(y_true, y_pred)
            console.log(f"  {seen}/{args.n}  running macro-F1={run_f1:.3f}")

    elapsed = time.perf_counter() - t_start
    f1, per = macro_f1(y_true, y_pred)
    acc = sum(int(t == p) for t, p in zip(y_true, y_pred)) / max(len(y_true), 1)

    summ = Table(title="GigaAM-Emo на реальном Dusha test")
    summ.add_column("метрика", style="cyan")
    summ.add_column("значение", justify="right", style="green")
    summ.add_row("n_evaluated", str(len(y_true)))
    summ.add_row("skipped", str(skipped))
    summ.add_row("accuracy", f"{acc:.4f}")
    summ.add_row("f1_macro", f"{f1:.4f}")
    summ.add_row("latency_mean_ms", str(int(np.mean(latencies)) if latencies else 0))
    summ.add_row("latency_p95_ms", str(int(np.percentile(latencies, 95))
                                       if latencies else 0))
    summ.add_row("wall_sec", f"{elapsed:.0f}")
    console.print(summ)

    p_t = Table(title="Per-class")
    p_t.add_column("class", style="cyan")
    for col in ("precision", "recall", "f1", "support"):
        p_t.add_column(col, justify="right")
    for c in CLASSES:
        m = per[c]
        p_t.add_row(c, f"{m['precision']:.3f}", f"{m['recall']:.3f}",
                    f"{m['f1']:.3f}", str(m["support"]))
    console.print(p_t)

    c_t = Table(title="Confusion")
    c_t.add_column("true \\ pred", style="cyan")
    for c in CLASSES:
        c_t.add_column(c, justify="right")
    for t in CLASSES:
        row = [t] + [str(confusion.get((t, p), 0)) for p in CLASSES]
        c_t.add_row(*row)
    console.print(c_t)

    console.print(f"\n[bold]Сравнение:[/] GigaAM-Emo {f1:.4f}  vs  "
                  f"наш HuBERT+LA 0.799  vs  HuBERT argmax 0.776")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "n_evaluated": len(y_true), "n_skipped": skipped,
        "accuracy": round(acc, 4), "f1_macro": round(f1, 4),
        "latency_mean_ms": int(np.mean(latencies)) if latencies else 0,
        "latency_p95_ms": int(np.percentile(latencies, 95)) if latencies else 0,
        "wall_sec": int(elapsed),
        "per_class": per,
        "confusion": {f"{t}->{p}": c for (t, p), c in confusion.items()},
        "comparison": {
            "gigaam_emo": round(f1, 4),
            "ours_hubert_logit_adj": 0.799,
            "ours_hubert_argmax": 0.776,
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    console.log(f"[bold green]Saved[/] → {OUT}")


if __name__ == "__main__":
    main()
