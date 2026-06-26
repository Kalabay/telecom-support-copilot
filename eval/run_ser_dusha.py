"""Честный eval SER на РЕАЛЬНОМ Dusha test split (а не на TTS-синтетике)."""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import app  # noqa: F401,E402  (bootstrap .env → HF_HOME на K:)

from datasets import Audio, load_dataset  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from app.pipeline.ser import get_recognizer  # noqa: E402

OUT = PROJECT_ROOT / "eval" / "results" / "ser_dusha_real.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="сколько тест-сэмплов прогнать")
    args = ap.parse_args()

    console = Console()
    console.rule(f"[bold cyan]SER eval — РЕАЛЬНЫЙ Dusha test[/]  (стрим, n={args.n})")

    rec = get_recognizer()
    console.log("Загружаю SER-модель…")
    rec._ensure_loaded()  # noqa: SLF001

    console.log("Открываю стрим xbgoose/dusha:test …")
    ds = load_dataset("xbgoose/dusha", split="test", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    classes = ["neutral", "angry", "positive", "sad", "other"]
    per_class = {c: {"tp": 0, "fp": 0, "fn": 0} for c in classes}
    confusion: Counter = Counter()
    correct = 0
    total = 0
    latencies: list[int] = []
    skipped = 0

    t_start = time.perf_counter()
    for item in ds:
        if total >= args.n:
            break
        true_label = str(item["emotion"]).lower().strip()
        if true_label not in per_class:
            skipped += 1
            continue

        audio = item["audio"]
        raw = audio.get("bytes")
        if raw is None and audio.get("path"):
            raw = Path(audio["path"]).read_bytes()
        if not raw:
            skipped += 1
            continue

        result = rec.predict(raw)
        pred = result.state.label.value
        latencies.append(result.inference_ms)

        confusion[(true_label, pred)] += 1
        if pred == true_label:
            correct += 1
            per_class[true_label]["tp"] += 1
        else:
            per_class[true_label]["fn"] += 1
            if pred in per_class:
                per_class[pred]["fp"] += 1

        total += 1
        if total % 50 == 0:
            acc = correct / total
            console.log(f"  {total} сэмплов · running acc={acc:.3f}")

    elapsed = time.perf_counter() - t_start

    f1s = []
    per_class_metrics = {}
    for c in classes:
        tp, fp, fn = per_class[c]["tp"], per_class[c]["fp"], per_class[c]["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        recl = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * recl / (prec + recl) if (prec + recl) else 0.0
        support = tp + fn
        f1s.append(f1)
        per_class_metrics[c] = {
            "precision": round(prec, 3),
            "recall": round(recl, 3),
            "f1": round(f1, 3),
            "support": support,
        }

    metrics = {
        "n_evaluated": total,
        "n_skipped": skipped,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "f1_macro": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "latency_mean_ms": int(sum(latencies) / len(latencies)) if latencies else 0,
        "latency_p95_ms": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)]
        if latencies else 0,
        "wall_sec": round(elapsed, 1),
    }

    summ = Table(title="Итог (реальный Dusha test)")
    summ.add_column("метрика", style="cyan")
    summ.add_column("значение", style="green", justify="right")
    for k in ["n_evaluated", "accuracy", "f1_macro", "latency_mean_ms", "latency_p95_ms"]:
        summ.add_row(k, str(metrics[k]))
    console.print(summ)

    cls_t = Table(title="Per-class")
    cls_t.add_column("класс", style="cyan")
    for col in ("precision", "recall", "f1", "support"):
        cls_t.add_column(col, justify="right")
    for c in classes:
        m = per_class_metrics[c]
        cls_t.add_row(c, str(m["precision"]), str(m["recall"]), str(m["f1"]), str(m["support"]))
    console.print(cls_t)

    conf_t = Table(title="Confusion (строки=true, столбцы=pred)")
    conf_t.add_column("true \\ pred", style="cyan")
    for c in classes:
        conf_t.add_column(c, justify="right")
    for t in classes:
        row = [t] + [str(confusion.get((t, p), 0)) for p in classes]
        conf_t.add_row(*row)
    console.print(conf_t)

    console.print(
        f"\n[bold]Сравнение:[/] реальный Dusha [green]{metrics['f1_macro']}[/] "
        f"F1-macro  vs  TTS-синтетика 0.222  vs  карточка модели ~0.81"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    import json
    OUT.write_text(
        json.dumps(
            {
                "metrics": metrics,
                "per_class": per_class_metrics,
                "confusion": {f"{t}->{p}": c for (t, p), c in confusion.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    console.log(f"[bold green]Saved[/] → {OUT}")


if __name__ == "__main__":
    main()
