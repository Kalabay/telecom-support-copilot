"""Прогон SER-модуля по data/audio_samples/manifest.json: accuracy + F1-macro + confusion."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from app.pipeline.ser import get_recognizer  # noqa: E402

import argparse  # noqa: E402

DEFAULT_AUDIO_DIR = PROJECT_ROOT / "data" / "audio_samples"
DEFAULT_MANIFEST = DEFAULT_AUDIO_DIR / "manifest.json"
DEFAULT_OUT = PROJECT_ROOT / "eval" / "results" / "ser.json"


def f1_macro(per_class: dict[str, dict[str, int]]) -> float:
    f1s = []
    for cls, c in per_class.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="SER evaluation")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="path to audio manifest.json")
    parser.add_argument("--audio-dir", type=Path, default=None,
                        help="directory with wav files (default = manifest's parent)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="results json output path")
    args = parser.parse_args()

    manifest_path = args.manifest
    audio_dir = args.audio_dir or manifest_path.parent
    out_file = args.out

    console = Console()
    if not manifest_path.exists():
        console.print(
            f"[red]Manifest not found at {manifest_path}[/]\n"
            "Generate it first (scripts/fetch_test_audio.py or scripts/synth_audio.py)."
        )
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    console.rule(f"[bold cyan]SER eval[/]  {len(manifest)} clips from {manifest_path.name}")

    rec = get_recognizer()
    console.log("Warming up SER (first inference loads weights)…")
    rec._ensure_loaded()  # noqa: SLF001

    classes = sorted({m["label"] for m in manifest})
    per_class: dict[str, dict[str, int]] = {c: {"tp": 0, "fp": 0, "fn": 0} for c in classes}
    confusion: dict[tuple[str, str], int] = Counter()
    correct = 0
    per_clip: list[dict] = []

    table = Table(title="Per-clip predictions")
    table.add_column("file", style="cyan")
    table.add_column("true", style="green")
    table.add_column("pred", style="magenta")
    table.add_column("conf", justify="right")
    table.add_column("ms", justify="right")
    table.add_column("dur ms", justify="right")

    for item in manifest:
        audio_path = audio_dir / item["file"]
        true_label = item["label"]
        result = rec.predict(str(audio_path))
        pred_label = result.state.label.value
        confidence = result.state.confidence

        confusion[(true_label, pred_label)] += 1
        if pred_label == true_label:
            correct += 1
            per_class[true_label]["tp"] += 1
        else:
            per_class[true_label]["fn"] += 1
            if pred_label in per_class:
                per_class[pred_label]["fp"] += 1

        per_clip.append(
            {
                "file": item["file"],
                "true": true_label,
                "pred": pred_label,
                "confidence": confidence,
                "probs": result.probs,
                "inference_ms": result.inference_ms,
                "duration_ms": result.duration_ms,
            }
        )
        table.add_row(
            item["file"],
            true_label,
            pred_label,
            f"{confidence:.2f}",
            str(result.inference_ms),
            str(result.duration_ms),
        )

    console.print(table)

    conf_table = Table(title="Confusion (rows=true, cols=pred)")
    conf_table.add_column("true \\ pred", style="cyan")
    for c in classes:
        conf_table.add_column(c, justify="right")
    for true in classes:
        row = [true]
        for pred in classes:
            row.append(str(confusion.get((true, pred), 0)))
        conf_table.add_row(*row)
    console.print(conf_table)

    metrics = {
        "accuracy": round(correct / len(manifest), 3),
        "f1_macro": round(f1_macro(per_class), 3),
        "n_samples": len(manifest),
        "classes": classes,
        "latency_mean_ms": int(sum(c["inference_ms"] for c in per_clip) / len(per_clip)),
        "latency_p95_ms": sorted(c["inference_ms"] for c in per_clip)[
            max(0, int(len(per_clip) * 0.95) - 1)
        ],
    }

    summary = Table(title="Summary")
    summary.add_column("metric", style="cyan")
    summary.add_column("value", style="green", justify="right")
    for k, v in metrics.items():
        summary.add_row(k, str(v))
    console.print(summary)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        json.dumps(
            {
                "metrics": metrics,
                "per_clip": per_clip,
                "confusion": {f"{t}->{p}": c for (t, p), c in confusion.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    console.log(f"[bold green]Saved[/] → {out_file}")


if __name__ == "__main__":
    main()
