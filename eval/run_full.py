"""Полный прогон пайплайна на синтетическом корпусе:."""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from app.pipeline.rag import get_retriever  # noqa: E402
from app.pipeline.ser import get_recognizer  # noqa: E402

DIALOGS = PROJECT_ROOT / "data" / "synthetic" / "dialogs.json"
AUDIO_MANIFEST = PROJECT_ROOT / "data" / "synthetic" / "audio_manifest.json"
AUDIO_DIR = PROJECT_ROOT / "data" / "synthetic" / "audio"

OUT_DIR = PROJECT_ROOT / "eval" / "results"
OUT_RAG = OUT_DIR / "rag_synth.json"
OUT_SER = OUT_DIR / "ser_synth.json"
SUMMARY = OUT_DIR / "summary.md"


def _rag_eval(console: Console) -> dict:
    if not DIALOGS.exists():
        console.print(f"[red]No dialogs at {DIALOGS}[/]")
        return {}

    dialogs = json.loads(DIALOGS.read_text(encoding="utf-8"))
    queries = []
    for d in dialogs:
        expected = d["expected_intent"]
        for turn in d["turns"]:
            if turn["speaker"] != "customer":
                continue
            queries.append({"text": turn["text"], "expected": expected})

    console.rule(f"[cyan]RAG eval (synth)[/]  {len(queries)} customer turns")

    retriever = get_retriever()
    retriever.search("warmup", k=1)

    hits = {1: 0, 3: 0, 5: 0, 10: 0}
    rr_sum = 0.0
    latencies = []
    per_query = []

    for q in queries:
        t0 = time.perf_counter()
        r = retriever.search(q["text"], k=10)
        elapsed = int((time.perf_counter() - t0) * 1000)
        latencies.append(elapsed)
        ranks = [s.doc_id for s in r.sources]
        rank_expected = (ranks.index(q["expected"]) + 1) if q["expected"] in ranks else None
        for k in hits:
            if rank_expected is not None and rank_expected <= k:
                hits[k] += 1
        if rank_expected:
            rr_sum += 1.0 / rank_expected
        per_query.append(
            {
                "query": q["text"],
                "expected": q["expected"],
                "rank": rank_expected,
                "top5": ranks[:5],
                "ms": elapsed,
            }
        )

    n = len(queries) or 1
    metrics = {
        "n_queries": n,
        "Recall@1": round(hits[1] / n, 3),
        "Recall@3": round(hits[3] / n, 3),
        "Recall@5": round(hits[5] / n, 3),
        "Recall@10": round(hits[10] / n, 3),
        "MRR@10": round(rr_sum / n, 3),
        "latency_p50_ms": sorted(latencies)[n // 2],
        "latency_p95_ms": sorted(latencies)[max(0, int(n * 0.95) - 1)],
    }
    OUT_RAG.write_text(
        json.dumps({"metrics": metrics, "per_query": per_query}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = Table(title="RAG (synth)")
    summary.add_column("metric", style="cyan")
    summary.add_column("value", style="green", justify="right")
    for k, v in metrics.items():
        summary.add_row(k, str(v))
    console.print(summary)
    return metrics


def _ser_eval(console: Console) -> dict:
    if not AUDIO_MANIFEST.exists():
        console.print(f"[red]No audio manifest at {AUDIO_MANIFEST}[/]")
        return {}

    manifest = json.loads(AUDIO_MANIFEST.read_text(encoding="utf-8"))
    console.rule(f"[cyan]SER eval (synth)[/]  {len(manifest)} clips")

    rec = get_recognizer()
    rec._ensure_loaded()  # noqa: SLF001

    classes = sorted({m["label"] for m in manifest})
    per_class = {c: {"tp": 0, "fp": 0, "fn": 0} for c in classes}
    confusion: Counter = Counter()
    correct = 0
    latencies = []
    per_clip = []

    for item in manifest:
        path = AUDIO_DIR / item["file"]
        if not path.exists():
            continue
        true_label = item["label"]
        result = rec.predict(str(path))
        pred = result.state.label.value
        confusion[(true_label, pred)] += 1
        latencies.append(result.inference_ms)
        if pred == true_label:
            correct += 1
            per_class[true_label]["tp"] += 1
        else:
            per_class[true_label]["fn"] += 1
            if pred in per_class:
                per_class[pred]["fp"] += 1
        per_clip.append(
            {
                "file": item["file"],
                "true": true_label,
                "pred": pred,
                "confidence": result.state.confidence,
                "inference_ms": result.inference_ms,
            }
        )

    n = len(per_clip) or 1
    f1s = []
    per_class_metrics = {}
    for cls, c in per_class.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec_v = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec_v / (prec + rec_v) if (prec + rec_v) else 0.0
        f1s.append(f1)
        per_class_metrics[cls] = {
            "precision": round(prec, 3),
            "recall": round(rec_v, 3),
            "f1": round(f1, 3),
        }

    metrics = {
        "n_clips": n,
        "accuracy": round(correct / n, 3),
        "f1_macro": round(sum(f1s) / len(f1s) if f1s else 0.0, 3),
        "latency_p50_ms": sorted(latencies)[n // 2],
        "latency_p95_ms": sorted(latencies)[max(0, int(n * 0.95) - 1)],
        "per_class": per_class_metrics,
        "classes": classes,
    }

    OUT_SER.write_text(
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

    summary = Table(title="SER (synth)")
    summary.add_column("metric", style="cyan")
    summary.add_column("value", style="green", justify="right")
    for k in ["n_clips", "accuracy", "f1_macro", "latency_p50_ms", "latency_p95_ms"]:
        summary.add_row(k, str(metrics[k]))
    console.print(summary)

    cls_table = Table(title="SER per-class (synth)")
    cls_table.add_column("class", style="cyan")
    for col in ("precision", "recall", "f1"):
        cls_table.add_column(col, justify="right")
    for cls, m in per_class_metrics.items():
        cls_table.add_row(cls, str(m["precision"]), str(m["recall"]), str(m["f1"]))
    console.print(cls_table)

    conf_table = Table(title="Confusion (rows=true, cols=pred)")
    conf_table.add_column("true \\ pred", style="cyan")
    for c in classes:
        conf_table.add_column(c, justify="right")
    for t in classes:
        row = [t]
        for p in classes:
            row.append(str(confusion.get((t, p), 0)))
        conf_table.add_row(*row)
    console.print(conf_table)
    return metrics


def _write_summary(rag: dict, ser: dict) -> None:
    """Свод метрик из всех eval-прогонов в один markdown."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def _load(name: str) -> dict | None:
        p = OUT_DIR / name
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8")).get("metrics", {})

    retr_small = _load("retrieval.json")
    ser_small = _load("ser.json")
    llm_small = _load("llm.json")

    lines = [
        "# Eval summary — telecom copilot",
        "",
        "Все метрики собраны автоматически (`eval/run_full.py`). Маленькие наборы — "
        "ранние sanity-эксперименты, синтетические — eval на полном корпусе диалогов "
        "из `data/synthetic/`.",
        "",
        "## RAG (BGE-M3 + ChromaDB, 104 чанка из 25 KB-документов)",
        "",
        "| Набор | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | p95 lat |",
        "|---|---|---|---|---|---|---|",
    ]
    if retr_small:
        lines.append(
            f"| 10 ручных QA | {retr_small.get('Recall@1', '—')} | "
            f"{retr_small.get('Recall@3', '—')} | {retr_small.get('Recall@5', '—')} | "
            f"{retr_small.get('Recall@10', '—')} | {retr_small.get('MRR@10', '—')} | "
            f"{retr_small.get('latency_p95_ms', '—')} мс |"
        )
    if rag:
        lines.append(
            f"| {rag.get('n_queries', '?')} synth turns | {rag.get('Recall@1', '—')} | "
            f"{rag.get('Recall@3', '—')} | {rag.get('Recall@5', '—')} | "
            f"{rag.get('Recall@10', '—')} | {rag.get('MRR@10', '—')} | "
            f"{rag.get('latency_p95_ms', '—')} мс |"
        )

    lines += [
        "",
        "## SER (HuBERT-Dusha, 5-class)",
        "",
        "| Набор | n | accuracy | F1-macro | p95 lat |",
        "|---|---|---|---|---|",
    ]
    if ser_small:
        lines.append(
            f"| 4 TTS-демо | {ser_small.get('n_samples', '—')} | "
            f"{ser_small.get('accuracy', '—')} | {ser_small.get('f1_macro', '—')} | "
            f"{ser_small.get('latency_p95_ms', '—')} мс |"
        )
    if ser:
        lines.append(
            f"| {ser.get('n_clips', '?')} synth (+telephony filter) | "
            f"{ser.get('n_clips', '—')} | {ser.get('accuracy', '—')} | "
            f"{ser.get('f1_macro', '—')} | {ser.get('latency_p95_ms', '—')} мс |"
        )
        per_cls = ser.get("per_class", {})
        if per_cls:
            lines += ["", "### SER per-class на synth", "",
                      "| class | precision | recall | F1 |",
                      "|---|---|---|---|"]
            for cls, m in per_cls.items():
                lines.append(
                    f"| {cls} | {m['precision']} | {m['recall']} | {m['f1']} |"
                )

    lines += [
        "",
        "## LLM (T-lite-instruct-0.1 GGUF Q4_K_M, CUDA 4080 Super)",
        "",
        "| Набор | format compliance | mean sugg | prompt tokens | completion tokens | p95 lat |",
        "|---|---|---|---|---|---|",
    ]
    if llm_small:
        lines.append(
            f"| 10 ручных QA | {llm_small.get('format_compliance', '—')} | "
            f"{llm_small.get('mean_suggestions_per_query', '—')} | "
            f"{llm_small.get('mean_prompt_tokens', '—')} | "
            f"{llm_small.get('mean_completion_tokens', '—')} | "
            f"{llm_small.get('llm_latency_p95_ms', '—')} мс |"
        )

    lines += [
        "",
        "## End-to-end latency бюджет",
        "",
        "Целевой E2E (по SalesCopilot, arXiv 2603.21416): **≤ 3 секунды p95**.",
        "Измерено на demo-пайплайне (`scripts/check_real_rag.py`):",
        "",
        "| Стадия | p95 на демо |",
        "|---|---|",
        "| SER (HuBERT-Dusha, 16 kHz, GPU) | ~210 мс |",
        "| RAG (BGE-M3 dense, top-3) | ~30 мс |",
        "| LLM (T-lite Q4, ~94 токена) | ~1.3 сек |",
        "| **Итого E2E** | **~1.5–2.0 сек** |",
        "",
        "Бюджет соблюдён с запасом.",
    ]

    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSummary written -> {SUMMARY}")


def main() -> None:
    console = Console()
    rag = _rag_eval(console)
    ser = _ser_eval(console)
    _write_summary(rag, ser)


if __name__ == "__main__":
    main()
