"""Прогон retrieval-модуля по eval/qa_pairs.json: Recall@k и MRR@10."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from app.pipeline.rag import get_retriever  # noqa: E402

QA_FILE = PROJECT_ROOT / "eval" / "qa_pairs.json"
OUT_FILE = PROJECT_ROOT / "eval" / "results" / "retrieval.json"


def main() -> None:
    console = Console()
    data = json.loads(QA_FILE.read_text(encoding="utf-8"))
    items = data["items"]
    console.rule(f"[bold cyan]Retrieval eval[/]  {len(items)} QA pairs")

    retriever = get_retriever()
    console.log("Warming up retriever…")
    retriever.search("прогрев", k=1)

    per_query: list[dict] = []
    recall_at_k = {1: 0, 3: 0, 5: 0, 10: 0}
    rr_sum = 0.0
    latencies_ms: list[int] = []

    for item in items:
        query = item["query"]
        expected = item["expected_doc_id"]
        alt = set(item.get("alt_doc_ids", []))
        relevant: set[str] = {expected} | alt

        t0 = time.perf_counter()
        result = retriever.search(query, k=10)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        latencies_ms.append(elapsed_ms)

        ranks = [s.doc_id for s in result.sources]
        rank_expected = ranks.index(expected) + 1 if expected in ranks else None
        rank_any_relevant = next(
            (i + 1 for i, d in enumerate(ranks) if d in relevant), None
        )

        for k in recall_at_k:
            if rank_any_relevant is not None and rank_any_relevant <= k:
                recall_at_k[k] += 1
        if rank_expected is not None:
            rr_sum += 1.0 / rank_expected

        per_query.append(
            {
                "id": item["id"],
                "query": query,
                "expected": expected,
                "alt": item.get("alt_doc_ids", []),
                "top10": ranks,
                "rank_expected": rank_expected,
                "rank_any_relevant": rank_any_relevant,
                "latency_ms": elapsed_ms,
                "top1_score": result.sources[0].score if result.sources else None,
            }
        )

    n = len(items)
    metrics = {
        f"Recall@{k}": round(c / n, 3) for k, c in recall_at_k.items()
    }
    metrics["MRR@10"] = round(rr_sum / n, 3)
    metrics["latency_p50_ms"] = sorted(latencies_ms)[n // 2]
    metrics["latency_p95_ms"] = sorted(latencies_ms)[max(0, int(n * 0.95) - 1)]
    metrics["latency_mean_ms"] = int(sum(latencies_ms) / n)
    metrics["n_queries"] = n

    table = Table(title="Per-query results")
    table.add_column("id", style="cyan")
    table.add_column("expected", style="green")
    table.add_column("rank", justify="right")
    table.add_column("top1", style="magenta")
    table.add_column("ms", justify="right")
    for r in per_query:
        rank_display = str(r["rank_expected"]) if r["rank_expected"] else "—"
        top1 = r["top10"][0] if r["top10"] else "—"
        table.add_row(r["id"], r["expected"], rank_display, top1, str(r["latency_ms"]))
    console.print(table)

    summary = Table(title="Summary")
    summary.add_column("metric", style="cyan")
    summary.add_column("value", style="green", justify="right")
    for k, v in metrics.items():
        summary.add_row(k, str(v))
    console.print(summary)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(
        json.dumps(
            {"metrics": metrics, "per_query": per_query},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    console.log(f"[bold green]Saved[/] → {OUT_FILE}")


if __name__ == "__main__":
    main()
