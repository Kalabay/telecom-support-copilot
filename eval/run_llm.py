"""Прогон LLM по тем же 10 QA-парам из qa_pairs.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from app.models.schemas import Emotion, EmotionState  # noqa: E402
from app.pipeline.llm import get_generator  # noqa: E402
from app.pipeline.rag import get_retriever  # noqa: E402

QA_FILE = PROJECT_ROOT / "eval" / "qa_pairs.json"
OUT_FILE = PROJECT_ROOT / "eval" / "results" / "llm.json"

ANGRY_KEYWORDS = ("плачу", "расторгн", "верните", "ухожу", "надоело", "сколько можно")


def _emotion_for(query: str) -> EmotionState:
    is_angry = any(k in query.lower() for k in ANGRY_KEYWORDS)
    if is_angry:
        return EmotionState(
            label=Emotion.ANGRY,
            confidence=0.78,
            arousal=0.82,
            valence=-0.70,
            escalation_risk=True,
        )
    return EmotionState(
        label=Emotion.NEUTRAL,
        confidence=0.92,
        arousal=0.30,
        valence=0.0,
        escalation_risk=False,
    )


def main() -> None:
    console = Console()
    items = json.loads(QA_FILE.read_text(encoding="utf-8"))["items"]
    console.rule(f"[bold cyan]LLM eval[/]  {len(items)} queries")

    retriever = get_retriever()
    gen = get_generator()
    console.log("Warming up retriever + LLM…")
    retriever.search("warmup", k=1)

    per_query = []
    latencies = []
    format_ok = 0

    for item in items:
        query = item["query"]
        emotion = _emotion_for(query)
        rag = retriever.search(query, k=3)
        result = gen.generate([query], emotion, rag.sources, max_tokens=220)
        n_sugg = len(result.suggestions)
        if n_sugg >= 1:
            format_ok += 1
        latencies.append(result.total_ms)
        per_query.append(
            {
                "id": item["id"],
                "query": query,
                "emotion": emotion.label.value,
                "rag_top1": rag.sources[0].doc_id if rag.sources else None,
                "n_suggestions": n_sugg,
                "suggestions": result.suggestions,
                "raw": result.raw_completion[:400],
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "llm_ms": result.total_ms,
            }
        )

    n = len(items)
    table = Table(title="Per-query (suggestions truncated)")
    table.add_column("id", style="cyan")
    table.add_column("emotion", style="magenta")
    table.add_column("n", justify="right")
    table.add_column("first suggestion (60 chars)")
    table.add_column("ms", justify="right")
    for r in per_query:
        first = (r["suggestions"][0] if r["suggestions"] else "—")[:60] + (
            "…" if r["suggestions"] and len(r["suggestions"][0]) > 60 else ""
        )
        table.add_row(r["id"], r["emotion"], str(r["n_suggestions"]), first, str(r["llm_ms"]))
    console.print(table)

    metrics = {
        "format_compliance": round(format_ok / n, 3),
        "mean_suggestions_per_query": round(
            sum(r["n_suggestions"] for r in per_query) / n, 2
        ),
        "llm_latency_p50_ms": sorted(latencies)[n // 2],
        "llm_latency_p95_ms": sorted(latencies)[max(0, int(n * 0.95) - 1)],
        "llm_latency_mean_ms": int(sum(latencies) / n),
        "mean_prompt_tokens": int(sum(r["prompt_tokens"] for r in per_query) / n),
        "mean_completion_tokens": int(sum(r["completion_tokens"] for r in per_query) / n),
        "n_queries": n,
    }

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
