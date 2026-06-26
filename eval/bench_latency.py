"""Сравнение скорости генерации моделей по bench_*.json (мс на ответ + токены/с)."""
import json
import statistics as st
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "eval" / "results"
LABEL = {"tlite21": "T-lite-it-2.1 (8B)", "qwen3_14b": "Qwen3-14B",
         "vikhr_nemo12": "Vikhr-Nemo-12B", "gigachat20_v15": "GigaChat-20B-A3B v1.5"}

rows_by = {}
for p in sorted(R.glob("bench_*.json")):
    name = p.stem.replace("bench_", "")
    rows = json.loads(p.read_text(encoding="utf-8"))
    if rows and "ms" in rows[0]:
        rows_by[name] = rows

base_med = None
print("=== Скорость генерации (87 злых реплик) ===\n")
md = ["# Скорость генерации моделей (87 реплик)\n",
      "| Модель | медиана, с | среднее, с | p90, с | ток/ответ | ток/с | vs T-lite |",
      "|---|---|---|---|---|---|---|"]
for name in sorted(rows_by, key=lambda n: 0 if n == "tlite21" else 1):
    rows = rows_by[name]
    ms = [r["ms"] for r in rows]
    toks = [r.get("completion_tokens", 0) for r in rows]
    med, avg, p90 = st.median(ms), st.mean(ms), sorted(ms)[int(len(ms) * 0.9)]
    tps = (sum(toks) / (sum(ms) / 1000)) if sum(ms) else 0
    if name == "tlite21":
        base_med = med
    ratio = f"{med/base_med:.2f}x" if base_med else "—"
    print(f"{LABEL.get(name, name):26s} медиана {med/1000:5.2f}с  среднее {avg/1000:5.2f}с  "
          f"p90 {p90/1000:5.2f}с  {st.mean(toks):4.0f} ток  {tps:4.0f} ток/с  ({ratio})")
    md.append(f"| {LABEL.get(name, name)} | {med/1000:.2f} | {avg/1000:.2f} | {p90/1000:.2f} | "
              f"{st.mean(toks):.0f} | {tps:.0f} | {ratio} |")

(R / "latency_result.md").write_text("\n".join(md), encoding="utf-8")
print("\nsaved -> eval/results/latency_result.md")
