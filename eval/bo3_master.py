"""Итоговая best-of-3 таблица по всем моделям."""
import json
from collections import Counter
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "eval" / "results"
RANK = {"good": 2, "weak": 1, "bad": 0}
NAME = {2: "good", 1: "weak", 0: "bad"}
MODELS = ["mistral24", "gemma27", "qwen3_14b", "ruadapt32", "qwen3moe",
          "gigachat20_v15", "vikhr_nemo12", "tlite21"]
TITLE = {"mistral24": "Mistral-24B", "gemma27": "gemma-27b", "qwen3_14b": "Qwen3-14B",
         "ruadapt32": "Ruadapt-32B", "qwen3moe": "Qwen3-MoE", "gigachat20_v15": "GigaChat-20B",
         "vikhr_nemo12": "Vikhr-12B", "tlite21": "T-lite-2.1"}


def pct(c, n):
    return {k: c[k] / n for k in ("good", "weak", "bad")}


print(f"{'Модель':14s} | {'одиночный #1':^18s} | {'best-of-3':^18s} | {'bo3+MoE':^18s}")
print(f"{'':14s} | {'good / bad':^18s} | {'good / bad':^18s} | {'good / bad':^18s}")
print("-" * 78)
for m in MODELS:
    jf = R / f"bo3_{m}_judge.json"
    if not jf.exists():
        print(f"{TITLE.get(m,m):14s} | (нет судьи)")
        continue
    ver = json.loads(jf.read_text(encoding="utf-8"))["verdicts"]
    n = len(ver) or 1
    single, bo3, ens = Counter(), Counter(), Counter()
    for v in ver:
        single[v.get("v1", "bad")] += 1
        best = max(RANK.get(v.get(c, "bad"), 0) for c in ("v1", "v2", "v3"))
        bo3[NAME[best]] += 1
        ens[NAME[best] if best > 0 else "weak"] += 1
    s, b, e = pct(single, n), pct(bo3, n), pct(ens, n)
    print(f"{TITLE.get(m,m):14s} | {s['good']:5.1%} / {s['bad']:5.1%}     | "
          f"{b['good']:5.1%} / {b['bad']:5.1%}     | {e['good']:5.1%} / {e['bad']:5.1%}")
