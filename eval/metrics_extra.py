"""Доп. метрики сравнения моделей (кроме судьи): длина, риск-частота, разнообразие best-of-3."""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.pipeline.llm import _risk  # noqa: E402

R = ROOT / "eval" / "results"
MODELS = [
    ("mistral24", "Mistral-24B"),
    ("tlite21", "T-lite 8B"),
    ("vikhr_nemo12", "Vikhr-12B"),
    ("gigachat20_v15", "GigaChat-20B"),
    ("qwen3moe", "Qwen3-MoE"),
    ("ruadapt32", "Ruadapt-32B"),
    ("qwen3_14b", "Qwen3-14B"),
]
kb = {(r["dialogue_id"], r["turn_idx"]): r["doc"]
      for r in json.loads((R / "emo_input.json").read_text(encoding="utf-8"))}


def load(name):
    p = R / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def jacc(a, b):
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


print(f"{'Модель':14s} {'слов/ответ':>11s} {'риск≥2':>9s} {'разнообр.bo3':>13s}")
print("-" * 52)
for key, name in MODELS:
    bench = load(f"bench_{key}.json")
    if not bench:
        continue
    words = mean(len(x["filtered"].split()) for x in bench)
    risk_rate = mean(_risk(x["raw"], kb.get((x["dialogue_id"], x["turn_idx"]), "")) >= 2 for x in bench)
    bo3 = load(f"bo3_{key}.json")
    if bo3:
        div = mean(1 - mean([jacc(x["v1"], x["v2"]), jacc(x["v1"], x["v3"]), jacc(x["v2"], x["v3"])])
                   for x in bo3)
        divs = f"{div:.2f}"
    else:
        divs = "—"
    print(f"{name:14s} {words:>11.0f} {risk_rate:>8.0%} {divs:>13s}")
