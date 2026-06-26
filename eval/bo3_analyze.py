"""Сравнить: одиночный ответ (#1) vs best-of-3 (оператор берёт лучший из 3)."""
import json
from collections import Counter
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "eval" / "results"
ver = json.loads((R / "bo3_judge.json").read_text(encoding="utf-8"))["verdicts"]
RANK = {"good": 2, "weak": 1, "bad": 0}
NAME = {2: "good", 1: "weak", 0: "bad"}

single, bo3 = Counter(), Counter()
for v in ver:
    single[v.get("v1", "bad")] += 1
    best = max(RANK.get(v.get(k, "bad"), 0) for k in ("v1", "v2", "v3"))
    bo3[NAME[best]] += 1

n = len(ver) or 1
print("=== Лучшая версия (Mistral-24B + плейбук): #1 vs best-of-3, 87 злых ===\n")
for name, c in [("Одиночный (#1)", single), ("Best-of-3 (оператор выбирает)", bo3)]:
    print(f"{name:30s} good {c['good']/n:5.1%}  weak {c['weak']/n:5.1%}  bad {c['bad']/n:5.1%}  "
          f"(пригодны {(c['good']+c['weak'])/n:5.1%})")
print(f"\nbest-of-3: good = есть ≥1 хороший из 3; bad = ВСЕ 3 плохие (у оператора нет безопасного выбора)")
