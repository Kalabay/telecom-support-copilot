"""Связка ВНУТРИ best-of-3: 3 смелых плейбук-варианта + безопасный MoE как опция."""
import json
from collections import Counter
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "eval" / "results"
RANK = {"good": 2, "weak": 1, "bad": 0}
NAME = {2: "good", 1: "weak", 0: "bad"}

ver = json.loads((R / "bo3_judge.json").read_text(encoding="utf-8"))["verdicts"]

bo3, ens = Counter(), Counter()
rescued = 0
for v in ver:
    best3 = max(RANK.get(v.get(c, "bad"), 0) for c in ("v1", "v2", "v3"))
    bo3[NAME[best3]] += 1
    if best3 == 0:
        ens["weak"] += 1
        rescued += 1
    else:
        ens[NAME[best3]] += 1

n = len(ver) or 1
print("=== best-of-3 плейбук, без MoE-опции и С безопасной MoE-опцией (87 злых) ===\n")
for name, c in [("Best-of-3 (3 варианта)", bo3), ("Best-of-3 + безопасный MoE", ens)]:
    print(f"{name:30s} good {c['good']/n:5.1%}  weak {c['weak']/n:5.1%}  bad {c['bad']/n:5.1%}  "
          f"(usable {(c['good']+c['weak'])/n:5.1%})")
print(f"\nреплик, где все 3 смелых плохи -> спас безопасный MoE: {rescued}")
print("допущение: MoE bad=0 (подтверждено отдельным замером MoE)")
