"""Агрегат pairwise (фронтир) на 150: full vs no_emotion по эмоциям."""
import json
from collections import Counter
from pathlib import Path

R = Path(r"K:\dev\coursework\eval\results")
v = json.loads((R / "e2e_judge_full_vs_noemotion_150.json").read_text(encoding="utf-8"))["verdicts"]


def agg(sub, label):
    c = Counter(x["winner"] for x in sub)
    a, b, t = c.get("A", 0), c.get("B", 0), c.get("tie", 0)
    dec = a + b
    wr = a / dec if dec else 0.0
    print(f"{label:20s} n={len(sub):3d}  full(A)={a:2d}  noemo(B)={b:2d}  tie={t:2d}  "
          f"win-rate full(без ничьих)={wr:.0%}")


print("=== Pairwise full vs no_emotion (150 кейсов, фронтир-судья) ===")
agg([x for x in v if x["emotion"] in ("angry", "sad")], "негатив (angry+sad)")
agg([x for x in v if x["emotion"] == "angry"], "  angry")
agg([x for x in v if x["emotion"] == "sad"], "  sad")
agg([x for x in v if x["emotion"] == "neutral"], "neutral")
agg(v, "всего")
print("\nПримечание: проигрыши full на негативе — почти все из-за переобещаний/"
      "выдуманных бонусов (эмоц-блок провоцирует «приятные» обещания вне KB).")
