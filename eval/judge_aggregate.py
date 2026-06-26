import json
from collections import Counter
from pathlib import Path

R = Path(r"K:\dev\coursework\eval\results")
d = json.loads((R / "e2e_judge_full_vs_noemotion.json").read_text(encoding="utf-8"))
v = d["verdicts"]


def agg(subset, label):
    c = Counter(x["winner"] for x in subset)
    n = len(subset)
    a, b, t = c.get("A", 0), c.get("B", 0), c.get("tie", 0)
    decisive = a + b
    wr = a / decisive if decisive else 0.0
    print(f"{label:18s} n={n:2d}  full(A)={a:2d}  noemo(B)={b:2d}  tie={t:2d}  "
          f"full win-rate (без ничьих)={wr:.0%}")


print("=== Pairwise: full (RAG+эмоция) vs no_emotion ===")
agg([x for x in v if x["emotion"] in ("angry", "sad")], "негатив (angry+sad)")
agg([x for x in v if x["emotion"] == "neutral"], "neutral")
agg(v, "всего")
print("\nГипотеза: эмоц-блок помогает на негативе, нейтрален на neutral.")
print("Негатив-кейсы где B победил — отметить (честность):")
for x in v:
    if x["winner"] == "B" and x["emotion"] in ("angry", "sad"):
        print(f"  {x['id']} ({x['emotion']}): {x['why']}")
print("\nГаллюцинации full (важно для записки):")
for x in v:
    if "ГАЛЛЮЦИН" in x["why"].upper():
        print(f"  {x['id']}: {x['why']}")
