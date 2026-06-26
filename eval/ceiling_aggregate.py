import json
from collections import Counter
from pathlib import Path

R = Path(r"K:\dev\coursework\eval\results")
d = json.loads((R / "e2e_ceiling_tlite_vs_claude.json").read_text(encoding="utf-8"))
v = d["verdicts"]
c = Counter(x["winner"] for x in v)
n = len(v)
tlite, claude, tie = c.get("tlite", 0), c.get("claude", 0), c.get("tie", 0)

print("=== Фронтир-потолок: T-lite (локально, $0) vs Claude (фронтир) ===")
print(f"всего кейсов: {n}")
print(f"  ничья (T-lite не хуже фронтира): {tie}  ({tie/n:.0%})")
print(f"  Claude лучше:                    {claude}  ({claude/n:.0%})")
print(f"  T-lite лучше:                    {tlite}  ({tlite/n:.0%})")
print(f"\n>>> T-lite на уровне фронтира (ничья+победа): {(tie+tlite)/n:.0%} кейсов")
print(f">>> Claude превзошёл локальную: {claude/n:.0%} кейсов\n")

print("Где Claude победил и ПОЧЕМУ (ценно для записки — слабые места локальной):")
for x in v:
    if x["winner"] == "claude":
        tag = "ГАЛЛЮЦ" if "ГАЛЛЮЦ" in x["why"].upper() else "факт"
        print(f"  {x['id']} ({x['emotion']}, {tag}): {x['why']}")

print("\nПо типу эмоции (ничья = паритет с фронтиром):")
for emo in ("neutral", "angry", "sad"):
    sub = [x for x in v if x["emotion"] == emo]
    cc = Counter(x["winner"] for x in sub)
    print(f"  {emo:8s} n={len(sub):2d}  tie={cc.get('tie',0)}  claude={cc.get('claude',0)}  tlite={cc.get('tlite',0)}")
