"""Развернуть попарный судья эмоция-абляции: с эмоцией vs без."""
import json
from collections import Counter
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "eval" / "results"
ver = json.loads((R / "emo_judge.json").read_text(encoding="utf-8"))["verdicts"]
mp = {(m["dialogue_id"], m["turn_idx"]): m
      for m in json.loads((R / "emo_map.json").read_text(encoding="utf-8"))}
c = Counter()
for v in ver:
    m = mp[(v["dialogue_id"], v["turn_idx"])]
    w = v.get("winner")
    if w == "tie":
        c["tie"] += 1
    elif w in ("A", "B"):
        c[m[w]] += 1
n = sum(c.values()) or 1
print("=== Эмоция-абляция на лучшем пайплайне (плейбук+Mistral, 87 злых) ===\n")
print(f"  С эмоцией лучше:   {c['with']:3d}  ({c['with']/n:.0%})")
print(f"  Без эмоции лучше:  {c['without']:3d}  ({c['without']/n:.0%})")
print(f"  Ничья:             {c['tie']:3d}  ({c['tie']/n:.0%})")
wins = c['with'] + c['without']
if wins:
    print(f"\n  Среди небезничейных: с эмоцией {c['with']/wins:.0%}, без {c['without']/wins:.0%}")
