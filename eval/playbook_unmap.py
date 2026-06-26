"""Развернуть судью плейбука."""
import json
from collections import Counter
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "eval" / "results"
ver = json.loads((R / "playbook_judge.json").read_text(encoding="utf-8"))["verdicts"]
mp = {(m["dialogue_id"], m["turn_idx"]): m
      for m in json.loads((R / "playbook_map.json").read_text(encoding="utf-8"))}
agg = {}
for v in ver:
    m = mp[(v["dialogue_id"], v["turn_idx"])]
    for vk, lbl in m.items():
        if vk.startswith("v") and vk in v:
            agg.setdefault(lbl, Counter())[v[vk]] += 1
print("=== Плейбук (Mistral+gold+ответы) vs Mistral-plain (87 злых, слепой судья) ===\n")
for lbl in ["Плейбук (gold+ответы)", "Mistral-plain (ретрив)"]:
    c = agg.get(lbl, Counter()); n = sum(c.values()) or 1
    print(f"{lbl:26s} good {c['good']/n:5.1%}  weak {c['weak']/n:5.1%}  bad {c['bad']/n:5.1%}  "
          f"(пригодны {(c['good']+c['weak'])/n:5.1%})")
