"""Развернуть судью MoE vs Mistral vs 14B."""
import json
from collections import Counter
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "eval" / "results"
ver = json.loads((R / "moe_judge.json").read_text(encoding="utf-8"))["verdicts"]
mp = {(m["dialogue_id"], m["turn_idx"]): m
      for m in json.loads((R / "moe_map.json").read_text(encoding="utf-8"))}
agg = {}
for v in ver:
    m = mp[(v["dialogue_id"], v["turn_idx"])]
    for vk, lbl in m.items():
        if vk.startswith("v") and vk in v:
            agg.setdefault(lbl, Counter())[v[vk]] += 1
LAT = {"Qwen3-14B": "2.0с", "Mistral-24B": "2.8с", "Qwen3-30B-MoE": "0.79с"}
print("=== Qwen3-14B vs Mistral-24B vs Qwen3-30B-MoE (87 злых, слепой судья) ===\n")
for lbl in sorted(agg, key=lambda x: -agg[x]["good"]):
    c = agg[lbl]; n = sum(c.values()) or 1
    print(f"{lbl:16s} good {c['good']/n:5.1%}  weak {c['weak']/n:5.1%}  bad {c['bad']/n:5.1%}  "
          f"(пригодны {(c['good']+c['weak'])/n:5.1%})  скорость {LAT.get(lbl,'?')}")
