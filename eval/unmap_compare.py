"""Развернуть слепые вердикты v1..v4 обратно в 4 модели и посчитать 2x2 по доле bad."""
import json
from collections import Counter, defaultdict
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "eval" / "results"
verds = json.loads((R / "compare_judge_blind.json").read_text(encoding="utf-8"))["verdicts"]
mp = {(m["dialogue_id"], m["turn_idx"]): m
      for m in json.loads((R / "compare_map.json").read_text(encoding="utf-8"))}
ds = {(r["dialogue_id"], r["turn_idx"]): r["dataset"]
      for r in json.loads((R / "compare_all.json").read_text(encoding="utf-8"))}

MODELS = ["tlite_raw", "tlite_filtered", "claude_raw", "claude_filtered"]
LABEL = {"tlite_raw": "T-lite сырой", "tlite_filtered": "T-lite + фильтр",
         "claude_raw": "Claude сырой", "claude_filtered": "Claude + фильтр"}
agg = {m: Counter() for m in MODELS}
agg_ds = {m: defaultdict(Counter) for m in MODELS}

for v in verds:
    k = (v["dialogue_id"], v["turn_idx"])
    m4 = mp[k]
    for vk in ("v1", "v2", "v3", "v4"):
        model = m4[vk]
        verdict = v[vk]
        agg[model][verdict] += 1
        agg_ds[model][ds[k]][verdict] += 1


def line(name, c):
    n = sum(c.values()) or 1
    g, w, b = c["good"], c["weak"], c["bad"]
    return (f"{name:20s} n={n:3d}  good {g/n:5.1%}  weak {w/n:5.1%}  bad {b/n:5.1%}  "
            f"(пригодны {(g+w)/n:.1%})")


print("=== 2x2: модель x фильтр на 87 злых (слепой судья) ===\n")
for m in MODELS:
    print(line(LABEL[m], agg[m]))
print("\n--- по датасетам (доля bad) ---")
for m in MODELS:
    parts = []
    for tag in ("real", "vektor"):
        c = agg_ds[m][tag]; n = sum(c.values()) or 1
        parts.append(f"{tag} {c['bad']/n:.1%}")
    print(f"{LABEL[m]:20s} " + " | ".join(parts))

md = ["# 2×2: T-lite vs Claude × сырой/фильтр — доля плохих на злых (87 реплик, слепой судья)\n",
      "| Модель | good | weak | bad | пригодны |", "|---|---|---|---|---|"]
for m in MODELS:
    c = agg[m]; n = sum(c.values()) or 1
    md.append(f"| {LABEL[m]} | {c['good']/n:.1%} | {c['weak']/n:.1%} | "
              f"**{c['bad']/n:.1%}** | {(c['good']+c['weak'])/n:.1%} |")
md.append("\nСлепое судейство: 4 варианта на реплику перемешаны в v1..v4 без указания модели.")
(R / "compare_result.md").write_text("\n".join(md), encoding="utf-8")
print("\nsaved -> eval/results/compare_result.md")
