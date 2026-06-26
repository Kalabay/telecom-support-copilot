"""Развернуть слепые вердикты критик-прогона (4 колонки T-lite) и посчитать доли bad."""
import json
from collections import Counter, defaultdict
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "eval" / "results"
verds = json.loads((R / "critic_judge_blind.json").read_text(encoding="utf-8"))["verdicts"]
mp = {(m["dialogue_id"], m["turn_idx"]): m
      for m in json.loads((R / "critic_map.json").read_text(encoding="utf-8"))}
ds = {(r["dialogue_id"], r["turn_idx"]): r["dataset"]
      for r in json.loads((R / "critic_all.json").read_text(encoding="utf-8"))}

COLS = ["tlite_raw", "tlite_critic", "tlite_filter", "tlite_filter_critic"]
LABEL = {"tlite_raw": "T-lite сырой", "tlite_critic": "T-lite + критик",
         "tlite_filter": "T-lite + фильтр", "tlite_filter_critic": "T-lite фильтр+критик"}
agg = {m: Counter() for m in COLS}
agg_ds = {m: defaultdict(Counter) for m in COLS}

for v in verds:
    k = (v["dialogue_id"], v["turn_idx"])
    m4 = mp[k]
    for vk in ("v1", "v2", "v3", "v4"):
        agg[m4[vk]][v[vk]] += 1
        agg_ds[m4[vk]][ds[k]][v[vk]] += 1


def line(name, c):
    n = sum(c.values()) or 1
    g, w, b = c["good"], c["weak"], c["bad"]
    return (f"{name:24s} n={n:3d}  good {g/n:5.1%}  weak {w/n:5.1%}  bad {b/n:5.1%}  "
            f"(пригодны {(g+w)/n:.1%})")


print("=== T-lite + критик на 87 злых (слепой судья) ===\n")
for m in COLS:
    print(line(LABEL[m], agg[m]))
print("\n--- по датасетам (доля bad) ---")
for m in COLS:
    parts = [f"{tag} {agg_ds[m][tag]['bad']/(sum(agg_ds[m][tag].values()) or 1):.1%}"
             for tag in ("real", "vektor")]
    print(f"{LABEL[m]:24s} " + " | ".join(parts))

md = ["# T-lite + критик — доля плохих на злых (87 реплик, слепой судья)\n",
      "| Вариант | good | weak | bad | пригодны |", "|---|---|---|---|---|"]
for m in COLS:
    c = agg[m]; n = sum(c.values()) or 1
    md.append(f"| {LABEL[m]} | {c['good']/n:.1%} | {c['weak']/n:.1%} | "
              f"**{c['bad']/n:.1%}** | {(c['good']+c['weak'])/n:.1%} |")
(R / "critic_result.md").write_text("\n".join(md), encoding="utf-8")
print("\nsaved -> eval/results/critic_result.md")
