"""Развернуть слепые вердикты финального сравнения в модели и посчитать доли bad."""
import json
from collections import Counter, defaultdict
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "eval" / "results"
verds = json.loads((R / "final_judge_blind.json").read_text(encoding="utf-8"))["verdicts"]
mp = {(m["dialogue_id"], m["turn_idx"]): m
      for m in json.loads((R / "final_map.json").read_text(encoding="utf-8"))}

LABEL = {"tlite21": "T-lite-it-2.1 (8B)", "claude": "Claude (фронтир)",
         "qwen3_14b": "Qwen3-14B", "vikhr_nemo12": "Vikhr-Nemo-12B",
         "gigachat20_v15": "GigaChat-20B-A3B v1.5"}
agg = defaultdict(Counter)
agg_ds = defaultdict(lambda: defaultdict(Counter))

for v in verds:
    k = (v["dialogue_id"], v["turn_idx"])
    m = mp[k]
    for vk, model in m.items():
        if vk.startswith("v") and vk in v:
            agg[model][v[vk]] += 1
            agg_ds[model][m.get("dataset", "?")][v[vk]] += 1

order = sorted(agg, key=lambda x: agg[x]["bad"] / (sum(agg[x].values()) or 1))
print("=== Финальное сравнение моделей на 87 злых (сырой вывод, слепой судья) ===\n")
md = ["# Финальное сравнение моделей — доля плохих на злых (сырой вывод, слепой судья)\n",
      "| Модель | good | weak | bad | пригодны |", "|---|---|---|---|---|"]
for m in order:
    c = agg[m]; n = sum(c.values()) or 1
    g, w, b = c["good"], c["weak"], c["bad"]
    print(f"{LABEL.get(m, m):26s} n={n:3d}  good {g/n:5.1%}  weak {w/n:5.1%}  bad {b/n:5.1%}  "
          f"(пригодны {(g+w)/n:.1%})")
    md.append(f"| {LABEL.get(m, m)} | {g/n:.1%} | {w/n:.1%} | **{b/n:.1%}** | {(g+w)/n:.1%} |")

print("\n--- по датасетам (доля bad) ---")
for m in order:
    parts = [f"{tag} {agg_ds[m][tag]['bad']/(sum(agg_ds[m][tag].values()) or 1):.1%}"
             for tag in ("real", "vektor")]
    print(f"{LABEL.get(m, m):26s} " + " | ".join(parts))

(R / "final_result.md").write_text("\n".join(md), encoding="utf-8")
print("\nsaved -> eval/results/final_result.md")
