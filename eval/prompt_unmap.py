"""Развернуть слепые вердикты вариантов промпта -> good/weak/bad по каждому варианту."""
import json
from collections import Counter
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "eval" / "results"
ver = json.loads((R / "prompt_judge.json").read_text(encoding="utf-8"))["verdicts"]
mp = {(m["dialogue_id"], m["turn_idx"]): m
      for m in json.loads((R / "prompt_map.json").read_text(encoding="utf-8"))}

LBL = {"current": "current (закрученный)", "original": "original (ранний)",
       "specific": "specific (конкретика)", "fewshot": "fewshot (с примером)",
       "decisive": "decisive (решительный)", "claude": "Claude (референс)"}
agg = {}
for v in ver:
    m = mp[(v["dialogue_id"], v["turn_idx"])]
    for vk, lbl in m.items():
        if vk.startswith("v") and vk in v:
            agg.setdefault(lbl, Counter())[v[vk]] += 1

order = sorted(agg, key=lambda x: -agg[x]["good"])
print("=== Варианты промпта Qwen3 на 42 трудных репликах (слепой судья) ===\n")
md = ["# Варианты промпта Qwen3 — 42 трудные реплики (слепой судья, + Claude-референс)\n",
      "| Вариант | good | weak | bad | пригодны |", "|---|---|---|---|---|"]
for lbl in order:
    c = agg[lbl]; n = sum(c.values()) or 1
    g, w, b = c["good"], c["weak"], c["bad"]
    print(f"{LBL.get(lbl, lbl):26s} good {g/n:5.1%}  weak {w/n:5.1%}  bad {b/n:5.1%}  "
          f"(пригодны {(g+w)/n:.1%})")
    md.append(f"| {LBL.get(lbl, lbl)} | {g/n:.1%} | {w/n:.1%} | **{b/n:.1%}** | {(g+w)/n:.1%} |")
(R / "prompt_result.md").write_text("\n".join(md), encoding="utf-8")
print("\nsaved -> eval/results/prompt_result.md")
