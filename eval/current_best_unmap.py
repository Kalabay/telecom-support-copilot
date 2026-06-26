"""Развернуть вердикты baseline vs BEST."""
import json
from collections import Counter
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "eval" / "results"
ver = json.loads((R / "current_best_judge.json").read_text(encoding="utf-8"))["verdicts"]
mp = {(m["dialogue_id"], m["turn_idx"]): m
      for m in json.loads((R / "current_best_map.json").read_text(encoding="utf-8"))}
agg = {}
for v in ver:
    m = mp[(v["dialogue_id"], v["turn_idx"])]
    for vk, lbl in m.items():
        if vk.startswith("v") and vk in v:
            agg.setdefault(lbl, Counter())[v[vk]] += 1
print("=== Наша лучшая версия vs baseline (Qwen3, 87 злых, слепой судья) ===\n")
for lbl in ["baseline (обычный RAG)", "BEST (doc2query RAG)"]:
    c = agg.get(lbl, Counter()); n = sum(c.values()) or 1
    print(f"{lbl:24s} good {c['good']/n:5.1%}  weak {c['weak']/n:5.1%}  bad {c['bad']/n:5.1%}  "
          f"(пригодны {(c['good']+c['weak'])/n:.1%})")
