"""Трудный сабсет: реплики, где больше всего моделей выдумывают (bad)."""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "eval" / "results"
ver = {(v["dialogue_id"], v["turn_idx"]): v
       for v in json.loads((R / "final_judge_blind.json").read_text(encoding="utf-8"))["verdicts"]}
mp = {(m["dialogue_id"], m["turn_idx"]): m
      for m in json.loads((R / "final_map.json").read_text(encoding="utf-8"))}
base = json.loads((R / "compare_data.json").read_text(encoding="utf-8"))

bad_count = {}
for k, v in ver.items():
    bad_count[k] = sum(1 for vk in v if vk.startswith("v") and v[vk] == "bad")

from collections import Counter
print("распределение #моделей-с-bad на реплику:", dict(sorted(Counter(bad_count.values()).items())))

for thr in (3, 2, 1):
    hard = [d for d in base if bad_count.get((d["dialogue_id"], d["turn_idx"]), 0) >= thr]
    print(f"  порог >= {thr} bad: {len(hard)} реплик")
    if 15 <= len(hard) <= 40 or thr == 1:
        chosen, THR = hard, thr
        break

(R / "hard_subset.json").write_text(json.dumps(chosen, ensure_ascii=False, indent=2), encoding="utf-8")
real = sum(1 for d in chosen if d["dataset"] == "real")
print(f"\nвыбран порог >= {THR}: {len(chosen)} трудных реплик (real {real}, vektor {len(chosen)-real})")
print("saved -> eval/results/hard_subset.json")
