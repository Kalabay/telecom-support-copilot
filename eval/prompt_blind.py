import json
import random
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "eval" / "results"
hard = json.loads((R / "hard_subset.json").read_text(encoding="utf-8"))
claude = {(r["dialogue_id"], r["turn_idx"]): r.get("claude_raw", "")
          for r in json.loads((R / "compare_all.json").read_text(encoding="utf-8"))}

VARIANTS = ["current", "original", "specific", "fewshot", "decisive"]
data = {}
for v in VARIANTS:
    rows = json.loads((R / f"prompt_{v}.json").read_text(encoding="utf-8"))
    data[v] = {(r["dialogue_id"], r["turn_idx"]): r["suggestion"] for r in rows}
data["claude"] = claude
LABELS = VARIANTS + ["claude"]

rnd = random.Random(44)
blind, mapping = [], []
for d in hard:
    k = (d["dialogue_id"], d["turn_idx"])
    pairs = [(lbl, data[lbl].get(k, "")) for lbl in LABELS]
    rnd.shuffle(pairs)
    blind.append({"dialogue_id": k[0], "turn_idx": k[1], "asr_text": d["asr_text"],
                  "ideal_text": d["ideal_text"],
                  **{f"v{i+1}": t for i, (_, t) in enumerate(pairs)}})
    mapping.append({"dialogue_id": k[0], "turn_idx": k[1],
                    **{f"v{i+1}": lbl for i, (lbl, _) in enumerate(pairs)}})

(R / "prompt_blind.json").write_text(json.dumps(blind, ensure_ascii=False, indent=2), encoding="utf-8")
(R / "prompt_map.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"{len(blind)} реплик x v1..v{len(LABELS)} -> prompt_blind.json")
