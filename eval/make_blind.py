"""Слепой набор для судьи: перемешать 4 колонки в v1..v4, сохранить маппинг."""
import json
import random
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "eval" / "results"
rows = json.loads((R / "compare_all.json").read_text(encoding="utf-8"))
MODELS = ["tlite_raw", "tlite_filtered", "claude_raw", "claude_filtered"]
rnd = random.Random(42)

blind, mapping = [], []
for r in rows:
    pairs = [(m, r[m]) for m in MODELS]
    rnd.shuffle(pairs)
    vk = {f"v{i+1}": txt for i, (_, txt) in enumerate(pairs)}
    mk = {f"v{i+1}": m for i, (m, _) in enumerate(pairs)}
    blind.append({"dialogue_id": r["dialogue_id"], "turn_idx": r["turn_idx"],
                  "asr_text": r["asr_text"], "ideal_text": r["ideal_text"], **vk})
    mapping.append({"dialogue_id": r["dialogue_id"], "turn_idx": r["turn_idx"], **mk})

(R / "compare_blind.json").write_text(json.dumps(blind, ensure_ascii=False, indent=2), encoding="utf-8")
(R / "compare_map.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"слепой набор: {len(blind)} реплик x v1..v4 -> compare_blind.json (+ compare_map.json)")
