"""Слепой набор: текущий Qwen3 (ретрив) vs A (gold+2) vs B (только gold)."""
import json
import random
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "eval" / "results"
cur = {(r["dialogue_id"], r["turn_idx"]): r["raw"]
       for r in json.loads((R / "bench_qwen3_14b.json").read_text(encoding="utf-8"))}
perf = json.loads((R / "perfect_context.json").read_text(encoding="utf-8"))

rnd = random.Random(7)
blind, mapping = [], []
for d in perf:
    k = (d["dialogue_id"], d["turn_idx"])
    cols = {"retrieved (как сейчас)": cur.get(k, ""),
            "A_gold+2": d["answer_A_gold_plus2"],
            "B_gold_only": d["answer_B_gold_only"]}
    pairs = list(cols.items()); rnd.shuffle(pairs)
    blind.append({"dialogue_id": k[0], "turn_idx": k[1], "asr_text": d["asr_text"],
                  "ideal_text": d["ideal_text"], **{f"v{i+1}": t for i, (_, t) in enumerate(pairs)}})
    mapping.append({"dialogue_id": k[0], "turn_idx": k[1],
                    **{f"v{i+1}": lbl for i, (lbl, _) in enumerate(pairs)}})

(R / "perfect_blind.json").write_text(json.dumps(blind, ensure_ascii=False, indent=2), encoding="utf-8")
(R / "perfect_map.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"{len(blind)} реплик x v1..v3 -> perfect_blind.json")
