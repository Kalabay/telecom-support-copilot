"""Слепой набор по ОТФИЛЬТРОВАННЫМ выводам всех моделей (фильтр поверх каждой модели)."""
import json
import random
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "eval" / "results"
CONTENDERS = [
    ("tlite21", "compare_data.json", "tlite_filtered"),
    ("claude", "compare_all.json", "claude_filtered"),
    ("qwen3_14b", "bench_qwen3_14b.json", "filtered"),
    ("vikhr_nemo12", "bench_vikhr_nemo12.json", "filtered"),
    ("gigachat20_v15", "bench_gigachat20_v15.json", "filtered"),
]

present, data = [], {}
for label, fname, field in CONTENDERS:
    p = R / fname
    if not p.exists():
        continue
    rows = json.loads(p.read_text(encoding="utf-8"))
    data[label] = {(r["dialogue_id"], r["turn_idx"]): r.get(field, "") for r in rows}
    present.append(label)
print("участники (filtered):", present)

base = json.loads((R / "compare_data.json").read_text(encoding="utf-8"))
rnd = random.Random(43)
blind, mapping = [], []
for d in base:
    k = (d["dialogue_id"], d["turn_idx"])
    pairs = [(lbl, data[lbl].get(k, "")) for lbl in present]
    rnd.shuffle(pairs)
    blind.append({"dialogue_id": k[0], "turn_idx": k[1], "asr_text": d["asr_text"],
                  "ideal_text": d["ideal_text"],
                  **{f"v{i+1}": t for i, (_, t) in enumerate(pairs)}})
    mapping.append({"dialogue_id": k[0], "turn_idx": k[1], "dataset": d["dataset"],
                    **{f"v{i+1}": lbl for i, (lbl, _) in enumerate(pairs)}})

(R / "final_filtered_blind.json").write_text(json.dumps(blind, ensure_ascii=False, indent=2), encoding="utf-8")
(R / "final_filtered_map.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"{len(blind)} реплик x v1..v{len(present)} -> final_filtered_blind.json")
