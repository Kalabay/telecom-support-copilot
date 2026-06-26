"""Собрать слепой набор для финального сравнения всех моделей (сырой вывод)."""
import json
import random
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "eval" / "results"
CONTENDERS = [
    ("tlite21", "compare_data.json", "tlite_raw"),
    ("claude", "compare_all.json", "claude_raw"),
    ("qwen3_14b", "bench_qwen3_14b.json", "raw"),
    ("vikhr_nemo12", "bench_vikhr_nemo12.json", "raw"),
    ("gigachat20_v15", "bench_gigachat20_v15.json", "raw"),
]

present = []
data = {}
for label, fname, field in CONTENDERS:
    p = R / fname
    if not p.exists():
        print(f"  пропуск {label}: нет {fname}")
        continue
    rows = json.loads(p.read_text(encoding="utf-8"))
    data[label] = {(r["dialogue_id"], r["turn_idx"]): r.get(field, "") for r in rows}
    present.append((label, field))
print("участники:", [l for l, _ in present])

base = json.loads((R / "compare_data.json").read_text(encoding="utf-8"))
rnd = random.Random(42)
blind, mapping = [], []
for d in base:
    k = (d["dialogue_id"], d["turn_idx"])
    pairs = [(label, data[label].get(k, "")) for label, _ in present]
    rnd.shuffle(pairs)
    blind.append({"dialogue_id": k[0], "turn_idx": k[1],
                  "asr_text": d["asr_text"], "ideal_text": d["ideal_text"],
                  **{f"v{i+1}": t for i, (_, t) in enumerate(pairs)}})
    mapping.append({"dialogue_id": k[0], "turn_idx": k[1], "dataset": d["dataset"],
                    **{f"v{i+1}": label for i, (label, _) in enumerate(pairs)}})

(R / "final_blind.json").write_text(json.dumps(blind, ensure_ascii=False, indent=2), encoding="utf-8")
(R / "final_map.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
n_v = len(present)
print(f"слепой набор: {len(blind)} реплик x v1..v{n_v} -> final_blind.json (+ final_map.json)")
print(f"-> судье: оценить v1..v{n_v}")
