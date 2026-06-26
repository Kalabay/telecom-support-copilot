"""Слепой попарный набор для эмоция-абляции одной модели. Аргумент — имя модели."""
import json
import random
import sys
from pathlib import Path

model = sys.argv[1]
R = Path(__file__).resolve().parents[1] / "eval" / "results"
d = json.loads((R / f"emo_ablation_{model}.json").read_text(encoding="utf-8"))
rnd = random.Random(99)
blind, mp = [], []
for r in d:
    pair = [("with", r["with_emotion"]), ("without", r["without_emotion"])]
    rnd.shuffle(pair)
    key = {"dialogue_id": r["dialogue_id"], "turn_idx": r["turn_idx"]}
    blind.append({**key, "asr_text": r["asr_text"], "ideal_text": r["ideal_text"],
                  "A": pair[0][1], "B": pair[1][1]})
    mp.append({**key, "A": pair[0][0], "B": pair[1][0]})
(R / f"emo_blind_{model}.json").write_text(json.dumps(blind, ensure_ascii=False, indent=2), encoding="utf-8")
(R / f"emo_map_{model}.json").write_text(json.dumps(mp, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"{model}: {len(blind)} -> emo_blind_{model}.json")
