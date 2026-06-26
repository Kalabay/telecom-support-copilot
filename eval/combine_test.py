"""Связка Mistral-24B + MoE: брать ответ Mistral, но на рискованных подменять на MoE (0% bad)."""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
from app.pipeline.llm import _risk  # noqa: E402

R = PROJECT_ROOT / "eval" / "results"
mi = {(r["dialogue_id"], r["turn_idx"]): r for r in json.loads((R / "bench_mistral24.json").read_text(encoding="utf-8"))}
moe = {(r["dialogue_id"], r["turn_idx"]): r["raw"] for r in json.loads((R / "bench_qwen3moe.json").read_text(encoding="utf-8"))}
src = {(d["dialogue_id"], d["turn_idx"]): d["sources"] for d in json.loads((R / "compare_data.json").read_text(encoding="utf-8"))}

THR = 2
out = []
swapped = 0
for k, r in mi.items():
    kb = " ".join(s["snippet"] for s in src.get(k, []))
    risk = _risk(r["raw"], kb)
    if risk >= THR and k in moe:
        ans = moe[k]; swapped += 1
    else:
        ans = r["raw"]
    out.append({"dialogue_id": k[0], "turn_idx": k[1], "dataset": r["dataset"],
                "asr_text": r["asr_text"], "ideal_text": r["ideal_text"], "raw": ans})
(R / "bench_combined.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"связка собрана: {len(out)} реплик, подменено на MoE: {swapped}")
print("-> bench_combined.json")
