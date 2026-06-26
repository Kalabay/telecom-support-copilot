"""Способ A: регулярный фильтр риска на ответах плейбука -> подмена на безопасный MoE."""
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import sentence_transformers  # noqa: F401,E402
import chromadb  # noqa: E402
from chromadb.config import Settings as ChromaSettings  # noqa: E402
from app.pipeline.llm import _risk  # noqa: E402

R = PROJECT_ROOT / "eval" / "results"
pb = json.loads((R / "bench_playbook.json").read_text(encoding="utf-8"))
moe = {(r["dialogue_id"], r["turn_idx"]): r["raw"]
       for r in json.loads((R / "bench_qwen3moe.json").read_text(encoding="utf-8"))}
gold = {}
for f in ["e2e_real_result.json", "e2e_dialogues_result.json"]:
    for t in json.loads((R / f).read_text(encoding="utf-8"))["turns"]:
        if t.get("gold_doc_ids"):
            gold[(t["dialogue_id"], t["turn_idx"])] = t["gold_doc_ids"]
client = chromadb.PersistentClient(path=str(PROJECT_ROOT / "data" / "chroma"),
                                   settings=ChromaSettings(anonymized_telemetry=False))
got = client.get_collection("telecom_kb_frida").get(include=["documents", "metadatas"])
dtext = defaultdict(str)
for txt, m in zip(got["documents"], got["metadatas"]):
    if len(dtext[m["doc_id"]]) < 1400:
        dtext[m["doc_id"]] = (dtext[m["doc_id"]] + " " + txt).strip()

out = []; swapped = 0
for r in pb:
    k = (r["dialogue_id"], r["turn_idx"])
    kb = " ".join(dtext[g] for g in gold.get(k, []))
    if _risk(r["raw"], kb) >= 2 and k in moe:
        ans = moe[k]; swapped += 1
    else:
        ans = r["raw"]
    out.append({**{x: r[x] for x in ("dialogue_id", "turn_idx", "asr_text", "ideal_text")}, "raw": ans})
(R / "playbook_fixA.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"A (регулярка+MoE): подменено {swapped}/{len(out)} -> playbook_fixA.json")
