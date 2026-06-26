"""Полная связка: best-of-3 + safety-фильтр + MoE-фолбэк."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import sentence_transformers  # noqa: F401,E402
import chromadb  # noqa: E402
from chromadb.config import Settings as ChromaSettings  # noqa: E402
from app.pipeline.llm import _risk  # noqa: E402

R = PROJECT_ROOT / "eval" / "results"
RANK = {"good": 2, "weak": 1, "bad": 0}
NAME = {2: "good", 1: "weak", 0: "bad"}

blind = {(b["dialogue_id"], b["turn_idx"]): b
         for b in json.loads((R / "bo3_blind.json").read_text(encoding="utf-8"))}
ver = {(v["dialogue_id"], v["turn_idx"]): v
       for v in json.loads((R / "bo3_judge.json").read_text(encoding="utf-8"))["verdicts"]}
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

bo3, combined = Counter(), Counter()
all3bad = 0
for k, b in blind.items():
    vv = ver.get(k, {})
    kb = " ".join(dtext[g] for g in gold.get(k, []))
    verdicts = {col: vv.get(col, "bad") for col in ("v1", "v2", "v3")}

    bo3[NAME[max(RANK[verdicts[c]] for c in verdicts)]] += 1

    safe = [c for c in ("v1", "v2", "v3") if _risk(b[c], kb) < 2]
    if safe:
        combined[NAME[max(RANK[verdicts[c]] for c in safe)]] += 1
    else:
        combined["weak"] += 1
        all3bad += 1

n = len(blind) or 1
print("=== Полная связка: best-of-3 + safety + MoE-фолбэк (87 злых) ===\n")
for name, c in [("Best-of-3 (без фильтра)", bo3), ("Best-of-3 + safety + MoE", combined)]:
    print(f"{name:28s} good {c['good']/n:5.1%}  weak {c['weak']/n:5.1%}  bad {c['bad']/n:5.1%}  "
          f"(pretty {(c['good']+c['weak'])/n:5.1%})")
print(f"\nвсе 3 варианта рискованны -> MoE-фолбэк: {all3bad} реплик")
