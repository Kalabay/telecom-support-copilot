import json
from collections import defaultdict
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "eval" / "results"

gold = {}
for f in ["e2e_real_result.json", "e2e_dialogues_result.json"]:
    for t in json.loads((R / f).read_text(encoding="utf-8"))["turns"]:
        if t.get("gold_doc_ids"):
            gold[(t["dialogue_id"], t["turn_idx"])] = t["gold_doc_ids"]

client = chromadb.PersistentClient(path=str(ROOT / "data" / "chroma"),
                                   settings=ChromaSettings(anonymized_telemetry=False))
got = client.get_collection("telecom_kb_frida").get(include=["documents", "metadatas"])
dtext = defaultdict(str)
for txt, m in zip(got["documents"], got["metadatas"]):
    if len(dtext[m["doc_id"]]) < 1200:
        dtext[m["doc_id"]] = (dtext[m["doc_id"]] + " " + txt).strip()
answers = {o["doc_id"]: o.get("answers", []) for o in
           json.loads((R / "doc_answers.json").read_text(encoding="utf-8"))}

out = []
for d in json.loads((R / "compare_data.json").read_text(encoding="utf-8")):
    gd = [x for x in gold.get((d["dialogue_id"], d["turn_idx"]), []) if x in dtext]
    if not gd:
        continue
    out.append({
        "dialogue_id": d["dialogue_id"], "turn_idx": d["turn_idx"],
        "asr_text": d["asr_text"], "ideal_text": d["ideal_text"],
        "doc": "\n---\n".join(dtext[x] for x in gd),
        "answers": [a for x in gd for a in answers.get(x, [])],
        "transcript": d["transcript"],
    })
(R / "emo_input.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"{len(out)} -> emo_input.json")
