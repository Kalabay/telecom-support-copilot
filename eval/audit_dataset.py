"""Аудит бенчмарка: найти реплики, где модели стабильно дают bad, и вытащить их контекст."""
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import sentence_transformers  # noqa: F401,E402
import chromadb  # noqa: E402
from chromadb.config import Settings as ChromaSettings  # noqa: E402

R = PROJECT_ROOT / "eval" / "results"
PAIRS = [("final_judge_blind.json", "final_map.json"),
         ("current_best_judge.json", "current_best_map.json"),
         ("ruadapt_judge.json", "ruadapt_map.json"),
         ("perfect_judge.json", "perfect_map.json")]

bad = Counter(); total = Counter()
for jf, mf in PAIRS:
    if not (R / jf).exists() or not (R / mf).exists():
        continue
    ver = json.loads((R / jf).read_text(encoding="utf-8"))["verdicts"]
    mp = {(m["dialogue_id"], m["turn_idx"]): m for m in json.loads((R / mf).read_text(encoding="utf-8"))}
    for v in ver:
        k = (v["dialogue_id"], v["turn_idx"])
        for vk in v:
            if vk.startswith("v") and vk in mp.get(k, {}):
                total[k] += 1
                if v[vk] == "bad":
                    bad[k] += 1

ctx = {}
for f in ["e2e_real_result.json", "e2e_dialogues_result.json"]:
    for t in json.loads((R / f).read_text(encoding="utf-8"))["turns"]:
        ctx[(t["dialogue_id"], t["turn_idx"])] = {
            "asr": t.get("asr_text", ""), "clean": t.get("clean_text", ""),
            "gold": t.get("gold_doc_ids", []), "wer": t.get("wer"),
            "ideal": next((tt.get("ideal_text", "") for tt in []), "")}
for d in json.loads((R / "compare_data.json").read_text(encoding="utf-8")):
    k = (d["dialogue_id"], d["turn_idx"])
    if k in ctx:
        ctx[k]["ideal"] = d["ideal_text"]

client = chromadb.PersistentClient(path=str(PROJECT_ROOT / "data" / "chroma"),
                                   settings=ChromaSettings(anonymized_telemetry=False))
got = client.get_collection("telecom_kb_frida").get(include=["documents", "metadatas"])
dtext = defaultdict(str)
for txt, m in zip(got["documents"], got["metadatas"]):
    if len(dtext[m["doc_id"]]) < 400:
        dtext[m["doc_id"]] = (dtext[m["doc_id"]] + " " + txt).strip()

ranked = sorted(bad.items(), key=lambda x: -x[1])
print(f"всего реплик с хотя бы 1 bad: {len(bad)}; распределение bad-рейта:")
print("  ", dict(Counter(round(bad[k]/total[k], 1) for k in bad)))
out = []
for k, b in ranked[:14]:
    c = ctx.get(k, {})
    out.append({"turn": f"{k[0]} t{k[1]}", "bad": f"{b}/{total[k]}", "wer": c.get("wer"),
                "asr": c.get("asr", ""), "clean": c.get("clean", ""), "ideal": c.get("ideal", ""),
                "gold": c.get("gold", []),
                "gold_text": {g: dtext.get(g, "(нет)")[:250] for g in c.get("gold", [])}})
(R / "audit_bad.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nтоп-{len(out)} проблемных -> audit_bad.json")
