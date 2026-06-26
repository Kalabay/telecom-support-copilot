"""ВРЕМЕННЫЙ эксперимент: даём Qwen3 идеальный контекст (gold-документ) на злых репликах."""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import app  # noqa: F401,E402
import sentence_transformers  # noqa: F401,E402
import torch  # noqa: F401,E402
import chromadb  # noqa: E402
from chromadb.config import Settings as ChromaSettings  # noqa: E402

from app.models.schemas import Emotion, EmotionState, KBSource  # noqa: E402

R = PROJECT_ROOT / "eval" / "results"


def main() -> None:
    from app.pipeline.llm import get_generator
    gold = {}
    for f in ["e2e_real_result.json", "e2e_dialogues_result.json"]:
        for t in json.loads((R / f).read_text(encoding="utf-8"))["turns"]:
            if t.get("gold_doc_ids"):
                gold[(t["dialogue_id"], t["turn_idx"])] = t["gold_doc_ids"]
    client = chromadb.PersistentClient(path=str(PROJECT_ROOT / "data" / "chroma"),
                                       settings=ChromaSettings(anonymized_telemetry=False))
    got = client.get_collection("telecom_kb_frida").get(include=["documents", "metadatas"])
    dtext = defaultdict(str); dtitle = {}; dcomp = {}; by_comp = defaultdict(set)
    for txt, m in zip(got["documents"], got["metadatas"]):
        did = m["doc_id"]
        if len(dtext[did]) < 1400:
            dtext[did] = (dtext[did] + " " + txt).strip()
        dtitle[did] = m.get("title", did); dcomp[did] = m["company"]; by_comp[m["company"]].add(did)

    gen = get_generator(); gen._ensure_loaded()
    print("Qwen3 загружен", flush=True)
    emo = EmotionState(label=Emotion.ANGRY, confidence=0.85, arousal=0.85, valence=-0.7, escalation_risk=True)
    rnd = random.Random(42)

    angry = json.loads((R / "compare_data.json").read_text(encoding="utf-8"))
    out = []
    for i, d in enumerate(angry):
        key = (d["dialogue_id"], d["turn_idx"])
        gd = [g for g in gold.get(key, []) if g in dtext]
        if not gd:
            continue
        comp = d["company"]
        gold_src = [KBSource(doc_id=g, title=dtitle[g], snippet=dtext[g], score=1.0) for g in gd]
        pool = [x for x in by_comp[comp] if x not in gd]
        dist = rnd.sample(pool, min(2, len(pool)))
        dist_src = [KBSource(doc_id=x, title=dtitle[x], snippet=dtext[x], score=0.5) for x in dist]
        srcA = gold_src + dist_src; rnd.shuffle(srcA)
        sugA = gen.generate(d["transcript"], emo, srcA, safe=False).suggestions
        sugB = gen.generate(d["transcript"], emo, list(gold_src), safe=False).suggestions
        out.append({"dialogue_id": d["dialogue_id"], "turn_idx": d["turn_idx"], "dataset": d["dataset"],
                    "asr_text": d["asr_text"], "ideal_text": d["ideal_text"], "gold": gd,
                    "answer_A_gold_plus2": sugA[0] if sugA else "",
                    "answer_B_gold_only": sugB[0] if sugB else ""})
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(angry)}", flush=True)

    (R / "perfect_context.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(out)} реплик с gold -> perfect_context.json")


if __name__ == "__main__":
    main()
