"""Способ B: NLI-фильтр верности. Фактические утверждения ответа проверяем на следование из."""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import sentence_transformers  # noqa: F401,E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402
import chromadb  # noqa: E402
from chromadb.config import Settings as ChromaSettings  # noqa: E402

R = PROJECT_ROOT / "eval" / "results"
KEYS = re.compile(r"\d|верн|срок|дн[яей]|час|минут|мастер|бесплатн|плат|авари|сбо|опци|тариф|"
                  r"сгор|перерасч|компенс|подключ|оформ|начисл|списан|руб|гб|мбит|щит|апн|apn", re.I)
THR = 0.45


def main() -> None:
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

    name = "cointegrated/rubert-base-cased-nli-threeway"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).eval()
    ent_idx = next(i for i, l in model.config.id2label.items() if "entail" in l.lower())
    print(f"NLI готов, entailment-индекс {ent_idx}", flush=True)

    out = []; swapped = 0
    for i, r in enumerate(pb):
        k = (r["dialogue_id"], r["turn_idx"])
        kb = " ".join(dtext[g] for g in gold.get(k, []))[:1500]
        sents = [s.strip() for s in re.split(r"[.!?]+", r["raw"]) if len(s.strip()) > 10]
        claims = [s for s in sents if KEYS.search(s)]
        flagged = False
        for s in claims:
            inp = tok(kb, s, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                p = F.softmax(model(**inp).logits, dim=-1)[0]
            if p[ent_idx].item() < THR:
                flagged = True; break
        if flagged and k in moe:
            use = moe[k]; swapped += 1
        else:
            use = r["raw"]
        out.append({**{x: r[x] for x in ("dialogue_id", "turn_idx", "asr_text", "ideal_text")}, "raw": use})
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(pb)}", flush=True)
    (R / "playbook_fixB.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"B (NLI+MoE): подменено {swapped}/{len(out)} -> playbook_fixB.json")


if __name__ == "__main__":
    main()
