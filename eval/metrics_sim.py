"""Семантическая близость ответа к эталону (FRIDA cosine) и доля пустых/отказов по моделям."""
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "eval" / "results"
MODELS = [
    ("mistral24", "Mistral-24B"), ("tlite21", "T-lite 8B"),
    ("vikhr_nemo12", "Vikhr-12B"), ("gigachat20_v15", "GigaChat-20B"),
    ("qwen3moe", "Qwen3-MoE"), ("ruadapt32", "Ruadapt-32B"), ("qwen3_14b", "Qwen3-14B"),
]
PFX = "paraphrase: "

from sentence_transformers import SentenceTransformer  # noqa: E402
m = SentenceTransformer("ai-forever/FRIDA", trust_remote_code=True,
                        device="cuda", cache_folder=str(ROOT / ".hf_cache"))

print(f"{'Модель':14s} {'FRIDA-близость':>15s} {'пусто/отказ':>12s}")
print("-" * 44)
for key, name in MODELS:
    p = R / f"bench_{key}.json"
    if not p.exists():
        continue
    bench = json.loads(p.read_text(encoding="utf-8"))
    ans = [x["filtered"].strip() for x in bench]
    idl = [x["ideal_text"].strip() for x in bench]
    empty = sum(1 for a in ans if len(a.split()) < 3) / len(ans)
    ea = m.encode([PFX + a for a in ans], normalize_embeddings=True, show_progress_bar=False)
    ei = m.encode([PFX + i for i in idl], normalize_embeddings=True, show_progress_bar=False)
    cos = float((ea * ei).sum(1).mean())
    print(f"{name:14s} {cos:>15.3f} {empty:>11.0%}")
