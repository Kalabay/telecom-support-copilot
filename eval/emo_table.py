"""Сводная таблица эмоция-абляции по всем моделям: с эмоцией vs без (попарный судья)."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parents[1] / "eval" / "results"
MODELS = [
    ("mistral24", "Mistral-24B"),
    ("tlite21", "T-lite 8B"),
    ("vikhr_nemo12", "Vikhr-12B"),
    ("gigachat20_v15", "GigaChat-20B"),
    ("claude", "Claude (идеал)"),
]

print("ЭМОЦИЯ-АБЛЯЦИЯ по моделям (плейбук, best, попарный судья на злых клиентах)\n")
print(f"{'Модель':18s} {'с эмоцией':>14s} {'без эмоции':>14s} {'ничья':>10s}")
print("-" * 60)
for key, name in MODELS:
    jf = R / f"emo_judge_{key}.json"
    mf = R / f"emo_map_{key}.json"
    if not (jf.exists() and mf.exists()):
        print(f"{name:18s} {'(нет данных)':>14s}")
        continue
    ver = json.loads(jf.read_text(encoding="utf-8"))["verdicts"]
    mp = {(m["dialogue_id"], m["turn_idx"]): m
          for m in json.loads(mf.read_text(encoding="utf-8"))}
    c = Counter()
    for v in ver:
        m = mp.get((v["dialogue_id"], v["turn_idx"]))
        if not m:
            continue
        w = v.get("winner")
        if w == "tie":
            c["tie"] += 1
        elif w in ("A", "B"):
            c[m[w]] += 1
    n = sum(c.values()) or 1
    a = f"{c['with']} ({c['with']/n:.0%})"
    b = f"{c['without']} ({c['without']/n:.0%})"
    t = f"{c['tie']} ({c['tie']/n:.0%})"
    print(f"{name:18s} {a:>14s} {b:>14s} {t:>10s}")
