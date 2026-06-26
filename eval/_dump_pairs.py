"""Дамп пар подсказок (full vs no_emotion) для фронтир-судьи."""
import json
from pathlib import Path

R = Path(r"K:\dev\coursework\eval\results")
full = {r["id"]: r for r in json.loads((R / "e2e_full.json").read_text(encoding="utf-8"))["rows"]}
noemo = {r["id"]: r for r in json.loads((R / "e2e_no_emotion.json").read_text(encoding="utf-8"))["rows"]}

for cid in sorted(full):
    f, ne = full[cid], noemo[cid]
    print(f"### {cid} | {f['emotion']} | {f['company']}")
    print(f"КЛИЕНТ: {f['text']}")
    print(f"  [A/full]   {' /// '.join(f['suggestions'])}")
    print(f"  [B/noemo]  {' /// '.join(ne['suggestions'])}")
    print()
