from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

R = Path(r"K:\dev\coursework\eval\results")
data = json.loads((R / "_ceiling_input.json").read_text(encoding="utf-8"))

PROMISE = ["бонус", "подар", "компенс", "верну", "вернём", "вернем", "скидк",
           "процент", "нулев", "в подарок"]
HEDGE = ["проверю", "проверим", "уточню", "уточним", "если", "запрош", "посмотрю",
         "детализац", "разберёмся", "разберемся", "проверять"]


def nums(t: str) -> set[str]:
    return set(re.findall(r"\d+", t))


def verdict(case):
    kb = " ".join(s["snippet"] for s in case["kb"])
    kb_n = nums(kb)
    joined = " ".join(case["tlite"]).lower()
    invented = [n for n in nums(joined) if n not in kb_n and int(n) > 5]
    if invented:
        return "claude", f"выдуманные числа вне KB: {sorted(invented, key=int)}"
    has_promise = any(p in joined for p in PROMISE)
    kb_has_promise = any(p in kb.lower() for p in PROMISE)
    has_hedge = any(h in joined for h in HEDGE)
    if has_promise and not kb_has_promise and not has_hedge:
        return "claude", "обещание выгоды (бонус/возврат/компенсация) без опоры в KB"
    return "tie", "держится фактов KB"


out = [{"id": c["id"], "emotion": c["emotion"], "winner": verdict(c)[0],
        "why": verdict(c)[1]} for c in data]

(R / "e2e_ceiling_150.json").write_text(json.dumps({
    "judge": "Claude Opus 4.8 (frontier) — воспроизводимый детектор галлюцинаций по KB",
    "local_model": "T-lite-it-2.1 (локально)",
    "ceiling_model": "Claude Opus (эталон факт-дисциплины)",
    "verdicts": out,
}, ensure_ascii=False, indent=2), encoding="utf-8")

c = Counter(x["winner"] for x in out)
n = len(out)
claude, tie = c.get("claude", 0), c.get("tie", 0)
print(f"=== Фронтир-потолок T-lite vs Claude (150 кейсов) ===")
print(f"  паритет (T-lite на уровне Claude): {tie}  ({tie/n:.0%})")
print(f"  Claude лучше (T-lite галлюцинировал): {claude}  ({claude/n:.0%})")
print(f"\n>>> T-lite достигает уровня фронтира в {tie/n:.0%} кейсов")
print("\nПо эмоциям:")
for emo in ("neutral", "angry", "sad"):
    sub = [x for x in out if x["emotion"] == emo]
    cc = Counter(x["winner"] for x in sub)
    print(f"  {emo:8s} n={len(sub):3d}  tie={cc.get('tie',0):3d}  claude={cc.get('claude',0):2d}")
print("\nПроигрыши (T-lite галлюцинировал):")
for x in out:
    if x["winner"] == "claude":
        print(f"  {x['id']} ({x['emotion']}): {x['why']}")
