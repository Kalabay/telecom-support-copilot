"""Объединить per-company e2e-кейсы в единый e2e_cases.json."""
import json
from collections import Counter
from pathlib import Path

E = Path(r"K:\dev\coursework\eval")
companies = ["mts", "beeline", "megafon", "tele2", "rostelecom"]
all_cases = []
for c in companies:
    f = E / f"e2e_cases_{c}.json"
    cases = json.loads(f.read_text(encoding="utf-8"))["cases"]
    all_cases.extend(cases)

emo = Counter(x["emotion"] for x in all_cases)
comp = Counter(x["company"] for x in all_cases)
ids = [x["emotion"] for x in all_cases]
assert all(x["emotion"] in ("angry", "sad", "neutral") for x in all_cases), "плохая эмоция"
assert len({x["id"] for x in all_cases}) == len(all_cases), "дубли id"

(E / "e2e_cases.json").write_text(
    json.dumps({"_comment": "150 e2e-кейсов, 5 компаний x 30, баланс эмоций",
                "cases": all_cases}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"всего: {len(all_cases)} кейсов")
print(f"по компаниям: {dict(comp)}")
print(f"по эмоциям: {dict(emo)}")
