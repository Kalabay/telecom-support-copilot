"""Метрики ретрива на всех срезах: чистые запросы vs E2E-разговорные, по компаниям."""
import json
from collections import defaultdict
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "eval" / "results"


def e2e_metrics(path):
    """R@1, R@3, MRR из сохранённых rag_top + gold_doc_ids."""
    turns = [t for t in json.loads(Path(path).read_text(encoding="utf-8"))["turns"]
             if t.get("rag_top") and t.get("gold_doc_ids")]
    agg = defaultdict(lambda: {"n": 0, "r1": 0, "r3": 0, "rr": 0.0})
    for t in turns:
        gold = set(t["gold_doc_ids"]); top = t["rag_top"]
        for key in ("ВСЕ", t["company"]):
            a = agg[key]; a["n"] += 1
            if gold & set(top[:1]): a["r1"] += 1
            if gold & set(top[:3]): a["r3"] += 1
            rr = next((1/(i+1) for i, d in enumerate(top) if d in gold), 0.0)
            a["rr"] += rr
    return {k: {"n": v["n"], "R@1": v["r1"]/v["n"], "R@3": v["r3"]/v["n"],
                "MRR": v["rr"]/v["n"]} for k, v in agg.items()}


print("=" * 72)
print("СРЕЗ 1: E2E (разговорный запрос = реплика клиента) — реальные компании")
print("=" * 72)
for k, m in sorted(e2e_metrics(R / "e2e_real_result.json").items(), key=lambda x: x[0] != "ВСЕ"):
    print(f"  {k:12s} n={m['n']:3d}  R@1={m['R@1']:.3f}  R@3={m['R@3']:.3f}  MRR={m['MRR']:.3f}")

print("\n" + "=" * 72)
print("СРЕЗ 2: E2E (разговорный) — синтетика «Вектор»")
print("=" * 72)
for k, m in sorted(e2e_metrics(R / "e2e_dialogues_result.json").items(), key=lambda x: x[0] != "ВСЕ"):
    print(f"  {k:12s} n={m['n']:3d}  R@1={m['R@1']:.3f}  R@3={m['R@3']:.3f}  MRR={m['MRR']:.3f}")

print("\n" + "=" * 72)
print("СРЕЗ 3: чистые запросы (gold-вопрос) — реальные компании (из rag_multi_eval.json)")
print("=" * 72)
clean = json.loads((R / "rag_multi_eval.json").read_text(encoding="utf-8"))[0]
print(f"  {'ВСЕ':12s} n={clean['n_queries']}  R@1={clean['recall@1']:.3f}  "
      f"R@3={clean['recall@3']:.3f}  R@5={clean['recall@5']:.3f}  MRR={clean['mrr']:.3f}  "
      f"(эмбеддер {clean['model']})")
for c, m in clean["by_company"].items():
    print(f"  {c:12s} n={m['n']:3d}  R@1={m['recall@1']:.3f}  R@3={m['recall@3']:.3f}  "
          f"R@5={m['recall@5']:.3f}  MRR={m['mrr']:.3f}")
