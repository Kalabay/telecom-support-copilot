import json
from pathlib import Path

runs = json.loads(Path(r"K:\dev\coursework\eval\results\rag_multi_eval.json").read_text(encoding="utf-8"))
print(f"{'config':30s} R@1     R@3     R@5     MRR     p95ms")
for r in runs:
    name = r["model"].split("/")[-1] + ("+rr" if r["reranker"] else "")
    print(f"{name:30s} {r['recall@1']:.4f}  {r['recall@3']:.4f}  {r['recall@5']:.4f}  {r['mrr']:.4f}  {r['latency_ms_p95']}")
