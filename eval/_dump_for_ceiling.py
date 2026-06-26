import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"K:\dev\coursework\backend")))
import app  # noqa

from app.pipeline.rag import get_retriever

R = Path(r"K:\dev\coursework\eval\results")
full = {r["id"]: r for r in json.loads((R / "e2e_full.json").read_text(encoding="utf-8"))["rows"]}
retr = get_retriever()

out = []
for cid in sorted(full):
    r = full[cid]
    res = retr.search(r["text"], k=3, company=r["company"])
    kb = [{"doc_id": s.doc_id, "snippet": s.snippet} for s in res.sources]
    out.append({
        "id": cid, "company": r["company"], "emotion": r["emotion"],
        "text": r["text"],
        "kb": kb,
        "tlite": r["suggestions"],
    })

(R / "_ceiling_input.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"saved {len(out)} cases -> _ceiling_input.json")
