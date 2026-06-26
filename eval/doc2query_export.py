import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import sentence_transformers  # noqa: F401,E402  (порядок импорта против access violation)
import chromadb  # noqa: E402
from chromadb.config import Settings as ChromaSettings  # noqa: E402

R = PROJECT_ROOT / "eval" / "results"
client = chromadb.PersistentClient(path=str(PROJECT_ROOT / "data" / "chroma"),
                                   settings=ChromaSettings(anonymized_telemetry=False))
col = client.get_collection("telecom_kb_frida")
got = col.get(include=["documents", "metadatas"])

docs = defaultdict(lambda: {"text": "", "company": "", "title": ""})
for txt, m in zip(got["documents"], got["metadatas"]):
    d = docs[m["doc_id"]]
    d["company"] = m["company"]; d["title"] = m.get("title", m["doc_id"])
    if len(d["text"]) < 1600:
        d["text"] = (d["text"] + " " + txt).strip()

out = [{"doc_id": k, "company": v["company"], "title": v["title"], "text": v["text"][:1600]}
       for k, v in docs.items()]
(R / "docs_for_doc2query.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"{len(out)} статей -> docs_for_doc2query.json")
