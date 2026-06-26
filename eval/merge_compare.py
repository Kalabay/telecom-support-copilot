from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.models.schemas import Emotion, EmotionState  # noqa: E402
from app.pipeline.llm import _safety_rank  # noqa: E402

R = PROJECT_ROOT / "eval" / "results"
data = {(d["dialogue_id"], d["turn_idx"]): d
        for d in json.loads((R / "compare_data.json").read_text(encoding="utf-8"))}
claude = json.loads((R / "compare_claude.json").read_text(encoding="utf-8"))

emo = EmotionState(label=Emotion.ANGRY, confidence=0.85, arousal=0.85,
                   valence=-0.7, escalation_risk=True)
out = []
for c in claude:
    k = (c["dialogue_id"], c["turn_idx"])
    d = data[k]
    variants = c.get("claude_variants") or []
    src = [SimpleNamespace(snippet=s["snippet"], title=s["title"]) for s in d["sources"]]
    craw = variants[0] if variants else ""
    cfilt = _safety_rank(variants, src, emo)[0] if variants else ""
    out.append({
        "dialogue_id": c["dialogue_id"], "turn_idx": c["turn_idx"],
        "dataset": d["dataset"], "company": d["company"],
        "asr_text": d["asr_text"], "ideal_text": d["ideal_text"],
        "tlite_raw": d["tlite_raw"], "tlite_filtered": d["tlite_filtered"],
        "claude_raw": craw, "claude_filtered": cfilt,
    })

(R / "compare_all.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
print(f"собрано {len(out)} реплик x 4 колонки -> compare_all.json")
