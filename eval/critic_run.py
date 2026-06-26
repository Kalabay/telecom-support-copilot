"""T-lite + LLM-критик: прогнать critique() на готовых подсказках T-lite (сырой и фильтр)."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import app  # noqa: F401,E402
import sentence_transformers  # noqa: F401,E402
import torch  # noqa: F401,E402

R = PROJECT_ROOT / "eval" / "results"
COLS = ["tlite_raw", "tlite_critic", "tlite_filter", "tlite_filter_critic"]


def main() -> None:
    from app.pipeline.llm import get_generator
    gen = get_generator(); gen._ensure_loaded()
    print("LLM загружен", flush=True)

    rows = json.loads((R / "compare_data.json").read_text(encoding="utf-8"))
    rnd = random.Random(42)
    allrows, blind, mapping = [], [], []
    for i, d in enumerate(rows):
        src = [SimpleNamespace(snippet=s["snippet"], title=s["title"]) for s in d["sources"]]
        raw = d["tlite_raw"]
        filt = d["tlite_filtered"]
        _, crit_raw = gen.critique(raw, src)
        _, crit_filt = gen.critique(filt, src)
        cols = {"tlite_raw": raw, "tlite_critic": crit_raw,
                "tlite_filter": filt, "tlite_filter_critic": crit_filt}
        allrows.append({"dialogue_id": d["dialogue_id"], "turn_idx": d["turn_idx"],
                        "dataset": d["dataset"], "asr_text": d["asr_text"],
                        "ideal_text": d["ideal_text"], **cols})
        pairs = [(c, cols[c]) for c in COLS]
        rnd.shuffle(pairs)
        blind.append({"dialogue_id": d["dialogue_id"], "turn_idx": d["turn_idx"],
                      "asr_text": d["asr_text"], "ideal_text": d["ideal_text"],
                      **{f"v{j+1}": t for j, (_, t) in enumerate(pairs)}})
        mapping.append({"dialogue_id": d["dialogue_id"], "turn_idx": d["turn_idx"],
                        **{f"v{j+1}": c for j, (c, _) in enumerate(pairs)}})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    (R / "critic_all.json").write_text(json.dumps(allrows, ensure_ascii=False, indent=2), encoding="utf-8")
    (R / "critic_blind.json").write_text(json.dumps(blind, ensure_ascii=False, indent=2), encoding="utf-8")
    (R / "critic_map.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nготово: {len(allrows)} реплик -> critic_all.json / critic_blind.json / critic_map.json")


if __name__ == "__main__":
    main()
