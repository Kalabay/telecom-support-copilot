"""Доказать, к какому источнику (crowd/podcast) относятся сэмплы xbgoose/dusha test."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import app  # noqa: F401,E402

from datasets import Audio, load_dataset  # noqa: E402

SETUPS = Path(r"K:\.caches\dusha_setups\paper_setups\test")
N = 500


def load_ids(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[rec["id"]] = rec["emotion"]
    return out


def main() -> None:
    crowd = load_ids(SETUPS / "crowd_test.jsonl")
    podcast = load_ids(SETUPS / "podcast_test.jsonl")
    print(f"crowd manifest ids={len(crowd)}, podcast ids={len(podcast)}, "
          f"overlap={len(set(crowd) & set(podcast))}")

    ds = load_dataset("xbgoose/dusha", split="test", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    in_crowd = in_podcast = neither = label_match = 0
    for i, item in enumerate(ds):
        if i >= N:
            break
        h = Path(item["audio"]["path"]).stem
        emo = str(item["emotion"]).lower()
        if h in crowd:
            in_crowd += 1
            if crowd[h].lower() == emo:
                label_match += 1
        elif h in podcast:
            in_podcast += 1
            if podcast[h].lower() == emo:
                label_match += 1
        else:
            neither += 1

    print(f"\nиз первых {N} сэмплов xbgoose/dusha test:")
    print(f"  в crowd_test   : {in_crowd}")
    print(f"  в podcast_test : {in_podcast}")
    print(f"  ни там ни там  : {neither}")
    print(f"  совпала метка с манифестом: {label_match}/{in_crowd + in_podcast}")


if __name__ == "__main__":
    main()
