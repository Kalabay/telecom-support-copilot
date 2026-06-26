"""Материализовать RESD локально (wav + manifest), эмоция -> dusha-4 имя."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import app  # noqa: F401,E402

import soundfile as sf  # noqa: E402
from datasets import Audio, load_dataset  # noqa: E402

RESD_MAP = {"anger": "angry", "happiness": "positive", "enthusiasm": "positive",
            "sadness": "sad", "neutral": "neutral"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    wav_dir = args.out / "wav"; wav_dir.mkdir(parents=True, exist_ok=True)
    man = args.out / "manifest.jsonl"

    ds = load_dataset("Aniemore/resd", split=args.split, streaming=True).cast_column(
        "speech", Audio(decode=False))
    w = 0
    with man.open("w", encoding="utf-8") as mf:
        for k, item in enumerate(ds):
            emo = str(item["emotion"]).lower()
            if emo not in RESD_MAP:
                continue
            au = item["speech"]; raw = au.get("bytes")
            try:
                data, sr = sf.read(io.BytesIO(raw), dtype="float32") if raw else sf.read(au["path"], dtype="float32")
            except Exception:  # noqa: BLE001
                continue
            if data.ndim > 1:
                data = data.mean(axis=1)
            fid = str(item.get("name") or f"resd_{args.split}_{k}")
            wp = wav_dir / f"{fid}.wav"
            sf.write(wp, data, sr, subtype="PCM_16")
            mf.write(json.dumps({"id": fid, "path": str(wp), "emotion": RESD_MAP[emo]},
                                 ensure_ascii=False) + "\n")
            w += 1
    print(f"RESD {args.split}: {w} wav -> {wav_dir}", flush=True)


if __name__ == "__main__":
    main()
