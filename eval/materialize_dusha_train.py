"""Материализовать подмножество Dusha crowd train локально для дообучения."""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import app  # noqa: F401,E402

import soundfile as sf  # noqa: E402
from datasets import Audio, load_dataset  # noqa: E402

SETUPS = Path(r"K:\.caches\dusha_setups\paper_setups")
CLASSES5 = ["neutral", "angry", "positive", "sad", "other"]


def load_train_manifest() -> dict[str, str]:
    """id -> emotion из crowd train (dusha_large.jsonl или crowd_*.jsonl)."""
    path = SETUPS / "crowd_large.jsonl"
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[rec["id"]] = rec["emotion"].lower()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25000)
    ap.add_argument("--out", type=Path, default=Path(r"K:\.caches\dusha_train"))
    args = ap.parse_args()

    manifest = load_train_manifest()
    print(f"crowd train manifest: {len(manifest)} сэмплов", flush=True)

    wav_dir = args.out / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    man_path = args.out / "manifest.jsonl"

    ds = load_dataset("xbgoose/dusha", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    written, scanned, t0 = 0, 0, time.perf_counter()
    with man_path.open("w", encoding="utf-8") as mf:
        for item in ds:
            if written >= args.n:
                break
            scanned += 1
            path = item["audio"].get("path", "")
            fid = Path(path).stem
            emo = manifest.get(fid) or str(item.get("emotion", "")).lower()
            if emo not in CLASSES5:
                continue
            raw = item["audio"].get("bytes")
            if not raw:
                continue
            try:
                data, sr = sf.read(io.BytesIO(raw), dtype="float32")
            except Exception:  # noqa: BLE001
                continue
            wpath = wav_dir / f"{fid}.wav"
            if not wpath.exists():
                sf.write(wpath, data, sr, subtype="PCM_16")
            mf.write(json.dumps({"id": fid, "path": str(wpath), "emotion": emo},
                                 ensure_ascii=False) + "\n")
            written += 1
            if written % 1000 == 0:
                el = time.perf_counter() - t0
                print(f"  {written}/{args.n}  scanned={scanned}  {el:.0f}s "
                      f"({written/el:.0f}/s)", flush=True)

    print(f"готово: {written} wav -> {wav_dir}, manifest -> {man_path}", flush=True)


if __name__ == "__main__":
    main()
