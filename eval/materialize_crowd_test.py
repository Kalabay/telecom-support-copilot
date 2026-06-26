"""Материализует 2000 crowd-test wav локально (только datasets, без torch — против segfault)."""
import io
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import app  # noqa: F401

import soundfile as sf
from datasets import Audio, load_dataset

SETUPS = Path(r"K:\.caches\dusha_setups\paper_setups\test")
OUT = Path(r"K:\.caches\dusha_test")
CLASSES = {"neutral", "angry", "positive", "sad"}
N = 2000

man = {}
for line in (SETUPS / "crowd_test.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        rec = json.loads(line)
        if rec["emotion"].lower() in CLASSES:
            man[rec["id"]] = rec["emotion"].lower()

wav_dir = OUT / "wav"
wav_dir.mkdir(parents=True, exist_ok=True)
ds = load_dataset("xbgoose/dusha", split="test", streaming=True).cast_column(
    "audio", Audio(decode=False))

mf = (OUT / "manifest.jsonl").open("w", encoding="utf-8")
cnt = 0
for item in ds:
    if cnt >= N:
        break
    fid = Path(item["audio"].get("path", "")).stem
    if fid not in man:
        continue
    raw = item["audio"].get("bytes")
    if not raw:
        continue
    try:
        w, sr = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception:  # noqa: BLE001
        continue
    if w.ndim > 1:
        w = w.mean(1)
    wp = wav_dir / f"{fid}.wav"
    sf.write(wp, w, sr, subtype="PCM_16")
    mf.write(json.dumps({"id": fid, "path": str(wp), "emotion": man[fid]},
                        ensure_ascii=False) + "\n")
    cnt += 1
    if cnt % 300 == 0:
        print(f"  {cnt}/{N}", flush=True)
mf.close()
print(f"materialized {cnt} crowd-test wav -> {wav_dir}", flush=True)
