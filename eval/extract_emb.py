"""Извлечь frozen-эмбеддинги аудио-экспертов для fusion (pooled, до классификатора)."""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import app  # noqa: F401,E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from datasets import Audio, load_dataset  # noqa: E402

CLASSES = ["neutral", "angry", "positive", "sad"]
C2I = {c: i for i, c in enumerate(CLASSES)}
SETUPS = Path(r"K:\.caches\dusha_setups\paper_setups\test")
TRAIN_MANIFEST = Path(r"K:\.caches\dusha_train\manifest.jsonl")

RESD_MAP = {"anger": "angry", "happiness": "positive", "enthusiasm": "positive",
            "sadness": "sad", "neutral": "neutral"}


def iter_resd(split: str, n: int):
    """(id, wav16k, label4) из Aniemore/resd, маппинг 7->4."""
    import librosa
    from datasets import Audio, load_dataset
    ds = load_dataset("Aniemore/resd", split=split, streaming=True).cast_column(
        "speech", Audio(decode=False))
    cnt = 0
    for k, item in enumerate(ds):
        if cnt >= n:
            break
        emo = str(item["emotion"]).lower()
        if emo not in RESD_MAP:
            continue
        au = item["speech"]
        raw = au.get("bytes")
        try:
            if raw:
                data, sr = sf.read(io.BytesIO(raw), dtype="float32")
            else:
                data, sr = sf.read(au["path"], dtype="float32")
        except Exception:  # noqa: BLE001
            continue
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != 16000:
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        fid = str(item.get("name") or f"resd_{split}_{k}")
        yield fid, data.astype("float32"), C2I[RESD_MAP[emo]]
        cnt += 1


def iter_train(n: int):
    """(id, wav_float32_16k, label4) из локальных train wav, 4-class."""
    import librosa
    rows = [json.loads(l) for l in TRAIN_MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    cnt = 0
    for r in rows:
        if cnt >= n:
            break
        emo = r["emotion"].lower()
        if emo not in C2I:
            continue
        try:
            data, sr = sf.read(r["path"], dtype="float32")
        except Exception:  # noqa: BLE001
            continue
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != 16000:
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        yield r["id"], data.astype("float32"), C2I[emo]
        cnt += 1


def iter_manifest(path: str, n: int):
    """(id, wav16k, label4) из произвольного manifest.jsonl {id,path,emotion}."""
    import librosa
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    cnt = 0
    for r in rows:
        if cnt >= n:
            break
        emo = r["emotion"].lower()
        if emo not in C2I:
            continue
        try:
            data, sr = sf.read(r["path"], dtype="float32")
        except Exception:  # noqa: BLE001
            continue
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != 16000:
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        yield r["id"], data.astype("float32"), C2I[emo]
        cnt += 1


def iter_test(n: int):
    """(id, wav_float32_16k, label4) из стрима crowd test, те же 2000."""
    import librosa
    man = {}
    for line in (SETUPS / "crowd_test.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec["emotion"].lower() in C2I:
                man[rec["id"]] = rec["emotion"].lower()
    ds = load_dataset("xbgoose/dusha", split="test", streaming=True).cast_column(
        "audio", Audio(decode=False))
    cnt = 0
    for item in ds:
        if cnt >= n:
            break
        fid = Path(item["audio"].get("path", "")).stem
        if fid not in man:
            continue
        raw = item["audio"].get("bytes")
        if not raw:
            continue
        try:
            data, sr = sf.read(io.BytesIO(raw), dtype="float32")
        except Exception:  # noqa: BLE001
            continue
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != 16000:
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        yield fid, data.astype("float32"), C2I[man[fid]]
        cnt += 1


def extract_gigaam(items):
    import torch
    _orig = torch.load
    torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
    import gigaam
    m = gigaam.load_model("emo")
    torch.load = _orig
    store = {}
    h = m.head.register_forward_pre_hook(lambda mod, inp: store.__setitem__("e", inp[0].detach().cpu().numpy()))
    tmp = Path(tempfile.mkdtemp(prefix="gae_")) / "x.wav"
    ids, embs, labs = [], [], []
    t0 = time.perf_counter()
    for i, (fid, wav, lab) in enumerate(items, 1):
        try:
            sf.write(tmp, wav, 16000, subtype="PCM_16")
            m.get_probs(str(tmp))
            e = store["e"]
            e = e[0] if e.ndim == 2 else e
            ids.append(fid); embs.append(e.astype("float32")); labs.append(lab)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {fid}: {exc}", flush=True)
        if i % 500 == 0:
            print(f"  {i}  {time.perf_counter()-t0:.0f}s", flush=True)
    h.remove()
    return ids, np.array(embs), np.array(labs)


def extract_hubert(items):
    import torch
    from app.pipeline.ser import get_recognizer
    rec = get_recognizer()
    rec._ensure_loaded()
    model, extractor, dev = rec._model, rec._extractor, rec._device
    store = {}
    h = model.classifier.register_forward_pre_hook(
        lambda mod, inp: store.__setitem__("e", inp[0].detach().cpu().numpy()))
    ids, embs, labs = [], [], []
    t0 = time.perf_counter()
    for i, (fid, wav, lab) in enumerate(items, 1):
        try:
            enc = extractor(wav, sampling_rate=16000, return_tensors="pt", padding=True)
            enc = {k: v.to(dev) for k, v in enc.items()}
            with torch.inference_mode():
                model(**enc)
            e = store["e"]
            e = e[0] if e.ndim == 2 else e
            ids.append(fid); embs.append(e.astype("float32")); labs.append(lab)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {fid}: {exc}", flush=True)
        if i % 500 == 0:
            print(f"  {i}  {time.perf_counter()-t0:.0f}s", flush=True)
    h.remove()
    return ids, np.array(embs), np.array(labs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["gigaam", "hubert"])
    ap.add_argument("--split", required=True,
                    help="train/test (Dusha) или resd:train / resd:test")
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.split.startswith("resd:"):
        items = iter_resd(args.split.split(":", 1)[1], args.n)
    elif args.split.startswith("manifest:"):
        items = iter_manifest(args.split.split(":", 1)[1], args.n)
    else:
        items = iter_train(args.n) if args.split == "train" else iter_test(args.n)
    if args.model == "gigaam":
        ids, embs, labs = extract_gigaam(items)
    else:
        ids, embs, labs = extract_hubert(items)

    print(f"эмбеддингов: {len(ids)}, dim={embs.shape[1] if len(embs) else '?'}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, ids=np.array(ids), emb=embs, labels=labs)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
