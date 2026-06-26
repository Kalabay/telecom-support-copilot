"""emotion2vec+ large на Dusha crowd — probs (9->4) + эмбеддинги (1024-d)."""
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

import warnings  # noqa: E402
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import librosa  # noqa: E402
from datasets import Audio, load_dataset  # noqa: E402

R = PROJECT_ROOT / "eval" / "results"
SETUPS = Path(r"K:\.caches\dusha_setups\paper_setups\test")
TRAIN_MAN = Path(r"K:\.caches\dusha_train\manifest.jsonl")
CLASSES = ["neutral", "angry", "positive", "sad"]
C2I = {c: i for i, c in enumerate(CLASSES)}
E2V_MAP = {0: "angry", 3: "positive", 4: "neutral", 6: "sad"}


def iter_test(n):
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
            w, sr = sf.read(io.BytesIO(raw), dtype="float32")
        except Exception:  # noqa: BLE001
            continue
        if w.ndim > 1:
            w = w.mean(1)
        if sr != 16000:
            w = librosa.resample(w, orig_sr=sr, target_sr=16000)
        yield fid, w.astype("float32"), C2I[man[fid]]
        cnt += 1


def iter_train(n):
    rows = [json.loads(l) for l in TRAIN_MAN.read_text(encoding="utf-8").splitlines() if l.strip()]
    cnt = 0
    for r in rows:
        if cnt >= n:
            break
        emo = r["emotion"].lower()
        if emo not in C2I:
            continue
        try:
            w, sr = sf.read(r["path"], dtype="float32")
        except Exception:  # noqa: BLE001
            continue
        if w.ndim > 1:
            w = w.mean(1)
        if sr != 16000:
            w = librosa.resample(w, orig_sr=sr, target_sr=16000)
        yield r["id"], w.astype("float32"), C2I[emo]
        cnt += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "train"], required=True)
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()

    from funasr import AutoModel
    print("loading emotion2vec+ large...", flush=True)
    m = AutoModel(model="emotion2vec/emotion2vec_plus_large", hub="hf", disable_update=True)

    items = iter_test(args.n) if args.split == "test" else iter_train(args.n)
    ids, probs4, embs, labs = [], [], [], []
    t0 = time.perf_counter()
    for i, (fid, w, lab) in enumerate(items, 1):
        try:
            r = m.generate(w, granularity="utterance", extract_embedding=True)[0]
            sc = np.asarray(r["scores"], dtype=np.float64)
            vec = np.zeros(4)
            for e_idx, cls in E2V_MAP.items():
                vec[C2I[cls]] = sc[e_idx]
            s = vec.sum()
            vec = vec / s if s > 0 else np.full(4, 0.25)
            ids.append(fid)
            probs4.append([round(float(x), 6) for x in vec])
            embs.append(np.asarray(r["feats"], dtype="float32"))
            labs.append(lab)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {fid}: {exc}", flush=True)
        if i % 200 == 0:
            print(f"  {i}/{args.n}  {time.perf_counter()-t0:.0f}s", flush=True)

    if args.split == "test":
        preds = [{"id": ids[k], "true": int(labs[k]), "pred": int(np.argmax(probs4[k])),
                  "probs": probs4[k]} for k in range(len(ids))]
        (R / "_pred_e2v_crowd.json").write_text(json.dumps(
            {"classes": CLASSES, "model": "emotion2vec+large", "preds": preds},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved -> _pred_e2v_crowd.json ({len(ids)})", flush=True)

    np.savez_compressed(R / f"emb_e2v_{args.split}.npz",
                        ids=np.array(ids), emb=np.array(embs), labels=np.array(labs))
    print(f"saved -> emb_e2v_{args.split}.npz ({len(ids)}, dim={np.array(embs).shape[1]})", flush=True)


if __name__ == "__main__":
    main()
