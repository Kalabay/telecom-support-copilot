"""RESD текстовый канал: ASR транскрибирует RESD-аудио -> CEDR-эмбеддинг."""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import app  # noqa: F401,E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from datasets import Audio, load_dataset  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

CLASSES = ["neutral", "angry", "positive", "sad"]
C2I = {c: i for i, c in enumerate(CLASSES)}
RESD_MAP = {"anger": "angry", "happiness": "positive", "enthusiasm": "positive",
            "sadness": "sad", "neutral": "neutral"}
CEDR = "seara/rubert-base-cased-russian-emotion-detection-cedr"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from app.pipeline.asr import get_asr
    asr = get_asr(); asr._ensure_loaded()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(CEDR)
    model = AutoModelForSequenceClassification.from_pretrained(CEDR).to(dev).eval()
    store = {}
    model.classifier.register_forward_pre_hook(
        lambda m, inp: store.__setitem__("e", inp[0].detach().cpu().numpy()))

    ds = load_dataset("Aniemore/resd", split=args.split, streaming=True).cast_column(
        "speech", Audio(decode=False))
    ids, embs, labs, texts = [], [], [], []
    cnt = 0
    import librosa
    for k, item in enumerate(ds):
        if cnt >= args.n:
            break
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
        if sr != 16000:
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        text = asr.transcribe(data.astype("float32"), "ru").text.strip()
        enc = tok([text or "."], padding=True, truncation=True, max_length=128,
                  return_tensors="pt").to(dev)
        with torch.inference_mode():
            model(**enc)
        fid = str(item.get("name") or f"resd_{args.split}_{k}")
        ids.append(fid); embs.append(store["e"][0].astype("float32"))
        labs.append(C2I[RESD_MAP[emo]]); texts.append(text)
        cnt += 1
        if cnt % 200 == 0:
            print(f"  {cnt}", flush=True)
    embs = np.array(embs)
    print(f"RESD text эмбеддингов: {len(ids)} dim={embs.shape[1]}", flush=True)
    print("примеры текста:", texts[:5], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, ids=np.array(ids), emb=embs, labels=np.array(labs),
                        texts=np.array(texts, dtype=object))
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
