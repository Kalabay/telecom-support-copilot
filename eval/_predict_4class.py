"""Предсказания одной модели на Dusha test с фильтрацией по источнику."""

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
SETUPS_DIR = Path(r"K:\.caches\dusha_setups\paper_setups\test")


def tta_variants(data: "np.ndarray") -> list["np.ndarray"]:
    """4 версии сигнала для test-time augmentation: orig, ×1.05, ×0.95, +шум."""
    import librosa
    out = [data]
    try:
        out.append(librosa.effects.time_stretch(data, rate=1.05))
        out.append(librosa.effects.time_stretch(data, rate=0.95))
    except Exception:  # noqa: BLE001
        pass
    noise = np.random.RandomState(42).randn(len(data)).astype("float32") * 0.005
    out.append((data + noise).astype("float32"))
    return out


def load_manifest(source: str) -> dict[str, str]:
    """id -> emotion из official Dusha paper_setups manifest."""
    path = SETUPS_DIR / f"{source}_test.jsonl"
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        emo = rec["emotion"].lower()
        if emo in C2I:
            out[rec["id"]] = emo
    return out


def collect_crowd(manifest_ids: dict[str, str], n_target: int) -> list[tuple[str, str, bytes]]:
    """Стрим xbgoose/dusha, берём только id из manifest (crowd), до n_target."""
    ds = load_dataset("xbgoose/dusha", split="test", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    items = []
    scanned = 0
    for item in ds:
        if len(items) >= n_target:
            break
        scanned += 1
        path = item["audio"].get("path", "")
        fid = Path(path).stem
        if fid not in manifest_ids:
            continue
        raw = item["audio"].get("bytes")
        if not raw:
            continue
        items.append((fid, manifest_ids[fid], raw))
    print(f"crowd: scanned {scanned}, kept {len(items)}", flush=True)
    return items


def collect_podcast_npy(manifest_ids: dict[str, str], features_dir: Path,
                         n_target: int) -> list[tuple[str, str, np.ndarray]]:
    """Берём .npy-фичи подкаста (если скачан features.tar)."""
    items = []
    for fid, emo in manifest_ids.items():
        if len(items) >= n_target:
            break
        npy = features_dir / f"{fid}.npy"
        if not npy.exists():
            continue
        items.append((fid, emo, np.load(npy)))
    print(f"podcast: found {len(items)} npy files", flush=True)
    return items



def predict_gigaam_audio(items: list[tuple[str, str, bytes]], tta: bool = False) -> list[dict]:
    import torch
    _orig_load = torch.load
    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
    import gigaam
    print(f"loading GigaAM… (tta={tta})", flush=True)
    m = gigaam.load_model("emo")
    torch.load = _orig_load
    tmpdir = Path(tempfile.mkdtemp(prefix="ga_"))
    tmp = tmpdir / "in.wav"
    out, t0 = [], time.perf_counter()
    for n, (fid, true_label, raw) in enumerate(items, 1):
        try:
            data, sr = sf.read(io.BytesIO(raw), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            variants = tta_variants(data) if tta else [data]
            acc = np.zeros(4, dtype=np.float64)
            for vi, var in enumerate(variants):
                vp = tmpdir / f"v{vi}.wav"
                sf.write(vp, var, sr, subtype="PCM_16")
                probs = m.get_probs(str(vp))
                pv = np.array([probs.get(c, 0.0) for c in CLASSES], dtype=np.float64)
                s = pv.sum()
                acc += pv / s if s > 0 else pv
            vec = acc / len(variants)
            s = vec.sum()
            if s > 0:
                vec = vec / s
            out.append({"id": fid, "true": C2I[true_label], "pred": int(vec.argmax()),
                         "probs": [round(float(x), 6) for x in vec]})
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {fid}: {exc}", flush=True)
        if n % 100 == 0:
            print(f"  {n}/{len(items)}  {time.perf_counter()-t0:.0f}s", flush=True)
    return out


def predict_gigaam_npy(items: list[tuple[str, str, np.ndarray]]) -> list[dict]:
    """GigaAM inference на pre-computed фичах (tensor = [T, F])."""
    import gigaam
    print("loading GigaAM…", flush=True)
    m = gigaam.load_model("emo")
    if items:
        print(f"npy shape example: {items[0][2].shape}", flush=True)
    out, t0 = [], time.perf_counter()
    for n, (fid, true_label, feat) in enumerate(items, 1):
        try:
            raise NotImplementedError("GigaAM требует raw audio, npy-фичи несовместимы")
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {fid}: {exc}", flush=True)
        if n % 100 == 0:
            print(f"  {n}/{len(items)}  {time.perf_counter()-t0:.0f}s", flush=True)
    return out



def predict_hubert_audio(items: list[tuple[str, str, bytes]], tta: bool = False) -> list[dict]:
    import math
    import torch
    import librosa
    from app.pipeline.ser import _CLASS_PRIOR, _LA_TAU, get_recognizer
    print(f"loading HuBERT-Dusha… (tta={tta})", flush=True)
    rec = get_recognizer()
    rec._ensure_loaded()
    model, extractor, dev = rec._model, rec._extractor, rec._device
    id2name = {i: model.config.id2label[i].lower() for i in model.config.id2label}
    map5to4 = {i: C2I[n] for i, n in id2name.items() if n in C2I}
    log_prior = np.array([math.log(_CLASS_PRIOR[c]) for c in CLASSES])
    out, t0 = [], time.perf_counter()
    for n, (fid, true_label, raw) in enumerate(items, 1):
        try:
            data, sr = sf.read(io.BytesIO(raw), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != 16000:
                data = librosa.resample(data, orig_sr=sr, target_sr=16000)
            variants = tta_variants(data) if tta else [data]
            acc = np.zeros(4, dtype=np.float64)
            for var in variants:
                inp = extractor(var, sampling_rate=16000, return_tensors="pt", padding=True)
                inp = {k: v.to(dev) for k, v in inp.items()}
                with torch.inference_mode():
                    logits = model(**inp).logits
                    probs5 = torch.softmax(logits, -1).cpu().numpy()[0]
                pv = np.zeros(4)
                for i5, i4 in map5to4.items():
                    pv[i4] = probs5[i5]
                s = pv.sum()
                acc += pv / s if s > 0 else pv
            vec = acc / len(variants)
            s = vec.sum()
            if s > 0:
                vec /= s
            out.append({"id": fid, "true": C2I[true_label],
                         "pred_argmax": int(vec.argmax()),
                         "pred_la": int((np.log(np.clip(vec, 1e-9, 1)) - _LA_TAU * log_prior).argmax()),
                         "probs": [round(float(x), 6) for x in vec]})
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {fid}: {exc}", flush=True)
        if n % 100 == 0:
            print(f"  {n}/{len(items)}  {time.perf_counter()-t0:.0f}s", flush=True)
    return out



def predict_kelon_audio(items: list[tuple[str, str, bytes]]) -> list[dict]:
    """3-й аудио-бэкбон: wav2vec2-XLS-R, обучен на Dusha (5-class -> 4-class)."""
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    from app.pipeline.ser import get_recognizer

    name = "KELONMYOSA/wav2vec2-xls-r-300m-emotion-ru"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import transformers.modeling_utils as _mu
    _mu.PreTrainedModel.all_tied_weights_keys = {}
    print(f"loading {name} on {dev}…", flush=True)
    fe = AutoFeatureExtractor.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForAudioClassification.from_pretrained(
        name, trust_remote_code=True).to(dev).eval()
    id2label = {int(i): l.lower() for i, l in model.config.id2label.items()}
    map5to4 = {i: C2I[n] for i, n in id2label.items() if n in C2I}
    rec = get_recognizer()

    out, t0 = [], time.perf_counter()
    for n, (fid, true_label, raw) in enumerate(items, 1):
        try:
            wav, _ = rec.load_audio(raw)
            inp = fe(wav, sampling_rate=16000, return_tensors="pt", padding=True)
            inp = {k: v.to(dev) for k, v in inp.items()}
            with torch.inference_mode():
                probs5 = torch.softmax(model(**inp).logits, -1).cpu().numpy()[0]
            vec = np.zeros(4)
            for i5, i4 in map5to4.items():
                vec[i4] = probs5[i5]
            s = vec.sum()
            vec = vec / s if s > 0 else np.full(4, 0.25)
            out.append({"id": fid, "true": C2I[true_label], "pred": int(vec.argmax()),
                         "probs": [round(float(x), 6) for x in vec]})
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {fid}: {exc}", flush=True)
        if n % 100 == 0:
            print(f"  {n}/{len(items)}  {time.perf_counter()-t0:.0f}s", flush=True)
    return out



def predict_ours_audio(items: list[tuple[str, str, bytes]],
                        ckpt: str = r"K:\.caches\ser_finetuned\best") -> list[dict]:
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    from app.pipeline.ser import get_recognizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading наша FT-модель {ckpt} on {dev}…", flush=True)
    fe = AutoFeatureExtractor.from_pretrained(ckpt)
    model = AutoModelForAudioClassification.from_pretrained(ckpt).to(dev).eval()
    id2name = {int(i): l.lower() for i, l in model.config.id2label.items()}
    map5to4 = {i: C2I[n] for i, n in id2name.items() if n in C2I}
    rec = get_recognizer()
    out, t0 = [], time.perf_counter()
    for n, (fid, true_label, raw) in enumerate(items, 1):
        try:
            wav, _ = rec.load_audio(raw)
            inp = fe(wav, sampling_rate=16000, return_tensors="pt", padding=True)
            inp = {k: v.to(dev) for k, v in inp.items()}
            with torch.inference_mode():
                probs5 = torch.softmax(model(**inp).logits, -1).cpu().numpy()[0]
            vec = np.zeros(4)
            for i5, i4 in map5to4.items():
                vec[i4] = probs5[i5]
            s = vec.sum()
            vec = vec / s if s > 0 else np.full(4, 0.25)
            out.append({"id": fid, "true": C2I[true_label], "pred": int(vec.argmax()),
                         "probs": [round(float(x), 6) for x in vec]})
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {fid}: {exc}", flush=True)
        if n % 100 == 0:
            print(f"  {n}/{len(items)}  {time.perf_counter()-t0:.0f}s", flush=True)
    return out



def predict_text_audio(items: list[tuple[str, str, bytes]]) -> list[dict]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    sys.path.insert(0, str(PROJECT_ROOT / "eval"))
    from train_text_head import TextHead  # noqa: E402

    from app.pipeline.asr import get_asr
    from app.pipeline.ser import get_recognizer

    ckpt = torch.load(PROJECT_ROOT / "data" / "dusha_text" / "text_head.pt",
                       map_location="cpu", weights_only=False)
    text_classes = ckpt["classes"]
    t_map = {i: C2I[c] for i, c in enumerate(text_classes) if c in C2I}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading ASR + rubert-tiny2 on {dev}…", flush=True)
    asr = get_asr()
    asr._ensure_loaded()
    rec = get_recognizer()
    tok = AutoTokenizer.from_pretrained(ckpt["text_model"])
    base = AutoModel.from_pretrained(ckpt["text_model"]).to(dev).eval()
    head = TextHead().to(dev).eval()
    head.load_state_dict(ckpt["state_dict"])

    def embed(text: str) -> "torch.Tensor":
        enc = tok([text], padding=True, truncation=True, max_length=128,
                  return_tensors="pt").to(dev)
        with torch.inference_mode():
            out = base(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            return (out * mask).sum(1) / mask.sum(1).clamp_min(1)

    out, t0 = [], time.perf_counter()
    for n, (fid, true_label, raw) in enumerate(items, 1):
        try:
            wav, _ = rec.load_audio(raw)
            text = asr.transcribe(wav, "ru").text.strip()
            if not text:
                vec = np.full(4, 0.25)
            else:
                with torch.inference_mode():
                    logits5 = head(embed(text))
                    probs5 = torch.softmax(logits5, -1).cpu().numpy()[0]
                vec = np.zeros(4)
                for i5, i4 in t_map.items():
                    vec[i4] = probs5[i5]
                s = vec.sum()
                vec = vec / s if s > 0 else np.full(4, 0.25)
            out.append({"id": fid, "true": C2I[true_label], "pred": int(vec.argmax()),
                         "text": text, "probs": [round(float(x), 6) for x in vec]})
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {fid}: {exc}", flush=True)
        if n % 100 == 0:
            print(f"  {n}/{len(items)}  {time.perf_counter()-t0:.0f}s", flush=True)
    return out



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["gigaam", "hubert", "text", "kelon", "ours"])
    ap.add_argument("--source", choices=["crowd", "podcast"], required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--features-dir", type=Path, default=None,
                    help="Путь к распакованному features.tar (только для --source podcast)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tta", action="store_true", help="test-time augmentation (×4 версии)")
    args = ap.parse_args()

    manifest = load_manifest(args.source)
    print(f"manifest {args.source}: {len(manifest)} 4-class samples", flush=True)

    if args.source == "crowd":
        items = collect_crowd(manifest, args.n)
        if args.model == "gigaam":
            preds = predict_gigaam_audio(items, tta=args.tta)
        elif args.model == "hubert":
            preds = predict_hubert_audio(items, tta=args.tta)
        elif args.model == "kelon":
            preds = predict_kelon_audio(items)
        elif args.model == "ours":
            preds = predict_ours_audio(items)
        else:
            preds = predict_text_audio(items)

    else:
        if args.features_dir is None or not args.features_dir.exists():
            print("ERROR: --features-dir не указан или не существует.")
            print("  Аудио подкастов Sber не раздаёт (лицензия).")
            print("  Для inference нужен распакованный features.tar (30 ГБ).")
            print("  Скачать: https://cdn.chatwm.opensmodel.sberdevices.ru/dusha/features.tar")
            sys.exit(1)
        items_npy = collect_podcast_npy(manifest, args.features_dir, args.n)
        if args.model == "gigaam":
            preds = predict_gigaam_npy(items_npy)
        else:
            print("ERROR: HuBERT требует raw audio, .npy-фичи Dusha несовместимы.")
            sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"classes": CLASSES, "source": args.source,
                "model": args.model, "n_total": args.n, "preds": preds}
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {len(preds)} preds -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
