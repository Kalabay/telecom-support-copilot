"""Прямое сравнение GigaAM-Emo (Sber, MIT) vs HuBERT-Dusha (наш текущий +LA)."""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import app  # noqa: F401,E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from datasets import Audio, load_dataset  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

import gigaam  # noqa: E402

from app.pipeline.ser import _CLASS_PRIOR, _LA_TAU, get_recognizer  # noqa: E402

CLASSES = ["neutral", "angry", "positive", "sad"]
C2I = {c: i for i, c in enumerate(CLASSES)}
OUT = PROJECT_ROOT / "eval" / "results" / "ser_compare_gigaam.json"


def macro_f1(y_true, y_pred):  # noqa: ANN001
    yt, yp = np.array(y_true), np.array(y_pred)
    per, f1s = {}, []
    for ci, c in enumerate(CLASSES):
        tp = int(((yt == ci) & (yp == ci)).sum())
        fp = int(((yt != ci) & (yp == ci)).sum())
        fn = int(((yt == ci) & (yp != ci)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per[c] = {"precision": round(prec, 3), "recall": round(rec, 3),
                  "f1": round(f1, 3), "support": int((yt == ci).sum())}
        f1s.append(f1)
    return float(np.mean(f1s)), per


def la_predict_4class(probs_4: np.ndarray) -> int:
    """logit adjustment по приору из ser.py, но только по 4 классам."""
    log_prior = np.array([math.log(_CLASS_PRIOR[c]) for c in CLASSES])
    score = np.log(np.clip(probs_4, 1e-9, 1)) - _LA_TAU * log_prior
    return int(score.argmax())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    args = ap.parse_args()
    console = Console()
    console.rule(f"[bold cyan]GigaAM-Emo vs HuBERT+LA[/]  Dusha test 4-class, n до {args.n}")

    console.log("Загружаю GigaAM-Emo…")
    ga = gigaam.load_model("emo")
    ga_idx = {n.lower(): i for i, n in enumerate(ga.id2name)}
    console.log(f"GigaAM классы: {ga.id2name}")

    console.log("Загружаю HuBERT-Dusha (CPU)…")
    rec = get_recognizer()
    rec._ensure_loaded()  # noqa: SLF001
    hu_model = rec._model  # noqa: SLF001
    hu_extractor = rec._extractor  # noqa: SLF001
    hu_dev = rec._device  # noqa: SLF001
    hu_id2name = {i: hu_model.config.id2label[i].lower() for i in hu_model.config.id2label}
    hu_5to4 = {i: C2I[n] for i, n in hu_id2name.items() if n in C2I}
    console.log(f"HuBERT классы: {list(hu_id2name.values())} (используем 4 из 5)")

    ds = load_dataset("xbgoose/dusha", split="test", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    y_true = []
    pred_ga, pred_hu_argmax, pred_hu_la = [], [], []
    lat_ga, lat_hu = [], []
    tmp_dir = tempfile.mkdtemp(prefix="ser_compare_")
    tmp_path = Path(tmp_dir) / "in.wav"

    seen = 0
    skipped = 0
    t0 = time.perf_counter()
    import torch
    for item in ds:
        if seen >= args.n:
            break
        true = str(item["emotion"]).lower().strip()
        if true not in C2I:
            continue
        raw = item["audio"].get("bytes")
        if not raw:
            skipped += 1
            continue

        try:
            data, sr = sf.read(io.BytesIO(raw), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        if data.size < sr * 0.3:
            skipped += 1
            continue

        try:
            sf.write(tmp_path, data, sr, subtype="PCM_16")
            t = time.perf_counter()
            ga_probs = ga.get_probs(str(tmp_path))
            lat_ga.append(int((time.perf_counter() - t) * 1000))
            ga_vec = np.array([ga_probs.get(c, 0.0) for c in CLASSES], dtype=np.float64)
        except Exception as exc:  # noqa: BLE001
            console.log(f"[yellow]GigaAM skip: {exc}[/]")
            skipped += 1
            continue
        p_ga = int(ga_vec.argmax())

        try:
            if sr != 16000:
                import librosa
                wav16 = librosa.resample(data, orig_sr=sr, target_sr=16000)
            else:
                wav16 = data
            inputs = hu_extractor(wav16, sampling_rate=16000, return_tensors="pt",
                                  padding=True)
            inputs = {k: v.to(hu_dev) for k, v in inputs.items()}
            t = time.perf_counter()
            with torch.inference_mode():
                hu_logits = hu_model(**inputs).logits
                hu_probs5 = torch.softmax(hu_logits, -1).cpu().numpy()[0]
            lat_hu.append(int((time.perf_counter() - t) * 1000))
            hu_vec = np.zeros(4)
            for i5, i4 in hu_5to4.items():
                hu_vec[i4] = hu_probs5[i5]
            s = hu_vec.sum()
            if s > 0:
                hu_vec /= s
        except Exception as exc:  # noqa: BLE001
            console.log(f"[yellow]HuBERT skip: {exc}[/]")
            skipped += 1
            continue
        p_hu_a = int(hu_vec.argmax())
        p_hu_la = la_predict_4class(hu_vec)

        y_true.append(C2I[true])
        pred_ga.append(p_ga)
        pred_hu_argmax.append(p_hu_a)
        pred_hu_la.append(p_hu_la)
        seen += 1
        if seen % 50 == 0:
            ga_f1_now, _ = macro_f1(y_true, pred_ga)
            hu_la_f1_now, _ = macro_f1(y_true, pred_hu_la)
            console.log(f"  {seen}/{args.n}  GigaAM={ga_f1_now:.3f}  "
                        f"HuBERT+LA={hu_la_f1_now:.3f}")

    elapsed = time.perf_counter() - t0
    console.log(f"Готово: {seen} сэмплов оценено за {elapsed:.0f}s ({skipped} пропущено)")

    ga_f1, ga_per = macro_f1(y_true, pred_ga)
    hu_a_f1, hu_a_per = macro_f1(y_true, pred_hu_argmax)
    hu_la_f1, hu_la_per = macro_f1(y_true, pred_hu_la)

    cmp = Table(title=f"macro-F1 на Dusha test, 4 класса (n={seen})")
    cmp.add_column("модель", style="cyan")
    cmp.add_column("macro-F1", justify="right")
    cmp.add_column("vs GigaAM", justify="right")
    cmp.add_row("HuBERT-Dusha argmax", f"{hu_a_f1:.4f}", f"{hu_a_f1-ga_f1:+.4f}")
    cmp.add_row("HuBERT-Dusha + logit adj (наш продакшен)", f"{hu_la_f1:.4f}",
                f"{hu_la_f1-ga_f1:+.4f}")
    cmp.add_row("[bold]GigaAM-Emo (Sber)[/]", f"[bold]{ga_f1:.4f}[/]", "—")
    console.print(cmp)

    per_t = Table(title="Per-class F1: HuBERT+LA vs GigaAM")
    per_t.add_column("class", style="cyan")
    per_t.add_column("HuBERT+LA", justify="right")
    per_t.add_column("GigaAM", justify="right")
    per_t.add_column("Δ (Ga-Hu)", justify="right")
    per_t.add_column("support", justify="right")
    for c in CLASSES:
        h = hu_la_per[c]["f1"]
        g = ga_per[c]["f1"]
        d = g - h
        col = "green" if d > 0.005 else ("red" if d < -0.005 else "")
        per_t.add_row(c, f"{h:.3f}", f"{g:.3f}",
                      f"[{col}]{d:+.3f}[/]" if col else f"{d:+.3f}",
                      str(ga_per[c]["support"]))
    console.print(per_t)

    lat_t = Table(title="Latency (CPU)")
    lat_t.add_column("модель", style="cyan")
    lat_t.add_column("mean ms", justify="right")
    lat_t.add_column("p95 ms", justify="right")
    lat_t.add_row("HuBERT-Dusha", str(int(np.mean(lat_hu))),
                  str(int(np.percentile(lat_hu, 95))))
    lat_t.add_row("GigaAM-Emo", str(int(np.mean(lat_ga))),
                  str(int(np.percentile(lat_ga, 95))))
    console.print(lat_t)

    verdict = (
        f"GigaAM-Emo ПОБЕЖДАЕТ нас на {ga_f1 - hu_la_f1:+.4f}"
        if ga_f1 > hu_la_f1 + 0.005
        else f"GigaAM-Emo ≈ нам ({ga_f1:.4f} vs {hu_la_f1:.4f})"
        if abs(ga_f1 - hu_la_f1) <= 0.005
        else f"наш HuBERT+LA ПОБЕЖДАЕТ ({hu_la_f1:.4f} vs {ga_f1:.4f}, +{hu_la_f1 - ga_f1:.4f})"
    )
    console.print(f"\n[bold]Вывод:[/] {verdict}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "n_evaluated": seen, "n_skipped": skipped, "wall_sec": int(elapsed),
        "classes": CLASSES,
        "gigaam_emo_f1": round(ga_f1, 4),
        "hubert_argmax_f1": round(hu_a_f1, 4),
        "hubert_logit_adj_f1": round(hu_la_f1, 4),
        "gigaam_per_class": ga_per,
        "hubert_la_per_class": hu_la_per,
        "latency_ms": {
            "gigaam_mean": int(np.mean(lat_ga)),
            "gigaam_p95": int(np.percentile(lat_ga, 95)),
            "hubert_mean": int(np.mean(lat_hu)),
            "hubert_p95": int(np.percentile(lat_hu, 95)),
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    console.log(f"[bold green]Saved[/] -> {OUT}")


if __name__ == "__main__":
    main()
