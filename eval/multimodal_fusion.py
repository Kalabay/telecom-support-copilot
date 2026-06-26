"""Late fusion audio+text SER на Dusha test, с подбором веса на val."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import app  # noqa: F401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from datasets import Audio, load_dataset  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

from app.pipeline.ser import _CLASS_PRIOR, _LA_TAU, get_recognizer  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "eval"))
from train_text_head import TextHead  # noqa: E402

CLASSES = ["neutral", "angry", "positive", "sad", "other"]
C2I = {c: i for i, c in enumerate(CLASSES)}
TEST_JSONL = PROJECT_ROOT / "data" / "dusha_text" / "test.jsonl"
HEAD_PT = PROJECT_ROOT / "data" / "dusha_text" / "text_head.pt"
OUT = PROJECT_ROOT / "eval" / "results" / "multimodal_fusion.json"


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, dict]:
    per, f1s = {}, []
    for ci, c in enumerate(CLASSES):
        tp = int(((y_true == ci) & (y_pred == ci)).sum())
        fp = int(((y_true != ci) & (y_pred == ci)).sum())
        fn = int(((y_true == ci) & (y_pred != ci)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per[c] = {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}
        f1s.append(f1)
    return float(np.mean(f1s)), per


def la_predict(probs: np.ndarray) -> np.ndarray:
    log_prior = np.array([math.log(_CLASS_PRIOR[c]) for c in CLASSES])
    return (np.log(np.clip(probs, 1e-9, 1.0)) - _LA_TAU * log_prior).argmax(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    args = ap.parse_args()
    console = Console()
    console.rule(f"[bold cyan]Late fusion audio+text[/]  n={args.n}")

    if not TEST_JSONL.exists():
        console.print(f"[red]Нет {TEST_JSONL}[/] — запусти "
                      f"transcribe_dusha.py --split test --n {args.n}")
        sys.exit(1)
    tlines = [json.loads(l) for l in TEST_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    text_by_idx = {d["idx"]: d["text"] for d in tlines}
    console.log(f"транскриптов в test.jsonl: {len(tlines)} (min idx {min(text_by_idx)}, "
                f"max idx {max(text_by_idx)})")

    if not HEAD_PT.exists():
        console.print(f"[red]Нет {HEAD_PT}[/] — запусти train_text_head.py")
        sys.exit(1)
    ckpt = torch.load(HEAD_PT, weights_only=False)
    rec = get_recognizer()
    rec._ensure_loaded()  # noqa: SLF001
    dev = rec._device  # noqa: SLF001
    audio_model = rec._model  # noqa: SLF001
    extractor = rec._extractor  # noqa: SLF001
    id2idx = {audio_model.config.id2label[i].lower(): i for i in audio_model.config.id2label}

    console.log(f"Гружу text encoder {ckpt['text_model']}…")
    tok = AutoTokenizer.from_pretrained(ckpt["text_model"])
    text_base = AutoModel.from_pretrained(ckpt["text_model"]).to(dev).eval()
    for p in text_base.parameters():
        p.requires_grad = False
    text_head = TextHead().to(dev).eval()
    text_head.load_state_dict(ckpt["state_dict"])

    ds = load_dataset("xbgoose/dusha", split="test", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    A, T, Y = [], [], []
    used = 0
    t0 = time.perf_counter()
    for idx, item in enumerate(ds):
        if used >= args.n:
            break
        emo = str(item["emotion"]).lower().strip()
        if emo not in C2I:
            continue
        raw = item["audio"].get("bytes")
        if not raw:
            continue
        wav, _ = rec.load_audio(raw)
        if wav.size < 16000 * 0.3:
            continue
        if wav.shape[0] > 16000 * 10:
            wav = wav[: 16000 * 10]
        text = text_by_idx.get(idx, "").strip()
        if not text:
            continue

        inp = extractor(wav, sampling_rate=16000, return_tensors="pt", padding=True)
        inp = {k: v.to(dev) for k, v in inp.items()}
        with torch.inference_mode():
            a_logits = audio_model(**inp).logits
            a_probs = torch.softmax(a_logits, -1).cpu().numpy()[0]
        a_probs = np.array([a_probs[id2idx[c]] for c in CLASSES])

        enc = tok([text], padding=True, truncation=True, max_length=128,
                  return_tensors="pt").to(dev)
        with torch.inference_mode():
            t_out = text_base(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            t_pooled = (t_out * mask).sum(1) / mask.sum(1).clamp_min(1)
            t_logits = text_head(t_pooled.float())
            t_probs = torch.softmax(t_logits, -1).cpu().numpy()[0]

        A.append(a_probs)
        T.append(t_probs)
        Y.append(C2I[emo])
        used += 1
        if used % 100 == 0:
            console.log(f"  {used}/{args.n}")
    console.log(f"Извлечено {used} пар за {time.perf_counter()-t0:.0f}s")

    A = np.stack(A); T = np.stack(T); Y = np.array(Y)
    half = len(Y) // 2
    A_te, T_te, y_te = A[:half], T[:half], Y[:half]
    A_va, T_va, y_va = A[half:], T[half:], Y[half:]

    audio_argmax_f1, audio_per = macro_f1(y_te, A_te.argmax(1))
    audio_la_f1, audio_la_per = macro_f1(y_te, la_predict(A_te))
    text_argmax_f1, text_per = macro_f1(y_te, T_te.argmax(1))

    grid = [round(a, 2) for a in np.arange(0.0, 1.01, 0.05)]
    best_alpha, best_val_f1 = 1.0, -1.0
    val_curve = {}
    for a in grid:
        fused_val = a * A_va + (1 - a) * T_va
        f1, _ = macro_f1(y_va, fused_val.argmax(1))
        val_curve[a] = round(f1, 4)
        if f1 > best_val_f1:
            best_val_f1, best_alpha = f1, a
    best_alpha_la, best_val_la = 1.0, -1.0
    val_curve_la = {}
    for a in grid:
        fused_val = a * A_va + (1 - a) * T_va
        pred = la_predict(fused_val)
        f1, _ = macro_f1(y_va, pred)
        val_curve_la[a] = round(f1, 4)
        if f1 > best_val_la:
            best_val_la, best_alpha_la = f1, a

    fused_te = best_alpha * A_te + (1 - best_alpha) * T_te
    fusion_f1, fusion_per = macro_f1(y_te, fused_te.argmax(1))
    fused_te_la = best_alpha_la * A_te + (1 - best_alpha_la) * T_te
    fusion_la_f1, fusion_la_per = macro_f1(y_te, la_predict(fused_te_la))

    cmp = Table(title=f"macro-F1 на TEST (n={half}, val для подбора {len(y_va)})")
    cmp.add_column("вариант", style="cyan")
    cmp.add_column("macro-F1", justify="right")
    cmp.add_column("delta vs audio-LA", justify="right")
    base = audio_la_f1
    def row(name, v):
        d = v - base
        col = "green" if d > 0 else ("yellow" if abs(d) < 0.005 else "red")
        cmp.add_row(name, f"{v:.4f}", f"[{col}]{d:+.4f}[/]")
    row("audio only argmax", audio_argmax_f1)
    row("audio + logit adjust  ← текущий продакшен", audio_la_f1)
    row("text only argmax", text_argmax_f1)
    row(f"late fusion argmax (α={best_alpha})", fusion_f1)
    row(f"late fusion + LA (α={best_alpha_la})", fusion_la_f1)
    console.print(cmp)

    per_t = Table(title="Per-class F1 (TEST): audio+LA vs best fusion+LA")
    per_t.add_column("class", style="cyan")
    per_t.add_column("audio+LA F1", justify="right")
    per_t.add_column("fusion+LA F1", justify="right")
    per_t.add_column("Δ", justify="right")
    for c in CLASSES:
        a = audio_la_per[c]["f1"]
        f = fusion_la_per[c]["f1"]
        d = f - a
        col = "green" if d > 0.005 else ("red" if d < -0.005 else "")
        per_t.add_row(c, f"{a:.3f}", f"{f:.3f}", f"[{col}]{d:+.3f}[/]" if col else f"{d:+.3f}")
    console.print(per_t)

    verdict_la = ("ПОМОГ" if fusion_la_f1 > audio_la_f1 + 0.003
                  else ("без эффекта" if abs(fusion_la_f1 - audio_la_f1) <= 0.003
                        else "ХУЖЕ"))
    console.print(f"\n[bold]Вывод:[/] late fusion + LA {verdict_la} относительно audio+LA "
                  f"({fusion_la_f1:.4f} vs {audio_la_f1:.4f}, Δ {fusion_la_f1-audio_la_f1:+.4f}, "
                  f"α={best_alpha_la})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "n_test": half, "n_val": len(y_va),
        "audio_argmax_f1": round(audio_argmax_f1, 4),
        "audio_la_f1": round(audio_la_f1, 4),
        "text_argmax_f1": round(text_argmax_f1, 4),
        "fusion_argmax_f1": round(fusion_f1, 4),
        "fusion_la_f1": round(fusion_la_f1, 4),
        "best_alpha_argmax": best_alpha,
        "best_alpha_la": best_alpha_la,
        "val_curve_argmax": val_curve,
        "val_curve_la": val_curve_la,
        "audio_la_per_class": audio_la_per,
        "fusion_la_per_class": fusion_la_per,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    console.log(f"[bold green]Saved[/] -> {OUT}")


if __name__ == "__main__":
    main()
