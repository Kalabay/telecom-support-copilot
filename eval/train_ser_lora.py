"""LoRA continue-fine-tune SER от стокового чекпойнта (правильный способ дообучения)."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import app  # noqa: F401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from datasets import Audio, load_dataset  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from app.pipeline.ser import _CLASS_PRIOR, _LA_TAU, get_recognizer  # noqa: E402

CLASSES = ["neutral", "angry", "positive", "sad", "other"]
C2I = {c: i for i, c in enumerate(CLASSES)}
SR = 16000
MAX_LEN = SR * 6
OUT = PROJECT_ROOT / "eval" / "results" / "ser_lora.json"


def macro_f1(y_true, y_pred):  # noqa: ANN001
    per, f1s = {}, []
    yt, yp = np.array(y_true), np.array(y_pred)
    for ci, c in enumerate(CLASSES):
        tp = int(((yt == ci) & (yp == ci)).sum())
        fp = int(((yt != ci) & (yp == ci)).sum())
        fn = int(((yt == ci) & (yp != ci)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per[c] = {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}
        f1s.append(f1)
    return float(np.mean(f1s)), per


def collect(split: str, n: int, rec, console):  # noqa: ANN001
    ds = load_dataset("xbgoose/dusha", split=split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    waves, labels = [], []
    for item in ds:
        if len(waves) >= n:
            break
        true = str(item["emotion"]).lower().strip()
        if true not in C2I:
            continue
        raw = item["audio"].get("bytes")
        if not raw:
            continue
        wav, _ = rec.load_audio(raw)
        if wav.size < SR * 0.3:
            continue
        if wav.shape[0] > MAX_LEN:
            wav = wav[:MAX_LEN]
        waves.append(wav.astype(np.float32))
        labels.append(C2I[true])
        if len(waves) % 250 == 0:
            console.log(f"  {split}: {len(waves)}/{n}")
    return waves, labels


def la_predict(probs_np):  # noqa: ANN001
    """argmax с logit adjustment (как в живом ser.py)."""
    log_prior = np.array([math.log(_CLASS_PRIOR[c]) for c in CLASSES])
    scores = np.log(np.clip(probs_np, 1e-9, 1)) - _LA_TAU * log_prior
    return scores.argmax(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--test", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    console = Console()
    console.rule(f"[bold cyan]LoRA continue-FT[/]  train={args.train} test={args.test} ep={args.epochs}")

    rec = get_recognizer()
    rec._ensure_loaded()  # noqa: SLF001
    model = rec._model  # noqa: SLF001
    extractor = rec._extractor  # noqa: SLF001
    dev = rec._device  # noqa: SLF001

    t0 = time.perf_counter()
    console.log("Стримлю train в RAM…")
    Xtr, ytr = collect("train", args.train, rec, console)
    console.log("Стримлю test в RAM…")
    Xte, yte = collect("test", args.test, rec, console)
    console.log(f"Данные в RAM за {time.perf_counter()-t0:.0f}s. train={len(Xtr)} test={len(Xte)}")
    dist = Counter(ytr)
    console.log("train dist: " + ", ".join(f"{CLASSES[i]}={dist.get(i,0)}" for i in range(5)))

    def eval_model() -> tuple[float, dict, float, dict]:
        model.eval()
        probs = []
        with torch.inference_mode():
            for i in range(0, len(Xte), 16):
                batch = Xte[i:i + 16]
                inp = extractor(batch, sampling_rate=SR, return_tensors="pt", padding=True)
                inp = {k: v.to(dev) for k, v in inp.items()}
                logits = model(**inp).logits
                probs.append(torch.softmax(logits, -1).float().cpu().numpy())
        P = np.concatenate(probs)
        f1_raw, per_raw = macro_f1(yte, P.argmax(1))
        f1_la, per_la = macro_f1(yte, la_predict(P))
        return f1_raw, per_raw, f1_la, per_la

    base_raw, base_raw_per, base_la, base_la_per = eval_model()
    console.log(f"BASELINE (до дообучения): argmax {base_raw:.4f} | +LA {base_la:.4f}")

    lcfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        modules_to_save=["projector", "classifier"],
        bias="none",
    )
    model = get_peft_model(model, lcfg)
    model.to(dev)
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception:  # noqa: BLE001
            pass
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.log(f"LoRA обучаемых параметров: {trainable/1e6:.2f}M")

    counts = np.array([dist.get(i, 1) for i in range(5)], dtype=np.float32)
    cls_w = torch.tensor((counts.sum() / (5 * counts)), dtype=torch.float32, device=dev)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)
    idx_all = np.arange(len(Xtr))

    best = {"f1_la": base_la, "epoch": 0, "raw": base_raw}
    for ep in range(1, args.epochs + 1):
        model.train()
        np.random.shuffle(idx_all)
        running = 0.0
        opt.zero_grad()
        for step, i in enumerate(range(0, len(idx_all), args.bs)):
            bidx = idx_all[i:i + args.bs]
            batch = [Xtr[j] for j in bidx]
            lbl = torch.tensor([ytr[j] for j in bidx], device=dev)
            inp = extractor(batch, sampling_rate=SR, return_tensors="pt", padding=True)
            inp = {k: v.to(dev) for k, v in inp.items()}
            logits = model(**inp).logits
            loss = F.cross_entropy(logits, lbl, weight=cls_w, label_smoothing=0.1) / args.accum
            loss.backward()
            running += float(loss) * args.accum
            if (step + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad()
        n_steps = len(range(0, len(idx_all), args.bs))
        raw, raw_per, la, la_per = eval_model()
        console.log(f"epoch {ep}/{args.epochs}  loss={running/n_steps:.3f}  "
                    f"test argmax={raw:.4f} +LA={la:.4f}")
        if la > best["f1_la"]:
            best = {"f1_la": la, "f1_raw": raw, "epoch": ep,
                    "per_raw": raw_per, "per_la": la_per}

    cmp = Table(title="macro-F1 на TEST (n={})".format(len(yte)))
    cmp.add_column("вариант", style="cyan")
    cmp.add_column("argmax", justify="right")
    cmp.add_column("+logit adj", justify="right")
    cmp.add_row("стоковая (до LoRA)", f"{base_raw:.4f}", f"{base_la:.4f}")
    best_la = best.get("f1_la", base_la)
    best_raw = best.get("f1_raw", base_raw)
    d = best_la - base_la
    col = "green" if d > 0 else "red"
    cmp.add_row(f"LoRA (best ep {best.get('epoch',0)})",
                f"{best_raw:.4f}", f"[{col}]{best_la:.4f} ({d:+.4f})[/]")
    console.print(cmp)

    verdict = ("LoRA ПОМОГЛА" if d > 0.003
               else "LoRA не помогла (в пределах шума или хуже)")
    console.print(f"\n[bold]Вывод:[/] {verdict} "
                  f"(лучшая +LA {best_la:.4f} vs стоковая +LA {base_la:.4f}, delta {d:+.4f})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "baseline_argmax": round(base_raw, 4),
        "baseline_logit_adj": round(base_la, 4),
        "lora_best_argmax": round(best_raw, 4),
        "lora_best_logit_adj": round(best_la, 4),
        "delta_vs_baseline_la": round(d, 4),
        "best_epoch": best.get("epoch", 0),
        "n_train": len(Xtr), "n_test": len(yte),
        "train_dist": {CLASSES[i]: dist.get(i, 0) for i in range(5)},
        "lora_per_class_la": best.get("per_la", base_la_per),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    console.log(f"[bold green]Saved[/] → {OUT}")


if __name__ == "__main__":
    main()
