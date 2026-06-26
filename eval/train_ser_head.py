"""Эксперимент с дообучением головы SER + честное сравнение со стоковой моделью."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import app  # noqa: F401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from datasets import Audio, load_dataset  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from app.pipeline.ser import get_recognizer  # noqa: E402

CLASSES = ["neutral", "angry", "positive", "sad", "other"]
C2I = {c: i for i, c in enumerate(CLASSES)}
OUT = PROJECT_ROOT / "eval" / "results" / "ser_head_finetune.json"


def macro_f1(y_true: list[int], y_pred: list[int]) -> tuple[float, dict]:
    per = {}
    f1s = []
    for ci, c in enumerate(CLASSES):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == ci and p == ci)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != ci and p == ci)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == ci and p != ci)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per[c] = {"precision": round(prec, 3), "recall": round(rec, 3),
                  "f1": round(f1, 3), "support": sum(1 for t in y_true if t == ci)}
        f1s.append(f1)
    return sum(f1s) / len(f1s), per


def extract(rec, raw: bytes):  # noqa: ANN001
    """Вернуть (feat_1024 np.float32, stock_pred_idx) или None."""
    model, extractor, device = rec._model, rec._extractor, rec._device  # noqa: SLF001
    wav, sr = rec.load_audio(raw)
    if wav.size < 16000 * 0.3:
        return None
    if wav.shape[0] > 16000 * 10:
        wav = wav[: 16000 * 10]
    inputs = extractor(wav, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model(**inputs, output_hidden_states=True)
        feat = out.hidden_states[-1].mean(dim=1).squeeze(0).float().cpu().numpy()
        stock_pred = int(out.logits.argmax(-1).item())
    stock_label = model.config.id2label[stock_pred].lower()
    return feat.astype(np.float32), C2I.get(stock_label, C2I["other"])


def collect(rec, split: str, n: int, console: Console):  # noqa: ANN001
    ds = load_dataset("xbgoose/dusha", split=split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    X, y, stock = [], [], []
    seen = 0
    for item in ds:
        if len(X) >= n:
            break
        true = str(item["emotion"]).lower().strip()
        if true not in C2I:
            continue
        raw = item["audio"].get("bytes")
        if not raw:
            continue
        r = extract(rec, raw)
        if r is None:
            continue
        feat, stock_pred = r
        X.append(feat)
        y.append(C2I[true])
        stock.append(stock_pred)
        seen += 1
        if seen % 200 == 0:
            console.log(f"  {split}: {seen}/{n}")
    return np.stack(X), np.array(y), np.array(stock)


class Head(nn.Module):
    def __init__(self, d_in=1024, d_hid=256, n_cls=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hid), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(d_hid, n_cls),
        )

    def forward(self, x):
        return self.net(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--test", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    console = Console()
    console.rule(f"[bold cyan]SER head fine-tune[/]  train={args.train} test={args.test}")
    rec = get_recognizer()
    rec._ensure_loaded()  # noqa: SLF001
    dev = rec._device  # noqa: SLF001

    t0 = time.perf_counter()
    console.log("Извлекаю фичи train…")
    Xtr, ytr, _ = collect(rec, "train", args.train, console)
    console.log("Извлекаю фичи test…")
    Xte, yte, stock_te = collect(rec, "test", args.test, console)
    console.log(f"Фичи готовы за {time.perf_counter()-t0:.0f}s. "
                f"train={len(Xtr)}, test={len(Xte)}")

    dist = Counter(int(v) for v in ytr)
    console.log("Train распределение: " + ", ".join(
        f"{CLASSES[i]}={dist.get(i,0)}" for i in range(5)))

    base_f1, base_per = macro_f1(list(yte), list(stock_te))

    counts = np.array([dist.get(i, 1) for i in range(5)], dtype=np.float32)
    weights = (counts.sum() / (5 * counts))
    wt = torch.tensor(weights, dtype=torch.float32, device=dev)

    head = Head().to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss(weight=wt, label_smoothing=0.1)

    Xtr_t = torch.tensor(Xtr, device=dev)
    ytr_t = torch.tensor(ytr, device=dev, dtype=torch.long)
    Xte_t = torch.tensor(Xte, device=dev)

    head.train()
    bs = 128
    n = len(Xtr_t)
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            logits = head(Xtr_t[idx])
            loss = lossf(logits, ytr_t[idx])
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        if (ep + 1) % 15 == 0:
            console.log(f"  epoch {ep+1}/{args.epochs}  loss={tot/n:.3f}")

    head.eval()
    with torch.inference_mode():
        new_pred = head(Xte_t).argmax(-1).cpu().tolist()
    new_f1, new_per = macro_f1(list(yte), new_pred)

    cmp = Table(title="Сравнение macro-F1 на одном и том же test")
    cmp.add_column("вариант", style="cyan")
    cmp.add_column("macro-F1", justify="right")
    cmp.add_column("delta", justify="right")
    cmp.add_row("стоковая голова (baseline)", f"{base_f1:.4f}", "—")
    delta = new_f1 - base_f1
    color = "green" if delta > 0 else "red"
    cmp.add_row("дообученная голова (probe)", f"{new_f1:.4f}",
                f"[{color}]{delta:+.4f}[/]")
    console.print(cmp)

    per_t = Table(title="Per-class F1: baseline -> probe")
    per_t.add_column("класс", style="cyan")
    per_t.add_column("baseline F1", justify="right")
    per_t.add_column("probe F1", justify="right")
    for c in CLASSES:
        b = base_per[c]["f1"]
        nw = new_per[c]["f1"]
        per_t.add_row(c, f"{b:.3f}", f"{nw:.3f}")
    console.print(per_t)

    verdict = ("дообучение ПОМОГЛО" if delta > 0.005
               else "дообучение НЕ помогло (в пределах шума или хуже)")
    console.print(f"\n[bold]Вывод:[/] {verdict} (delta {delta:+.4f})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "baseline_macro_f1": round(base_f1, 4),
        "finetuned_macro_f1": round(new_f1, 4),
        "delta": round(delta, 4),
        "n_train": len(Xtr), "n_test": len(Xte),
        "train_dist": {CLASSES[i]: dist.get(i, 0) for i in range(5)},
        "baseline_per_class": base_per,
        "finetuned_per_class": new_per,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    console.log(f"[bold green]Saved[/] → {OUT}")


if __name__ == "__main__":
    main()
