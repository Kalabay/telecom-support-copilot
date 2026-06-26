"""Обучает текстовую SER-голову на транскриптах Dusha train (для late fusion)."""

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
import torch.nn.functional as F  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

CLASSES = ["neutral", "angry", "positive", "sad", "other"]
C2I = {c: i for i, c in enumerate(CLASSES)}
TRAIN_JSONL = PROJECT_ROOT / "data" / "dusha_text" / "train.jsonl"
TEST_JSONL = PROJECT_ROOT / "data" / "dusha_text" / "test.jsonl"
HEAD_OUT = PROJECT_ROOT / "data" / "dusha_text" / "text_head.pt"
TEXT_MODEL = "cointegrated/rubert-tiny2"


class TextHead(nn.Module):
    def __init__(self, d_in: int = 312, d_hid: int = 128, n_cls: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hid), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(d_hid, n_cls),
        )

    def forward(self, x):
        return self.net(x)


def load_jsonl(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def encode_texts(tok, base, texts: list[str], device: str, bs: int = 32) -> np.ndarray:
    """Mean-pooled последний hidden state (стандартный sentence embedding для tiny-моделей)."""
    base.eval()
    feats = []
    with torch.inference_mode():
        for i in range(0, len(texts), bs):
            chunk = texts[i:i + bs]
            enc = tok(chunk, padding=True, truncation=True, max_length=128,
                      return_tensors="pt").to(device)
            out = base(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1).clamp_min(1)
            feats.append(pooled.float().cpu().numpy())
    return np.concatenate(feats)


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--model", default=TEXT_MODEL, help="HF id текстового энкодера")
    ap.add_argument("--out", type=Path, default=HEAD_OUT, help="куда сохранить голову")
    args = ap.parse_args()
    text_model = args.model
    head_out = args.out

    console = Console()
    console.rule(f"[bold cyan]Текстовая SER-голова на rubert-tiny2[/]  ep={args.epochs}")

    if not TRAIN_JSONL.exists():
        console.print(f"[red]Нет {TRAIN_JSONL}[/] — сначала запусти "
                      f"transcribe_dusha.py --split train")
        sys.exit(1)

    train_data = [d for d in load_jsonl(TRAIN_JSONL) if d["emotion"] in C2I and d["text"].strip()]
    test_data = ([d for d in load_jsonl(TEST_JSONL) if d["emotion"] in C2I and d["text"].strip()]
                 if TEST_JSONL.exists() else [])
    console.log(f"train={len(train_data)}  test={len(test_data)}")
    dist = Counter(d["emotion"] for d in train_data)
    console.log("train dist: " + ", ".join(f"{c}={dist.get(c,0)}" for c in CLASSES))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    console.log(f"Гружу {text_model}…")
    tok = AutoTokenizer.from_pretrained(text_model)
    base = AutoModel.from_pretrained(text_model).to(dev)
    for p in base.parameters():
        p.requires_grad = False
    d_in = int(base.config.hidden_size)
    console.log(f"hidden_size = {d_in}")

    t0 = time.perf_counter()
    Xtr = encode_texts(tok, base, [d["text"] for d in train_data], dev)
    ytr = np.array([C2I[d["emotion"]] for d in train_data], dtype=np.int64)
    if test_data:
        Xte = encode_texts(tok, base, [d["text"] for d in test_data], dev)
        yte = np.array([C2I[d["emotion"]] for d in test_data], dtype=np.int64)
    else:
        Xte, yte = None, None
    console.log(f"Эмбеддинги извлечены за {time.perf_counter()-t0:.1f}s")

    counts = np.array([dist.get(c, 1) for c in CLASSES], dtype=np.float32)
    cls_w = torch.tensor((counts.sum() / (5 * counts)), dtype=torch.float32, device=dev)

    head = TextHead(d_in=d_in).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    Xtr_t = torch.tensor(Xtr, device=dev)
    ytr_t = torch.tensor(ytr, device=dev)
    if Xte is not None:
        Xte_t = torch.tensor(Xte, device=dev)

    n = len(Xtr_t)
    for ep in range(1, args.epochs + 1):
        head.train()
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, args.bs):
            idx = perm[i:i + args.bs]
            opt.zero_grad()
            logits = head(Xtr_t[idx])
            loss = F.cross_entropy(logits, ytr_t[idx], weight=cls_w, label_smoothing=0.1)
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        if ep % 5 == 0 and Xte is not None:
            head.eval()
            with torch.inference_mode():
                pred = head(Xte_t).argmax(-1).cpu().numpy()
            f1, _ = macro_f1(yte, pred)
            console.log(f"  ep {ep}/{args.epochs}  loss={tot/n:.3f}  test_text_only_f1={f1:.3f}")

    head_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(),
                "text_model": text_model,
                "d_in": d_in,
                "classes": CLASSES}, head_out)
    console.log(f"[bold green]Saved[/] -> {head_out}")

    if Xte is not None:
        head.eval()
        with torch.inference_mode():
            pred = head(Xte_t).argmax(-1).cpu().numpy()
        f1, per = macro_f1(yte, pred)
        t = Table(title="Text-only SER на Dusha test (только текст, без аудио)")
        t.add_column("class", style="cyan")
        for col in ("precision", "recall", "f1"):
            t.add_column(col, justify="right")
        for c in CLASSES:
            m = per[c]
            t.add_row(c, f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['f1']:.3f}")
        console.print(t)
        console.print(f"[bold]text-only macro-F1 = {f1:.4f}[/]  (для сравнения: audio "
                      f"argmax 0.776, audio+LA 0.799)")


if __name__ == "__main__":
    main()
