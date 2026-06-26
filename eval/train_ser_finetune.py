"""Partial fine-tuning русского wav2vec2 на Dusha crowd (наш собственный SER)."""

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
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    get_linear_schedule_with_warmup,
)

CLASSES = ["neutral", "angry", "positive", "sad", "other"]
C2I = {c: i for i, c in enumerate(CLASSES)}
MAX_SEC = 8.0
SR = 16000


class DushaDS(Dataset):
    def __init__(self, rows, fe):
        self.rows, self.fe = rows, fe

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        data, sr = sf.read(r["path"], dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != SR:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=SR)
        data = data[: int(MAX_SEC * SR)]
        if len(data) < SR // 2:
            data = np.pad(data, (0, SR // 2 - len(data)))
        return data.astype("float32"), C2I[r["emotion"]]


def make_collate(fe):
    def collate(batch):
        wavs = [b[0] for b in batch]
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        enc = fe(wavs, sampling_rate=SR, padding=True, return_tensors="pt")
        return enc["input_values"], enc.get("attention_mask"), labels
    return collate


def macro_f1_np(y_true, y_pred, n_cls=len(CLASSES)) -> float:
    f1s = []
    for c in range(n_cls):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s))


@torch.inference_mode()
def evaluate(model, loader, dev) -> tuple[float, np.ndarray]:
    model.eval()
    preds, trues = [], []
    for iv, am, lb in loader:
        iv = iv.to(dev)
        kw = {"attention_mask": am.to(dev)} if am is not None else {}
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(iv, **kw).logits
        preds.append(logits.float().argmax(-1).cpu().numpy())
        trues.append(lb.numpy())
    yp, yt = np.concatenate(preds), np.concatenate(trues)
    return macro_f1_np(yt, yp), yp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path(r"K:\.caches\dusha_train"))
    ap.add_argument("--base", default="jonatasgrosman/wav2vec2-large-xlsr-53-russian")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--out", type=Path, default=Path(r"K:\.caches\ser_finetuned"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in (args.data / "manifest.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    rng = np.random.RandomState(42)
    rng.shuffle(rows)
    nval = int(len(rows) * args.val_frac)
    val_rows, train_rows = rows[:nval], rows[nval:]
    print(f"train={len(train_rows)} val={len(val_rows)}", flush=True)
    print("train dist:", dict(Counter(r["emotion"] for r in train_rows)), flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fe = AutoFeatureExtractor.from_pretrained(args.base)
    model = AutoModelForAudioClassification.from_pretrained(
        args.base, num_labels=len(CLASSES),
        label2id=C2I, id2label={i: c for c, i in C2I.items()},
        ignore_mismatched_sizes=True).to(dev)
    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
        print("feature_encoder заморожен (partial FT)", flush=True)

    collate = make_collate(fe)
    tl = DataLoader(DushaDS(train_rows, fe), batch_size=args.bs, shuffle=True,
                    collate_fn=collate, num_workers=0, pin_memory=True, drop_last=True)
    vl = DataLoader(DushaDS(val_rows, fe), batch_size=args.bs, shuffle=False,
                    collate_fn=collate, num_workers=0, pin_memory=True)

    cnt = Counter(r["emotion"] for r in train_rows)
    w = np.array([len(train_rows) / (len(CLASSES) * max(cnt.get(c, 1), 1)) for c in CLASSES])
    w = np.clip(w, 0.5, 3.0)
    cls_w = torch.tensor(w, dtype=torch.float32, device=dev)
    print("class weights:", {c: round(float(x), 2) for c, x in zip(CLASSES, w)}, flush=True)

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = len(tl) // args.accum
    total_steps = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda")
    lossf = torch.nn.CrossEntropyLoss(weight=cls_w, label_smoothing=0.1)

    best_f1, best_state = -1.0, None
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, running = time.perf_counter(), 0.0
        opt.zero_grad()
        for it, (iv, am, lb) in enumerate(tl, 1):
            iv, lb = iv.to(dev), lb.to(dev)
            kw = {"attention_mask": am.to(dev)} if am is not None else {}
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(iv, **kw).logits
                loss = lossf(logits, lb) / args.accum
            scaler.scale(loss).backward()
            running += float(loss) * args.accum
            if it % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sched.step()
            if it % 200 == 0:
                el = time.perf_counter() - t0
                print(f"  ep{ep} it{it}/{len(tl)} loss={running/it:.3f} "
                      f"{it/el:.1f}it/s", flush=True)
        f1, _ = evaluate(model, vl, dev)
        print(f"[ep {ep}] train_loss={running/len(tl):.3f}  val_macro_f1={f1:.4f}  "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ^ новый лучший ({f1:.4f}), сохраняю", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    out = args.out / "best"
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    fe.save_pretrained(str(out))
    print(f"BEST val_macro_f1={best_f1:.4f}  saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
