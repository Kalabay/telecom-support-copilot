"""Обучает и СОХРАНЯЕТ fusion-голову (GMU) как production-артефакт."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import app  # noqa: F401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

R = PROJECT_ROOT / "eval" / "results"
ART = PROJECT_ROOT / "backend" / "app" / "pipeline" / "artifacts"
CLASSES = ["neutral", "angry", "positive", "sad"]
TRAIN_NPZ = {"ga": "emb_gigaam_train.npz", "hu": "emb_hubert_train.npz",
             "text": "emb_text_train.npz"}
TEST_NPZ = {"ga": "emb_gigaam_test.npz", "hu": "emb_hubert_test.npz",
            "text": "emb_text_test.npz"}


def load(name):
    d = np.load(R / name, allow_pickle=True)
    return {str(i): (e, int(l)) for i, e, l in zip(d["ids"], d["emb"], d["labels"])}


def assemble(npz_map, mods):
    chans = {m: load(npz_map[m]) for m in mods}
    ids = sorted(set.intersection(*[set(c) for c in chans.values()]))
    y = np.array([chans[mods[0]][i][1] for i in ids])
    mats = {m: np.stack([chans[m][i][0] for i in ids]).astype("float32") for m in mods}
    return ids, mats, y


def macro_f1(yt, yp):
    f = []
    for c in range(4):
        tp = int(((yt == c) & (yp == c)).sum()); fp = int(((yt != c) & (yp == c)).sum())
        fn = int(((yt == c) & (yp != c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        f.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f))


class GMUFusion(nn.Module):
    """Gated multimodal unit поверх frozen-эмбеддингов экспертов."""

    def __init__(self, dims, d=256, n_cls=4):
        super().__init__()
        self.proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(x), nn.Linear(x, d)) for x in dims])
        self.gate = nn.Linear(sum(dims), d * len(dims))
        self.nm = len(dims); self.d = d
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                   nn.Dropout(0.3), nn.Linear(d, n_cls))

    def forward(self, xs, mod_drop=0.0, ret_gate=False):
        h = [torch.tanh(self.proj[i](xs[i])) for i in range(self.nm)]
        if self.training and mod_drop > 0 and self.nm > 1:
            for i in range(self.nm):
                m = (torch.rand(xs[i].size(0), 1, device=xs[i].device) > mod_drop).float()
                h[i] = h[i] * m
        H = torch.stack(h, 1)
        g = torch.softmax(self.gate(torch.cat(xs, -1)).view(-1, self.nm, self.d), 1)
        fused = (g * H).sum(1)
        out = self.head(fused)
        return (out, g.mean(dim=(0, 2))) if ret_gate else out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mods", nargs="+", default=["ga", "hu", "text"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--mod-drop", type=float, default=0.2)
    ap.add_argument("--out", type=Path, default=ART / "fusion_head.pt")
    args = ap.parse_args()

    mods = args.mods
    tr_ids, Xtr, ytr = assemble(TRAIN_NPZ, mods)
    te_ids, Xte, yte = assemble(TEST_NPZ, mods)
    dims = [Xtr[m].shape[1] for m in mods]
    print(f"модальности={mods} dims={dims} train={len(tr_ids)} test={len(te_ids)}")

    stats = {}
    for m in mods:
        mu = Xtr[m].mean(0, keepdims=True); sd = Xtr[m].std(0, keepdims=True) + 1e-6
        stats[m] = (mu, sd)
        Xtr[m] = (Xtr[m] - mu) / sd
        Xte[m] = (Xte[m] - mu) / sd

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model = GMUFusion(dims).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lf = nn.CrossEntropyLoss()
    Xtr_t = [torch.tensor(Xtr[m], device=dev) for m in mods]
    yt = torch.tensor(ytr, device=dev, dtype=torch.long)
    Xte_t = [torch.tensor(Xte[m], device=dev) for m in mods]
    n = len(ytr)
    best_f1, best_state = -1.0, None
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(n)
        for b in range(0, n, 256):
            idx = perm[b:b + 256]
            opt.zero_grad()
            loss = lf(model([x[idx] for x in Xtr_t], mod_drop=args.mod_drop), yt[idx])
            loss.backward(); opt.step()
        sch.step()
        model.eval()
        with torch.inference_mode():
            pred = model(Xte_t).argmax(-1).cpu().numpy()
        f1 = macro_f1(yte, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        _, gate = model(Xte_t, ret_gate=True)
    print(f"BEST test macro-F1={best_f1:.4f}  gate(масса по модальностям)="
          f"{dict(zip(mods, np.round(gate.cpu().numpy(), 3)))}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "mods": mods,
        "dims": dims,
        "classes": CLASSES,
        "zscore": {m: (stats[m][0].astype("float32"), stats[m][1].astype("float32")) for m in mods},
        "test_macro_f1": best_f1,
    }, args.out)
    print(f"saved fusion artifact -> {args.out}")


if __name__ == "__main__":
    main()
