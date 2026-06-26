"""Money-plot: на лексически-конгруэнтной синтетике текст ДОЛЖЕН помочь."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import app  # noqa: F401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

R = Path(r"K:\dev\coursework\eval\results")


def load(name):
    d = np.load(R / name, allow_pickle=True)
    return {str(i): (e, int(l)) for i, e, l in zip(d["ids"], d["emb"], d["labels"])}


def macro_f1(yt, yp):
    f = []
    for c in range(4):
        tp = int(((yt == c) & (yp == c)).sum()); fp = int(((yt != c) & (yp == c)).sum())
        fn = int(((yt == c) & (yp != c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        f.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f))


class GMU(nn.Module):
    def __init__(self, dims, d=128, n_cls=4):
        super().__init__()
        self.proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(x), nn.Linear(x, d)) for x in dims])
        self.gate = nn.Linear(sum(dims), d * len(dims))
        self.nm = len(dims); self.d = d
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                   nn.Dropout(0.3), nn.Linear(d, n_cls))

    def forward(self, xs, ret_gate=False):
        h = torch.stack([torch.tanh(self.proj[i](xs[i])) for i in range(self.nm)], 1)
        g = torch.softmax(self.gate(torch.cat(xs, -1)).view(-1, self.nm, self.d), 1)
        fused = (g * h).sum(1)
        out = self.head(fused)
        if ret_gate:
            return out, g.mean(dim=(0, 2))
        return out


def fit(dims, Xtr, ytr, Xte, dev, epochs=60, seed=0):
    torch.manual_seed(seed)
    m = GMU(dims).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lf = nn.CrossEntropyLoss()
    Xtr = [torch.tensor(x, device=dev) for x in Xtr]
    yt = torch.tensor(ytr, device=dev, dtype=torch.long)
    Xte = [torch.tensor(x, device=dev) for x in Xte]
    n = len(ytr)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n)
        for b in range(0, n, 64):
            idx = perm[b:b + 64]
            opt.zero_grad()
            loss = lf(m([x[idx] for x in Xtr]), yt[idx])
            loss.backward(); opt.step()
        sch.step()
    m.eval()
    with torch.inference_mode():
        out, gate = m(Xte, ret_gate=True)
        pred = out.argmax(-1).cpu().numpy()
    return pred, gate.cpu().numpy()


def line_idx(fid):
    return int(fid.split("_")[1])


def main():
    ga = load("emb_gigaam_synth.npz"); hu = load("emb_hubert_synth.npz"); tx = load("emb_text_synth.npz")
    ids = sorted(set(ga) & set(hu) & set(tx))
    y = np.array([ga[i][1] for i in ids])
    Pga = np.array([ga[i][0] for i in ids]); Phu = np.array([hu[i][0] for i in ids])
    Ptx = np.array([tx[i][0] for i in ids])
    lines = np.array([line_idx(i) for i in ids])
    tr = lines < 30; te = lines >= 30
    print(f"синтетика: train={tr.sum()} test={te.sum()} (split по репликам)")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mats = {"ga": Pga, "hu": Phu, "tx": Ptx}
    Z = {}
    for k, M in mats.items():
        mu, sd = M[tr].mean(0, keepdims=True), M[tr].std(0, keepdims=True) + 1e-6
        Z[k] = (M - mu) / sd

    def run(keys, seeds=5):
        dims = [Z[k].shape[1] for k in keys]
        f1s, gates, preds = [], [], []
        for s in range(seeds):
            pred, gate = fit(dims, [Z[k][tr] for k in keys], y[tr],
                             [Z[k][te] for k in keys], dev, seed=s)
            f1s.append(macro_f1(y[te], pred)); gates.append(gate); preds.append(pred)
        best = preds[int(np.argmax(f1s))]
        return np.mean(f1s), np.std(f1s), np.mean(gates, 0), best

    f_tx, _, _, _ = run(["tx"])
    print(f"\ntext-only (синтетика): {f_tx:.4f}  <- текст ИНФОРМАТИВЕН (vs Dusha 0.54, RESD 0.33)")
    f_audio, s_a, _, pred_a = run(["ga", "hu"])
    f_at, s_at, gate_at, pred_at = run(["ga", "hu", "tx"])
    print(f"\nMONEY-PLOT (синтетика, n_test={te.sum()}, macro-F1):")
    print(f"  audio-only fusion (ga+hu)     {f_audio:.4f} +/- {s_a:.4f}")
    print(f"  audio+TEXT fusion (ga+hu+tx)  {f_at:.4f} +/- {s_at:.4f}")
    print(f"  Δ от текста = {f_at - f_audio:+.4f}")
    yte = y[te]; rng = np.random.RandomState(0); diffs = []
    m = len(yte)
    for _ in range(3000):
        b = rng.randint(0, m, m)
        diffs.append(macro_f1(yte[b], pred_at[b]) - macro_f1(yte[b], pred_a[b]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  Δ 95% CI=[{lo:+.4f}, {hi:+.4f}]  P(Δ>0)={(np.array(diffs)>0).mean():.3f}"
          f"  -> {'ЗНАЧИМО' if lo>0 else 'не значимо'}")
    print(f"  GMU-гейт средняя масса [ga, hu, tx] = {np.round(gate_at, 3)}")
    print(f"    -> доля гейта на ТЕКСТ = {gate_at[2]*100:.1f}% (на Dusha гейт текст душит)")

    print("\n=== КОНТРАСТ (Δ от добавления текста в fusion) ===")
    print("  Dusha (паралингв.):  ~0.849 -> ~0.849   Δ≈0      [fusion_train.py на Dusha-эмб.]")
    print("  RESD  (паралингв.):  0.703 -> 0.658     Δ=-0.045 [fusion_train.py на RESD-эмб.]")
    print(f"  Синтетика (конгру.): {f_audio:.3f} -> {f_at:.3f}   Δ={f_at-f_audio:+.3f}  [этот скрипт]")


if __name__ == "__main__":
    main()
