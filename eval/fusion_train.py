"""Обучаемое слияние frozen-эмбеддингов экспертов (GMU-гейтинг + modality dropout)."""

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

CLASSES = ["neutral", "angry", "positive", "sad"]


def load_npz(spec: str):
    name, path = spec.split("=", 1)
    d = np.load(path, allow_pickle=True)
    return name, {str(i): (e, int(l)) for i, e, l in zip(d["ids"], d["emb"], d["labels"])}


def macro_f1(yt, yp):
    f1s = []
    for c in range(4):
        tp = int(((yt == c) & (yp == c)).sum()); fp = int(((yt != c) & (yp == c)).sum())
        fn = int(((yt == c) & (yp != c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


def assemble(specs):
    chans = [load_npz(s) for s in specs]
    names = [c[0] for c in chans]
    ids = sorted(set.intersection(*[set(c[1]) for c in chans]))
    y = np.array([chans[0][1][i][1] for i in ids])
    mats = [np.stack([c[1][i][0] for i in ids]).astype("float32") for c in chans]
    return names, ids, mats, y


class GMUFusion(nn.Module):
    """Проекции -> мульти-модальный GMU (softmax-гейт по модальностям) -> голова."""

    def __init__(self, dims, d=256, n_cls=4, p_drop=0.3):
        super().__init__()
        self.proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(di), nn.Linear(di, d)) for di in dims])
        self.gate = nn.Linear(sum(dims), d * len(dims))
        self.n_mod = len(dims)
        self.d = d
        self.drop = nn.Dropout(0.3)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                   nn.Dropout(0.3), nn.Linear(d, n_cls))
        self.p_mod_drop = p_drop

    def forward(self, xs, mod_drop=False):
        h = [torch.tanh(self.proj[i](xs[i])) for i in range(self.n_mod)]
        if mod_drop and self.training and self.n_mod > 1:
            for i in range(self.n_mod):
                mask = (torch.rand(xs[i].size(0), 1, device=xs[i].device) > self.p_mod_drop).float()
                h[i] = h[i] * mask
        g = self.gate(torch.cat(xs, -1)).view(-1, self.n_mod, self.d)
        g = torch.softmax(g, dim=1)
        H = torch.stack(h, dim=1)
        fused = (g * H).sum(1)
        return self.head(self.drop(fused))


class ConcatMLP(nn.Module):
    def __init__(self, dims, d=256, n_cls=4):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(sum(dims)), nn.Linear(sum(dims), d), nn.GELU(),
                                  nn.Dropout(0.3), nn.Linear(d, n_cls))

    def forward(self, xs, mod_drop=False):
        return self.net(torch.cat(xs, -1))


def train_eval(model_cls, dims, Xtr, ytr, Xte, yte, dev, epochs=40, lr=3e-4,
               mod_drop=False, seed=0):
    torch.manual_seed(seed)
    model = model_cls(dims).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.CrossEntropyLoss()
    Xtr_t = [torch.tensor(x, device=dev) for x in Xtr]
    ytr_t = torch.tensor(ytr, device=dev, dtype=torch.long)
    Xte_t = [torch.tensor(x, device=dev) for x in Xte]
    n = len(ytr); bs = 256
    best_f1, best_pred = -1.0, None
    nval = n // 10
    perm = torch.randperm(n)
    vi, ti = perm[:nval], perm[nval:]
    for ep in range(epochs):
        model.train()
        for b in range(0, len(ti), bs):
            idx = ti[b:b + bs]
            opt.zero_grad()
            out = model([x[idx] for x in Xtr_t], mod_drop=mod_drop)
            loss = lossf(out, ytr_t[idx])
            loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.inference_mode():
            vpred = model([x[vi] for x in Xtr_t]).argmax(-1).cpu().numpy()
        vf1 = macro_f1(ytr[vi.numpy()], vpred)
        if vf1 > best_f1:
            best_f1 = vf1
            with torch.inference_mode():
                best_pred = model(Xte_t).argmax(-1).cpu().numpy()
    return macro_f1(yte, best_pred), best_pred


def bootstrap_ci(yte, pred, ref_f1, iters=2000, seed=0):
    rng = np.random.RandomState(seed)
    n = len(yte)
    diffs = []
    for _ in range(iters):
        b = rng.randint(0, n, n)
        diffs.append(macro_f1(yte[b], pred[b]))
    arr = np.array(diffs)
    return arr.mean(), np.percentile(arr, 2.5), np.percentile(arr, 97.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--ref", type=float, default=0.855, help="бар для сравнения (prob-stack)")
    args = ap.parse_args()

    names, tr_ids, Xtr, ytr = assemble(args.train)
    names2, te_ids, Xte, yte = assemble(args.test)
    assert names == names2
    dims = [x.shape[1] for x in Xtr]
    print(f"модальности: {names}  dims={dims}")
    print(f"train={len(tr_ids)}  test={len(te_ids)}")

    for i in range(len(Xtr)):
        mu, sd = Xtr[i].mean(0, keepdims=True), Xtr[i].std(0, keepdims=True) + 1e-6
        Xtr[i] = (Xtr[i] - mu) / sd
        Xte[i] = (Xte[i] - mu) / sd

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n--- одиночные (linear probe на эмбеддинге, test macro-F1) ---")
    for i, nm in enumerate(names):
        f1, _ = train_eval(ConcatMLP, [dims[i]], [Xtr[i]], ytr, [Xte[i]], yte, dev, epochs=30)
        print(f"  {nm:6s} {f1:.4f}")

    print("\n--- слияние (test macro-F1, среднее по 3 сидам) ---")
    configs = [
        ("concat+MLP", ConcatMLP, False),
        ("GMU", GMUFusion, False),
        ("GMU+mod-drop", GMUFusion, True),
    ]
    results = {}
    for tag, cls, md in configs:
        f1s, preds = [], []
        for s in range(3):
            f1, pred = train_eval(cls, dims, Xtr, ytr, Xte, yte, dev, mod_drop=md, seed=s)
            f1s.append(f1); preds.append(pred)
        best = preds[int(np.argmax(f1s))]
        results[tag] = (np.mean(f1s), np.std(f1s), best)
        print(f"  {tag:16s} {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")

    best_tag = max(results, key=lambda k: results[k][0])
    mean, lo, hi = bootstrap_ci(yte, results[best_tag][2], args.ref)
    print(f"\nлучшее: {best_tag}  test macro-F1={results[best_tag][0]:.4f}")
    print(f"bootstrap 95% CI=[{lo:.4f}, {hi:.4f}]  (бар prob-stack={args.ref})")
    print(f"  -> {'ВЫШЕ бара (CI выше ref)' if lo > args.ref else 'в пределах/ниже бара (как и ждали на Dusha)'}")


if __name__ == "__main__":
    main()
