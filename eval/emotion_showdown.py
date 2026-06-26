"""ЕДИНОЕ сравнение всех способов работы с эмоцией — одна метрика, один тест, CI."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import app  # noqa: F401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

R = PROJECT_ROOT / "eval" / "results"
CLASSES = ["neutral", "angry", "positive", "sad"]


def macro_f1(yt, yp):
    f = []
    for c in range(4):
        tp = int(((yt == c) & (yp == c)).sum()); fp = int(((yt != c) & (yp == c)).sum())
        fn = int(((yt == c) & (yp != c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        f.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f))


def load_pred(name):
    obj = json.loads((R / name).read_text(encoding="utf-8"))["preds"]
    return {str(p["id"]): p for p in obj}


def load_emb(name):
    d = np.load(R / name, allow_pickle=True)
    return {str(i): (e, int(l)) for i, e, l in zip(d["ids"], d["emb"], d["labels"])}


preds = {k: load_pred(f"_pred_{k}_crowd.json") for k in ["ga", "hu", "kl", "ours", "tx"]}
emb_te = {k: load_emb(f"emb_{n}_test.npz") for k, n in
          [("ga", "gigaam"), ("hu", "hubert"), ("tx", "text")]}
emb_tr = {k: load_emb(f"emb_{n}_train.npz") for k, n in
          [("ga", "gigaam"), ("hu", "hubert"), ("tx", "text")]}

ids = sorted(set.intersection(*[set(p) for p in preds.values()],
                              *[set(e) for e in emb_te.values()]))
y = np.array([preds["ga"][i]["true"] for i in ids])
P = {k: np.array([preds[k][i]["probs"] for i in ids]) for k in preds}
Ete = {k: np.array([emb_te[k][i][0] for i in ids]) for k in emb_te}
print(f"общих id: {len(ids)}")

tr_ids = sorted(set.intersection(*[set(e) for e in emb_tr.values()]))
ytr = np.array([emb_tr["ga"][i][1] for i in tr_ids])
Etr = {k: np.array([emb_tr[k][i][0] for i in tr_ids]) for k in emb_tr}


def boot_ci(yt, yp, iters=2000, seed=0):
    rng = np.random.RandomState(seed); n = len(yt); s = []
    for _ in range(iters):
        b = rng.randint(0, n, n); s.append(macro_f1(yt[b], yp[b]))
    return np.percentile(s, 2.5), np.percentile(s, 97.5)


def zscore(tr, te):
    mu, sd = tr.mean(0, keepdims=True), tr.std(0, keepdims=True) + 1e-6
    return (tr - mu) / sd, (te - mu) / sd


class GMU(nn.Module):
    def __init__(self, dims, d=256, n=4):
        super().__init__()
        self.proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(x), nn.Linear(x, d)) for x in dims])
        self.gate = nn.Linear(sum(dims), d * len(dims)); self.nm = len(dims); self.d = d
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(0.3), nn.Linear(d, n))

    def forward(self, xs):
        h = torch.stack([torch.tanh(self.proj[i](xs[i])) for i in range(self.nm)], 1)
        g = torch.softmax(self.gate(torch.cat(xs, -1)).view(-1, self.nm, self.d), 1)
        return self.head((g * h).sum(1))


class CatMLP(nn.Module):
    def __init__(self, dims, d=256, n=4):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(sum(dims)), nn.Linear(sum(dims), d),
                                  nn.GELU(), nn.Dropout(0.3), nn.Linear(d, n))

    def forward(self, xs):
        return self.net(torch.cat(xs, -1))


def train_fusion(cls, keys, epochs=50, seed=0):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    Xtr, Xte = [], []
    for k in keys:
        a, b = zscore(Etr[k], Ete[k]); Xtr.append(a); Xte.append(b)
    dims = [x.shape[1] for x in Xtr]
    m = cls(dims).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lf = nn.CrossEntropyLoss()
    Xtr_t = [torch.tensor(x, device=dev) for x in Xtr]
    yt = torch.tensor(ytr, device=dev, dtype=torch.long)
    Xte_t = [torch.tensor(x, device=dev) for x in Xte]
    nN = len(ytr)
    for _ in range(epochs):
        m.train(); perm = torch.randperm(nN)
        for bi in range(0, nN, 256):
            idx = perm[bi:bi + 256]; opt.zero_grad()
            lf(m([x[idx] for x in Xtr_t]), yt[idx]).backward(); opt.step()
        sch.step()
    m.eval()
    with torch.inference_mode():
        return m(Xte_t).argmax(-1).cpu().numpy()


def stack(keys, prior=False):
    X = np.hstack([P[k] for k in keys])
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    pred = np.zeros(len(y), dtype=int)
    for tr, te in skf.split(X, y):
        c = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
        pred[te] = c.predict(X[te])
    return pred


methods = []
methods.append(("1. HuBERT (argmax)", P["hu"].argmax(1)))
methods.append(("2. GigaAM (argmax)", P["ga"].argmax(1)))
methods.append(("3. Наша FT-модель wav2vec2", P["ours"].argmax(1)))
methods.append(("4. Текст CEDR (паралингв. контраст)", P["tx"].argmax(1)))
prior = np.bincount(ytr, minlength=4) / len(ytr)
la = (np.log(np.clip(P["hu"], 1e-9, 1)) - 0.9 * np.log(prior)).argmax(1)
methods.append(("5. HuBERT + logit adjustment", la))
methods.append(("6. Ансамбль среднее (GA+HU)", (0.5 * P["ga"] + 0.5 * P["hu"]).argmax(1)))
methods.append(("7. Ансамбль stacking (GA+HU)", stack(["ga", "hu"])))
methods.append(("8. Ансамбль stacking (GA+HU+наша)", stack(["ga", "hu", "ours"])))
methods.append(("9. Fusion concat+MLP (эмб. GA+HU)", train_fusion(CatMLP, ["ga", "hu"])))
methods.append(("10. Fusion GMU-гейтинг (эмб. GA+HU)", train_fusion(GMU, ["ga", "hu"])))
methods.append(("11. Fusion GMU +текст (GA+HU+CEDR)", train_fusion(GMU, ["ga", "hu", "tx"])))

res = []
for name, pred in methods:
    f1 = macro_f1(y, pred)
    lo, hi = boot_ci(y, pred)
    res.append((name, f1, lo, hi))

res_sorted = sorted(res, key=lambda x: -x[1])
print(f"\n{'='*78}")
print(f"СРАВНЕНИЕ СПОСОБОВ РАБОТЫ С ЭМОЦИЕЙ (Dusha crowd, n={len(ids)}, macro-F1)")
print(f"{'='*78}")
print(f"{'способ':42s} {'F1':>7s}  95% CI")
print("-" * 78)
for name, f1, lo, hi in res_sorted:
    print(f"{name:42s} {f1:.4f}  [{lo:.3f}, {hi:.3f}]")
print("-" * 78)
best = res_sorted[0]
print(f"ЛУЧШИЙ: {best[0]}  ({best[1]:.4f})")

R.joinpath("emotion_showdown.json").write_text(json.dumps(
    {"n": len(ids), "results": [{"method": n, "macro_f1": round(f, 4),
     "ci_low": round(lo, 4), "ci_high": round(hi, 4)} for n, f, lo, hi in res_sorted]},
    ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nsaved -> {R / 'emotion_showdown.json'}")
