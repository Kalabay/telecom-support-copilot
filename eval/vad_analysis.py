"""Разделяет ли реальная V/A-модель эмоции лучше нашей эвристической таблицы?"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

R = Path(r"K:\dev\coursework\eval\results")
rows = json.loads((R / "vad_crowd.json").read_text(encoding="utf-8"))
CLASSES = ["neutral", "angry", "positive", "sad"]
C2I = {c: i for i, c in enumerate(CLASSES)}

HEUR = {"neutral": (0.30, 0.00), "angry": (0.80, -0.70),
        "positive": (0.60, 0.70), "sad": (0.30, -0.50)}

y = np.array([C2I[r["emotion"]] for r in rows])
A = np.array([r["arousal"] for r in rows])
V = np.array([r["valence"] for r in rows])
D = np.array([r["dominance"] for r in rows])

print("=== 1. Средние V/A от РЕАЛЬНОЙ модели по эмоциям ===")
print(f"{'эмоция':10s} {'arousal':>9s} {'valence':>9s} {'dominance':>10s}   (наша эвристика A/V)")
for c in CLASSES:
    mask = y == C2I[c]
    print(f"{c:10s} {A[mask].mean():9.3f} {V[mask].mean():9.3f} {D[mask].mean():10.3f}"
          f"   эвр: {HEUR[c][0]:.2f}/{HEUR[c][1]:+.2f}")

print("\n=== 2. Насколько хорошо РЕАЛЬНЫЕ V/A разделяют эмоции ===")
X = np.column_stack([A, D, V])


def macro_f1(yt, yp):
    f = []
    for c in range(4):
        tp = int(((yt == c) & (yp == c)).sum()); fp = int(((yt != c) & (yp == c)).sum())
        fn = int(((yt == c) & (yp != c)).sum())
        p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
        f.append(2 * p * r / (p + r) if p + r else 0)
    return float(np.mean(f))


pred = cross_val_predict(LogisticRegression(max_iter=2000, class_weight="balanced"),
                          X, y, cv=5)
f1_vad = macro_f1(y, pred)
print(f"  classifier на реальных (arousal,dominance,valence) -> эмоция: macro-F1 = {f1_vad:.4f}")

print("\n  (Наша эвристика ставит ФИКСИРОВАННЫЕ a/v по уже-предсказанному классу —")
print("   она не несёт независимого сигнала. Реальная V/A — несёт, вопрос сколько.)")

print("\n=== 3. Корреляция реальных V/A с правильными эмоциями ===")
for axis, vals in [("arousal", A), ("valence", V), ("dominance", D)]:
    ang = vals[y == C2I["angry"]].mean()
    pos = vals[y == C2I["positive"]].mean()
    sad = vals[y == C2I["sad"]].mean()
    neu = vals[y == C2I["neutral"]].mean()
    print(f"  {axis:9s}: neutral={neu:.2f} angry={ang:.2f} positive={pos:.2f} sad={sad:.2f}")

print("\nВЫВОД:")
print(f"  Реальная V/A разделяет 4 эмоции на macro-F1={f1_vad:.3f} (одни 3 числа).")
print(f"  Наши SER-модели дают ~0.85. Значит V/A — слабее как классификатор,")
print(f"  но даёт ИНТЕРПРЕТИРУЕМЫЕ непрерывные оси для escalation/тона.")
