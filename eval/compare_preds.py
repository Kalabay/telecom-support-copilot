"""Сравнить macro-F1 двух predictions-файлов на пересечении idx."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, f1_score


def load(p: Path) -> dict[str, dict]:
    obj = json.loads(p.read_text(encoding="utf-8"))
    key = "id" if "id" in obj["preds"][0] else "idx"
    return {str(rec[key]): rec for rec in obj["preds"]}, obj["classes"]


def main() -> None:
    ga_path = Path(sys.argv[1])
    hu_path = Path(sys.argv[2])
    ga, classes_ga = load(ga_path)
    hu, classes_hu = load(hu_path)
    assert classes_ga == classes_hu, "разные классы"
    classes = classes_ga

    common = sorted(set(ga) & set(hu))
    print(f"общих idx: {len(common)}  (gigaam={len(ga)}, hubert={len(hu)})")

    y_true = np.array([ga[i]["true"] for i in common])
    y_ga = np.array([ga[i]["pred"] for i in common])
    y_hu_arg = np.array([hu[i]["pred_argmax"] for i in common])
    y_hu_la = np.array([hu[i]["pred_la"] for i in common])

    print("\nраспределение true:", dict(Counter([classes[t] for t in y_true])))

    def report(name: str, y_pred: np.ndarray) -> None:
        f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
        print(f"\n=== {name}  macro-F1 = {f1m:.4f} ===")
        print(classification_report(y_true, y_pred, target_names=classes,
                                      zero_division=0, digits=4))

    report("GigaAM-Emo (Sber)", y_ga)
    report("HuBERT-Dusha (argmax)", y_hu_arg)
    report("HuBERT-Dusha + logit adjustment (tau=0.9)", y_hu_la)


if __name__ == "__main__":
    main()
