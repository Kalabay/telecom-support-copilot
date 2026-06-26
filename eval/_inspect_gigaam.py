"""Разобрать архитектуру GigaAM-emo, найти где взять эмбеддинг (до классификатора)."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import app  # noqa: F401

import torch
from omegaconf.dictconfig import DictConfig

_orig = torch.load
torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
import gigaam  # noqa: E402

m = gigaam.load_model("emo")
print("TYPE:", type(m).__name__)
print("\n=== top-level attrs / submodules ===")
for name, mod in m.named_children():
    print(f"  {name}: {type(mod).__name__}")

print("\n=== полная структура (имена модулей) ===")
for name, mod in m.named_modules():
    if name.count(".") <= 1:
        print(f"  {name or '<root>'}: {type(mod).__name__}")

print("\n=== id2name ===", getattr(m, "id2name", None))

print("\n=== Linear-слои с out_features<=8 (кандидаты в классификатор) ===")
import torch.nn as nn
for name, mod in m.named_modules():
    if isinstance(mod, nn.Linear) and mod.out_features <= 8:
        print(f"  {name}: in={mod.in_features} out={mod.out_features}")
