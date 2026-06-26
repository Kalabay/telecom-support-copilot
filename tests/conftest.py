"""Общие фикстуры/настройки для тестов: прокидываем backend и sdk в sys.path."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_BACKEND = _ROOT / "backend"
_SDK = _ROOT / "sdk"

for _p in (_BACKEND, _SDK):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)
