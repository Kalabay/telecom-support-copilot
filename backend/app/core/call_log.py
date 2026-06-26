"""Логирование диалогов в JSONL — источник данных для дашборда QA."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from threading import Lock

from loguru import logger

LOG_DIR = Path(__file__).resolve().parents[3] / "data" / "call_logs"
LOG_FILE = LOG_DIR / "calls.jsonl"
_lock = Lock()


def log_event(event: dict) -> None:
    """Дописать событие (с меткой времени) в JSONL. Ошибки не роняют пайплайн."""
    try:
        record = {"ts": datetime.utcnow().isoformat(timespec="seconds"), **event}
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with _lock, LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"call log write failed: {exc}")


def read_events() -> list[dict]:
    """Прочитать все события из JSONL (битые строки пропускаются)."""
    if not LOG_FILE.exists():
        return []
    out: list[dict] = []
    with LOG_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out
