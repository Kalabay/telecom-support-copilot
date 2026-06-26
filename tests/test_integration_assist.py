"""Интеграционный тест: реальный POST /api/assist к работающему бэкенду."""

from __future__ import annotations

import pytest
import requests

from telecom_copilot import AssistResult, CopilotClient

BASE_URL = "http://localhost:8000"


def _backend_up() -> bool:
    try:
        r = requests.get(f"{BASE_URL}/api/health", timeout=2.0)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.integration
def test_real_assist() -> None:
    if not _backend_up():
        pytest.skip("backend на localhost:8000 недоступен")

    cop = CopilotClient(BASE_URL)
    res = cop.assist("Третий день нет интернета, ничего не работает!", company="orbita")

    assert isinstance(res, AssistResult)
    assert len(res.suggestions) >= 1
    assert res.suggestions[0].text.strip()
    assert res.total_ms >= 0
