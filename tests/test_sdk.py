"""Тесты SDK: monkeypatch HTTP, проверяем payload и парсинг ответа."""

from __future__ import annotations

import requests

from telecom_copilot import AssistResult, CopilotClient

ASSIST_JSON = {
    "suggestions": [
        {
            "text": "Понимаю ваше раздражение, давайте быстро всё восстановим.",
            "rank": 1,
            "sources": [
                {"doc_id": "no_internet", "title": "Нет интернета", "snippet": "...", "score": 0.8},
            ],
        },
        {"text": "Перезагрузите роутер, пожалуйста.", "rank": 2, "sources": []},
    ],
    "sources": [
        {"doc_id": "no_internet", "title": "Нет интернета", "snippet": "...", "score": 0.8},
    ],
    "emotion": {
        "label": "angry",
        "confidence": 0.7,
        "arousal": 0.8,
        "valence": -0.6,
        "escalation_risk": True,
    },
    "retrieval_ms": 12,
    "llm_ms": 340,
    "total_ms": 360,
}

KB_DOC_JSON = {
    "doc_id": "no_internet",
    "title": "Нет интернета",
    "company": "orbita",
    "text": "Полный текст статьи базы знаний.",
}


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:  # noqa: D401
        return None

    def json(self) -> dict:
        return self._payload


def test_assist_payload_and_parsing(monkeypatch) -> None:
    captured: dict = {}

    def fake_post(url, json=None, timeout=None, **kwargs):  # noqa: A002
        captured["url"] = url
        captured["json"] = json
        return _FakeResponse(ASSIST_JSON)

    monkeypatch.setattr(requests, "post", fake_post)

    cop = CopilotClient("http://localhost:8000")
    emotion = {"label": "angry", "confidence": 0.7, "arousal": 0.8, "valence": -0.6}
    history = [{"speaker": "customer", "text": "Здравствуйте"}]
    res = cop.assist(
        "Третий день нет интернета!",
        company="orbita",
        history=history,
        emotion=emotion,
    )

    payload = captured["json"]
    assert captured["url"].endswith("/api/assist")
    assert payload["text"] == "Третий день нет интернета!"
    assert payload["company"] == "orbita"
    assert payload["history"] == history
    assert payload["emotion"] == emotion

    assert isinstance(res, AssistResult)
    assert res.suggestions[0].text.startswith("Понимаю ваше раздражение")
    assert res.suggestions[0].rank == 1
    assert res.suggestions[0].sources[0].doc_id == "no_internet"
    assert res.sources[0].doc_id == "no_internet"
    assert res.emotion["label"] == "angry"
    assert res.retrieval_ms == 12
    assert res.llm_ms == 340
    assert res.total_ms == 360


def test_assist_omits_optional_fields(monkeypatch) -> None:
    captured: dict = {}

    def fake_post(url, json=None, timeout=None, **kwargs):  # noqa: A002
        captured["json"] = json
        return _FakeResponse(ASSIST_JSON)

    monkeypatch.setattr(requests, "post", fake_post)

    CopilotClient().assist("Нет связи")
    payload = captured["json"]
    assert payload["text"] == "Нет связи"
    assert "company" not in payload
    assert "history" not in payload
    assert "emotion" not in payload
    assert payload["k"] == 3


def test_kb_doc_parsing(monkeypatch) -> None:
    captured: dict = {}

    def fake_get(url, params=None, timeout=None, **kwargs):
        captured["url"] = url
        captured["params"] = params
        return _FakeResponse(KB_DOC_JSON)

    monkeypatch.setattr(requests, "get", fake_get)

    doc = CopilotClient().kb_doc("no_internet")
    assert captured["url"].endswith("/api/kb/doc")
    assert captured["params"] == {"doc_id": "no_internet"}
    assert doc["doc_id"] == "no_internet"
    assert doc["company"] == "orbita"
    assert doc["text"] == "Полный текст статьи базы знаний."
