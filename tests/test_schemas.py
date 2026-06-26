"""Тесты pydantic-схем бэкенда: round-trip, дефолты, валидация."""

from __future__ import annotations

from app.api.assist import AssistRequest
from app.models.schemas import ClientMessage, KBSource, TranscriptSegment


def test_transcript_segment_sources_round_trip() -> None:
    seg = TranscriptSegment(
        text="Третий день нет интернета!",
        speaker="customer",
        start_ms=0,
        end_ms=1500,
        confidence=0.92,
        sources=[
            KBSource(doc_id="no_internet", title="Нет интернета", snippet="...", score=0.81),
        ],
    )
    dumped = seg.model_dump()
    restored = TranscriptSegment.model_validate(dumped)

    assert restored == seg
    assert restored.sources[0].doc_id == "no_internet"
    assert restored.sources[0].score == 0.81
    assert restored.speaker == "customer"


def test_assist_request_defaults() -> None:
    req = AssistRequest(text="Нет связи")
    assert req.text == "Нет связи"
    assert req.company is None
    assert req.history == []
    assert req.emotion is None
    assert req.k == 3


def test_client_message_set_company() -> None:
    msg = ClientMessage(type="set_company", payload={"company": "mts"})
    assert msg.type == "set_company"
    assert msg.payload == {"company": "mts"}
