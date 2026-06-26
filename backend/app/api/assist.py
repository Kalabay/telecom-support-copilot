"""REST: POST /api/assist — весь копилот в одном синхронном вызове."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from app.models.schemas import Emotion, EmotionState, KBSource, Suggestion
from app.pipeline.llm import get_generator
from app.pipeline.rag import get_retriever

router = APIRouter()


class Turn(BaseModel):
    speaker: str = "customer"
    text: str


class AssistRequest(BaseModel):
    text: str = Field(..., description="Последняя реплика клиента")
    company: str | None = Field(
        None, description="Тенант базы знаний (mts/beeline/orbita/...); None — по всем"
    )
    history: list[Turn] = Field(default_factory=list, description="Предыдущие реплики")
    emotion: EmotionState | None = Field(
        None, description="Эмоция клиента (если известна из аудио); иначе ответ без учёта эмоции"
    )
    k: int = Field(3, ge=1, le=10, description="Сколько документов доставать из RAG")


class AssistResponse(BaseModel):
    suggestions: list[Suggestion]
    sources: list[KBSource]
    emotion: EmotionState | None
    retrieval_ms: int
    llm_ms: int
    total_ms: int


@router.post("/assist", response_model=AssistResponse)
def assist(req: AssistRequest) -> AssistResponse:
    if not req.text.strip():
        raise HTTPException(400, "text must not be empty")

    t0 = time.perf_counter()
    try:
        prev = next((t.text for t in reversed(req.history) if t.speaker == "customer"), "")
        query = f"{prev} {req.text}".strip()
        rag = get_retriever().search(query, k=req.k, company=req.company)

        turns = [{"speaker": t.speaker, "text": t.text} for t in req.history]
        turns.append({"speaker": "customer", "text": req.text})

        emotion = req.emotion or EmotionState(
            label=Emotion.NEUTRAL, confidence=1.0, arousal=0.2, valence=0.0,
            escalation_risk=False,
        )
        gen = get_generator().generate(
            turns, emotion, rag.sources, max_tokens=320,
            use_emotion=req.emotion is not None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("assist failed")
        raise HTTPException(500, f"assist failed: {exc}") from exc

    suggestions = [
        Suggestion(text=t, rank=i + 1, sources=rag.sources)
        for i, t in enumerate(gen.suggestions[:3])
    ]
    return AssistResponse(
        suggestions=suggestions,
        sources=rag.sources,
        emotion=req.emotion,
        retrieval_ms=rag.total_ms,
        llm_ms=gen.total_ms,
        total_ms=int((time.perf_counter() - t0) * 1000),
    )
