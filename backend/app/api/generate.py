"""REST: POST /api/generate — синхронная LLM-генерация на произвольном вводе."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from app.models.schemas import EmotionState, KBSource
from app.pipeline.llm import get_generator

router = APIRouter()


class GenerateRequest(BaseModel):
    transcript: list[str] = Field(..., description="История реплик клиента, последняя — самая свежая")
    emotion: EmotionState
    sources: list[KBSource] = Field(default_factory=list)
    max_tokens: int = 220
    use_emotion: bool = Field(True, description="Включать ли блок эмоции в промпт (для абляции)")


class GenerateResponse(BaseModel):
    suggestions: list[str]
    raw_completion: str
    prompt_tokens: int
    completion_tokens: int
    total_ms: int


@router.post("/generate", response_model=GenerateResponse)
def generate_suggestions(req: GenerateRequest) -> GenerateResponse:
    if not req.transcript:
        raise HTTPException(400, "transcript must not be empty")
    try:
        result = get_generator().generate(
            transcript=req.transcript,
            emotion=req.emotion,
            sources=req.sources,
            max_tokens=req.max_tokens,
            use_emotion=req.use_emotion,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("LLM generation failed")
        raise HTTPException(500, f"LLM generation failed: {exc}") from exc

    return GenerateResponse(
        suggestions=result.suggestions,
        raw_completion=result.raw_completion,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_ms=result.total_ms,
    )


class TitleRequest(BaseModel):
    text: str = Field(..., description="Первая реплика клиента")


class TitleResponse(BaseModel):
    title: str


@router.post("/title", response_model=TitleResponse)
def make_title(req: TitleRequest) -> TitleResponse:
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "text must not be empty")
    try:
        title = get_generator().make_title(text)
    except Exception as exc:  # noqa: BLE001
        logger.exception("title generation failed")
        raise HTTPException(500, f"title failed: {exc}") from exc
    return TitleResponse(title=title or "Новый чат")
