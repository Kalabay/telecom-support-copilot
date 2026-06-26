"""REST-эндпоинт SER: принимает аудио-файл, возвращает EmotionState + probs."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from loguru import logger
from pydantic import BaseModel

from app.core.config import settings
from app.models.schemas import EmotionState
from app.pipeline.ser import get_recognizer

router = APIRouter()


class SERResponse(BaseModel):
    emotion: EmotionState
    probs: dict[str, float]
    inference_ms: int
    duration_ms: int
    filename: str | None = None
    method: str = "hubert"
    gate: dict[str, float] | None = None


@router.post("/ser", response_model=SERResponse)
async def recognize_emotion(file: UploadFile = File(...)) -> SERResponse:
    """Принимает .wav/.mp3/.ogg, возвращает распознанную эмоцию."""
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio file")
    if len(audio_bytes) > 25 * 1024 * 1024:
        raise HTTPException(413, "Audio file too large (>25 MB)")

    try:
        if settings.use_fusion:
            from app.pipeline.fusion import get_fusion_recognizer
            result = get_fusion_recognizer().predict(audio_bytes)
        else:
            result = get_recognizer().predict(audio_bytes)
    except Exception as exc:  # noqa: BLE001
        logger.exception("SER inference failed")
        raise HTTPException(500, f"SER inference failed: {exc}") from exc

    gate = getattr(result, "gate", None)
    return SERResponse(
        emotion=result.state,
        probs=result.probs,
        inference_ms=result.inference_ms,
        duration_ms=result.duration_ms,
        filename=file.filename,
        method="fusion" if gate is not None else "hubert",
        gate=gate,
    )
