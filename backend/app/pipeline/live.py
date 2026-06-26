"""Live-пайплайн: голосовой ввод от оператора (тестовая запись клиента) → ASR → SER → RAG → LLM."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

from loguru import logger

from app.models.schemas import (
    CopilotUpdate,
    Emotion,
    EmotionState,
    LatencyBreakdown,
    Suggestion,
    TranscriptSegment,
)


async def run_voice_input(
    audio_bytes: bytes,
    transcript: list[TranscriptSegment],
    speaker: str = "customer",
    company: str | None = None,
) -> AsyncIterator[CopilotUpdate]:
    """Прогоняет один голосовой фрагмент через пайплайн и эмитит снапшоты."""
    from app.pipeline.asr import decode_audio_blob, get_asr
    from app.pipeline.llm import get_generator
    from app.pipeline.rag import get_retriever
    from app.pipeline.ser import get_recognizer

    loop = asyncio.get_event_loop()
    is_operator = speaker == "operator"

    yield CopilotUpdate(transcript=list(transcript), pipeline_stage="listening")

    try:
        wav = await loop.run_in_executor(None, decode_audio_blob, audio_bytes)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Audio decode failed: {exc}")
        yield CopilotUpdate(transcript=list(transcript), pipeline_stage="idle")
        return

    if wav.size < 16000 * 0.3:
        logger.warning(f"Audio too short: {wav.size / 16000:.2f}s")
        yield CopilotUpdate(transcript=list(transcript), pipeline_stage="idle")
        return

    yield CopilotUpdate(transcript=list(transcript), pipeline_stage="transcribing")

    asr_result = await loop.run_in_executor(None, lambda: get_asr().transcribe(wav, "ru"))
    text = asr_result.text.strip()
    if not text:
        logger.warning("ASR returned empty text")
        yield CopilotUpdate(transcript=list(transcript), pipeline_stage="idle")
        return

    start_ms = transcript[-1].end_ms + 300 if transcript else 0
    end_ms = start_ms + int(asr_result.duration_sec * 1000)
    seg = TranscriptSegment(
        text=text,
        speaker="operator" if is_operator else "customer",
        start_ms=start_ms,
        end_ms=end_ms,
        confidence=min(0.99, asr_result.language_prob),
    )
    transcript.append(seg)

    if is_operator:
        yield CopilotUpdate(transcript=list(transcript), pipeline_stage="ready")
        return

    yield CopilotUpdate(transcript=list(transcript), pipeline_stage="analyzing")

    from app.core.config import settings
    if settings.use_fusion:
        from app.pipeline.ensemble import get_ensemble_recognizer
        ser_predict = get_ensemble_recognizer().predict
    else:
        ser_predict = get_recognizer().predict
    ser_result = await loop.run_in_executor(None, lambda: ser_predict(wav))
    emotion = ser_result.state

    yield CopilotUpdate(
        transcript=list(transcript),
        emotion=emotion,
        pipeline_stage="retrieving",
    )

    prev_customer = next(
        (s.text for s in reversed(transcript[:-1]) if s.speaker == "customer"), ""
    )
    rag_query = f"{prev_customer} {text}".strip() if prev_customer else text
    rag_result = await loop.run_in_executor(
        None, lambda: get_retriever().search(rag_query, k=3, company=company)
    )
    sources = rag_result.sources

    yield CopilotUpdate(
        transcript=list(transcript),
        emotion=emotion,
        pipeline_stage="generating",
    )

    turns = [{"speaker": s.speaker, "text": s.text} for s in transcript]
    gen_result = await loop.run_in_executor(
        None,
        lambda: get_generator().generate(turns, emotion, sources, max_tokens=320),
    )
    suggestions: list[Suggestion] = []
    for rank, txt in enumerate(gen_result.suggestions[:3], start=1):
        suggestions.append(Suggestion(text=txt, rank=rank, sources=sources))

    latency = LatencyBreakdown(
        asr_ms=asr_result.inference_ms,
        ser_ms=ser_result.inference_ms,
        retrieval_ms=rag_result.total_ms,
        llm_ms=gen_result.total_ms,
        total_ms=(
            asr_result.inference_ms
            + ser_result.inference_ms
            + rag_result.total_ms
            + gen_result.total_ms
        ),
    )
    yield CopilotUpdate(
        transcript=list(transcript),
        emotion=emotion,
        suggestions=suggestions,
        latency=latency,
        pipeline_stage="ready",
    )


_ = (Emotion, EmotionState, time)
