import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api import analytics, asr, assist, benchmark, generate, health, kb, ser, websocket
from app.core.config import settings


async def _warm_retriever() -> None:
    """Прогревает BGE-M3 + ChromaDB в фоне, чтобы первый WS-запрос не ждал ~10 сек."""
    try:
        from app.pipeline.rag import get_retriever

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: get_retriever().search("warmup", k=1))
        logger.info("Retriever warmed up")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Retriever warmup failed: {exc}")


async def _warm_ser() -> None:
    """Прогревает HuBERT-Dusha SER (~3 сек на K:\\.hf_cache)."""
    try:
        import numpy as np

        from app.pipeline.ser import TARGET_SR, get_recognizer

        loop = asyncio.get_event_loop()
        silence = np.zeros(TARGET_SR, dtype=np.float32)
        await loop.run_in_executor(None, lambda: get_recognizer().predict(silence))
        logger.info("SER warmed up")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"SER warmup failed: {exc}")


async def _warm_asr() -> None:
    """Прогревает faster-whisper."""
    logger.info("ASR warm task entered")
    try:
        import numpy as np

        from app.pipeline.asr import TARGET_SR, get_asr

        logger.info("ASR module imported")
        loop = asyncio.get_event_loop()
        silence = np.zeros(TARGET_SR, dtype=np.float32)
        await loop.run_in_executor(None, lambda: get_asr().transcribe(silence))
        logger.info("ASR warmed up")
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"ASR warmup failed: {exc}")


async def _warm_llm() -> None:
    """Прогревает T-lite GGUF — загрузка ~4.6 ГБ в VRAM, ~10 сек."""
    try:
        from app.models.schemas import Emotion, EmotionState, KBSource
        from app.pipeline.llm import get_generator

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: get_generator().generate(
                transcript=["Проверка"],
                emotion=EmotionState(
                    label=Emotion.NEUTRAL,
                    confidence=0.9,
                    arousal=0.3,
                    valence=0.0,
                    escalation_risk=False,
                ),
                sources=[
                    KBSource(
                        doc_id="warmup",
                        title="прогрев",
                        snippet="тестовый прогрев",
                        score=1.0,
                    )
                ],
                max_tokens=8,
            ),
        )
        logger.info("LLM warmed up")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"LLM warmup failed: {exc}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info(f"Starting {settings.app_name} (mock_mode={settings.mock_mode})")
    rag_task = asyncio.create_task(_warm_retriever())
    ser_task = asyncio.create_task(_warm_ser())
    llm_task = asyncio.create_task(_warm_llm())
    asr_task = asyncio.create_task(_warm_asr())
    yield
    for t in (rag_task, ser_task, llm_task, asr_task):
        t.cancel()
    logger.info("Shutting down")


app = FastAPI(
    title=settings.app_name,
    description="LLM-копилот для операторов телеком-техподдержки",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(asr.router, prefix="/api", tags=["asr"])
app.include_router(ser.router, prefix="/api", tags=["ser"])
app.include_router(generate.router, prefix="/api", tags=["llm"])
app.include_router(assist.router, prefix="/api", tags=["integration"])
app.include_router(kb.router, prefix="/api", tags=["kb"])
app.include_router(analytics.router, prefix="/api", tags=["analytics"])
app.include_router(benchmark.router, prefix="/api", tags=["benchmark"])
app.include_router(websocket.router, tags=["websocket"])
