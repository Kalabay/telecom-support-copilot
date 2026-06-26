"""Mock pipeline для разработки фронта до подключения реальных моделей."""

import asyncio
import json
import random
from collections.abc import AsyncIterator
from pathlib import Path

from loguru import logger

_ROOT = Path(__file__).resolve().parents[3]
_E2E_DIALOGUES = _ROOT / "eval" / "e2e_dialogues.json"
_E2E_MANIFEST = _ROOT / "eval" / "audio" / "e2e" / "manifest.json"
_DOC_ANSWERS = _ROOT / "eval" / "results" / "doc_answers.json"
DEMO_DIALOGUE_ID = "vek_10"

from app.models.schemas import (
    CopilotUpdate,
    Emotion,
    EmotionState,
    KBSource,
    LatencyBreakdown,
    Suggestion,
    TranscriptSegment,
)

DEMO_SCRIPT: list[dict] = [
    {
        "delay_ms": 1500,
        "transcript": "Алло, добрый день. У меня третий день не работает интернет.",
        "audio": "demo_01_greeting.wav",
        "emotion": Emotion.NEUTRAL,
        "arousal": 0.3,
        "valence": -0.2,
    },
    {
        "delay_ms": 2000,
        "transcript": "Я уже два раза перезагружал роутер, ничего не помогает.",
        "audio": "demo_02_diagnostic.wav",
        "emotion": Emotion.NEUTRAL,
        "arousal": 0.45,
        "valence": -0.3,
    },
    {
        "delay_ms": 1800,
        "transcript": "Я плачу вам каждый месяц, а услугу не получаю!",
        "audio": "demo_03_complaint.wav",
        "emotion": Emotion.ANGRY,
        "arousal": 0.75,
        "valence": -0.7,
    },
    {
        "delay_ms": 2200,
        "transcript": "Если сегодня не починят, я расторгну договор и уйду к Билайну!",
        "audio": "demo_04_threat.wav",
        "emotion": Emotion.ANGRY,
        "arousal": 0.9,
        "valence": -0.85,
    },
]

DEMO_KB_SOURCES = [
    KBSource(
        doc_id="mts_no_internet_diagnostics",
        title="МТС: диагностика отсутствия интернета",
        snippet="1. Проверьте индикаторы на роутере. 2. Перезагрузите устройство...",
        score=0.91,
    ),
    KBSource(
        doc_id="mts_outage_map",
        title="МТС: карта плановых работ и аварий",
        snippet="Проверьте регион клиента на наличие массовых отключений...",
        score=0.84,
    ),
    KBSource(
        doc_id="retention_script_angry",
        title="Скрипт удержания при угрозе расторжения",
        snippet="Принести извинения, предложить компенсацию (3-7 дней бесплатно)...",
        score=0.78,
    ),
]

DEMO_SUGGESTIONS_PROGRESSION = [
    [
        Suggestion(
            text="Понял вас. Давайте проверим, что с вашим подключением. "
                 "Уточните, пожалуйста, ваш адрес?",
            rank=1,
            intent="diagnose_internet",
            sources=DEMO_KB_SOURCES[:1],
        ),
        Suggestion(
            text="Подскажите модель роутера и какие индикаторы горят?",
            rank=2,
            intent="diagnose_internet",
            sources=DEMO_KB_SOURCES[:1],
        ),
    ],
    [
        Suggestion(
            text="Проверяю аварии в вашем районе... Одну минуту, пожалуйста.",
            rank=1,
            intent="check_outage",
            sources=DEMO_KB_SOURCES[1:2],
        ),
        Suggestion(
            text="Спасибо, что уже пробовали перезагрузку. "
                 "Сейчас посмотрю, нет ли работ в вашем районе.",
            rank=2,
            intent="check_outage",
            sources=DEMO_KB_SOURCES[1:2],
        ),
    ],
    [
        Suggestion(
            text="Полностью вас понимаю, ситуация неприятная. "
                 "Сейчас разберёмся и подберём решение.",
            rank=1,
            intent="empathy_apology",
            sources=DEMO_KB_SOURCES[2:],
        ),
        Suggestion(
            text="Приношу извинения за неудобства. "
                 "Давайте я зафиксирую обращение и подключу техника.",
            rank=2,
            intent="empathy_apology",
            sources=DEMO_KB_SOURCES[2:],
        ),
    ],
    [
        Suggestion(
            text="Понимаю ваше недовольство. Я могу прямо сейчас оформить компенсацию "
                 "за дни простоя и направить выездного техника на сегодня.",
            rank=1,
            intent="retention_compensation",
            sources=DEMO_KB_SOURCES[2:],
        ),
        Suggestion(
            text="Давайте не будем спешить с расторжением. "
                 "Я оформляю заявку приоритетным статусом, "
                 "техник приедет в течение 4 часов. Согласны?",
            rank=2,
            intent="retention_compensation",
            sources=DEMO_KB_SOURCES[2:],
        ),
    ],
]


def _fake_latency() -> LatencyBreakdown:
    asr = random.randint(180, 320)
    ser = random.randint(60, 110)
    retrieval = random.randint(40, 90)
    llm = random.randint(700, 1400)
    return LatencyBreakdown(
        asr_ms=asr,
        ser_ms=ser,
        retrieval_ms=retrieval,
        llm_ms=llm,
        total_ms=asr + ser + retrieval + llm,
    )


def _try_real_retrieval(query: str, k: int = 3) -> tuple[list[KBSource], int] | None:
    """Вернуть (sources, latency_ms) из ChromaDB или None, если индекс недоступен."""
    try:
        from app.pipeline.rag import get_retriever

        result = get_retriever().search(query, k=k)
        return result.sources, result.total_ms
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"RAG retrieval unavailable: {exc}")
        return None


def _try_real_ser(audio_filename: str | None) -> tuple[EmotionState, int] | None:
    """Вернуть (emotion_state, latency_ms) из SER на парном аудио или None."""
    if not audio_filename:
        return None
    try:
        from pathlib import Path

        from app.core.config import settings
        from app.pipeline.ser import get_recognizer

        path = Path(settings.audio_dir) / audio_filename
        if not path.exists():
            logger.warning(f"Demo audio not found: {path}")
            return None
        result = get_recognizer().predict(str(path))
        return result.state, result.inference_ms
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"SER inference unavailable: {exc}")
        return None


def _try_real_llm(
    transcript: list[str],
    emotion: EmotionState,
    sources: list[KBSource],
) -> tuple[list[str], int] | None:
    """Сгенерировать через T-lite список текстов suggestions или None при сбое."""
    try:
        from app.pipeline.llm import get_generator

        result = get_generator().generate(transcript, emotion, sources)
        if not result.suggestions:
            return None
        return result.suggestions, result.total_ms
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"LLM generation unavailable: {exc}")
        return None


_EMO_MAP = {
    "angry": Emotion.ANGRY, "sad": Emotion.SAD,
    "positive": Emotion.POSITIVE, "neutral": Emotion.NEUTRAL,
}
_EMO_VA = {
    Emotion.ANGRY: (0.85, -0.7), Emotion.SAD: (0.4, -0.5),
    Emotion.POSITIVE: (0.6, 0.55), Emotion.NEUTRAL: (0.4, 0.0),
}


def _load_demo() -> tuple[dict | None, dict, dict]:
    """Загрузить демо-диалог (реальный озвученный пример), манифест аудио и плейбук."""
    dlg = next((d for d in json.loads(_E2E_DIALOGUES.read_text(encoding="utf-8"))
                if d["dialogue_id"] == DEMO_DIALOGUE_ID), None) if _E2E_DIALOGUES.exists() else None
    man: dict = {}
    if _E2E_MANIFEST.exists():
        for m in json.loads(_E2E_MANIFEST.read_text(encoding="utf-8")):
            man[(m["dialogue_id"], m["turn_idx"])] = m
    answers: dict = {}
    if _DOC_ANSWERS.exists():
        answers = {o["doc_id"]: o.get("answers", [])
                   for o in json.loads(_DOC_ANSWERS.read_text(encoding="utf-8"))}
    return dlg, man, answers


async def run_demo() -> AsyncIterator[CopilotUpdate]:
    """Детерминированный демо-звонок на реальном озвученном примере (Вектор):
    играет аудио клиента, эталонные подсказки выбираются автоматически."""
    dlg, man, answers = _load_demo()
    if not dlg:
        yield CopilotUpdate(transcript=[], pipeline_stage="idle")
        return

    turns = dlg["turns"]
    ideals = {t["idx"]: t.get("ideal_text", "") for t in turns if t["role"] == "operator"}
    transcript: list[TranscriptSegment] = []
    cursor_ms = 0
    last_emotion: EmotionState | None = None

    yield CopilotUpdate(transcript=[], pipeline_stage="listening")

    for ct in (t for t in turns if t["role"] == "client"):
        idx = ct["idx"]
        mm = man.get((DEMO_DIALOGUE_ID, idx))
        clean = (mm or {}).get("clean_text") or ct.get("text", "")
        audio_url = f"/api/benchmark/audio/{mm['file']}" if mm and mm.get("ok") else None

        # клиент говорит — играем аудио
        yield CopilotUpdate(transcript=transcript, audio_url=audio_url, pipeline_stage="listening")
        await asyncio.sleep(max(2.0, min(9.0, len(clean) * 0.07 + 1.2)))

        yield CopilotUpdate(transcript=transcript, pipeline_stage="transcribing")
        await asyncio.sleep(0.4)
        start = cursor_ms
        end = cursor_ms + len(clean) * 60
        cursor_ms = end + 300
        transcript.append(TranscriptSegment(
            text=clean, speaker="customer", start_ms=start, end_ms=end, confidence=0.94))

        emo_enum = _EMO_MAP.get(ct.get("emotion", "neutral"), Emotion.NEUTRAL)
        arousal, valence = _EMO_VA[emo_enum]
        emotion = EmotionState(
            label=emo_enum, confidence=0.90, arousal=arousal, valence=valence,
            escalation_risk=(emo_enum == Emotion.ANGRY),
        )
        last_emotion = emotion
        yield CopilotUpdate(transcript=transcript, emotion=emotion, pipeline_stage="analyzing")
        await asyncio.sleep(0.4)
        yield CopilotUpdate(transcript=transcript, emotion=emotion, pipeline_stage="retrieving")
        await asyncio.sleep(0.4)
        yield CopilotUpdate(transcript=transcript, emotion=emotion, pipeline_stage="generating")
        await asyncio.sleep(0.5)

        ideal = ideals.get(idx + 1, "")
        gold = ct.get("gold_doc_ids", [])
        pb = answers.get(gold[0], []) if gold else []
        texts = [t for t in ([ideal] + [a for a in pb if a != ideal]) if t][:3] or [ideal or clean]
        srcs = [KBSource(doc_id=gold[0], title=gold[0].replace("_", " "), snippet="", score=0.9)] if gold else []
        suggestions = [Suggestion(text=tx, rank=r + 1, sources=srcs) for r, tx in enumerate(texts)]
        latency = LatencyBreakdown(asr_ms=240, ser_ms=80, retrieval_ms=60, llm_ms=1150, total_ms=1530)
        yield CopilotUpdate(
            transcript=transcript, emotion=emotion, suggestions=suggestions,
            latency=latency, pipeline_stage="ready",
        )

        # автоматический выбор лучшего → ответ оператора появляется сам
        await asyncio.sleep(2.4)
        chosen = suggestions[0].text
        ostart = cursor_ms
        oend = cursor_ms + len(chosen) * 55
        cursor_ms = oend + 400
        transcript.append(TranscriptSegment(
            text=chosen, speaker="operator", start_ms=ostart, end_ms=oend, confidence=1.0))
        yield CopilotUpdate(
            transcript=transcript, emotion=emotion, suggestions=[],
            latency=latency, pipeline_stage="ready",
        )

    await asyncio.sleep(1.2)
    yield CopilotUpdate(transcript=transcript, emotion=last_emotion, pipeline_stage="idle")
