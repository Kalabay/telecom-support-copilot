from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Emotion(str, Enum):
    """Dusha 5-class taxonomy."""

    NEUTRAL = "neutral"
    ANGRY = "angry"
    POSITIVE = "positive"
    SAD = "sad"
    OTHER = "other"


class EmotionState(BaseModel):
    label: Emotion
    confidence: float = Field(ge=0.0, le=1.0)
    arousal: float = Field(ge=0.0, le=1.0, description="0 = calm, 1 = excited")
    valence: float = Field(ge=-1.0, le=1.0, description="-1 = negative, 1 = positive")
    escalation_risk: bool = False


class KBSource(BaseModel):
    doc_id: str
    title: str
    snippet: str
    score: float


class TranscriptSegment(BaseModel):
    text: str
    speaker: Literal["customer", "operator"] = "customer"
    start_ms: int
    end_ms: int
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[KBSource] = []


class Suggestion(BaseModel):
    text: str
    rank: int
    sources: list[KBSource] = []
    intent: str | None = None


class LatencyBreakdown(BaseModel):
    asr_ms: int = 0
    ser_ms: int = 0
    retrieval_ms: int = 0
    llm_ms: int = 0
    total_ms: int = 0


class CopilotUpdate(BaseModel):
    """Полное состояние, которое бэкенд шлёт во фронт по WS."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    transcript: list[TranscriptSegment]
    emotion: EmotionState | None = None
    suggestions: list[Suggestion] = []
    latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown)
    pipeline_stage: Literal[
        "idle", "listening", "transcribing", "analyzing", "retrieving", "generating", "ready"
    ] = "idle"
    audio_url: str | None = None


class ClientMessage(BaseModel):
    """Сообщения от клиента (фронта) к бэкенду."""

    type: Literal[
        "audio_chunk", "start", "stop", "ping", "demo_trigger",
        "operator_said", "voice_speaker", "set_company",
    ]
    payload: dict | None = None
