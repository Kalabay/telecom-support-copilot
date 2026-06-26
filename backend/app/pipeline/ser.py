"""SER: Speech Emotion Recognition через HuBERT, fine-tuned на Dusha."""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from functools import lru_cache
from threading import Lock

import librosa
import numpy as np
import torch
from loguru import logger
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from app.models.schemas import Emotion, EmotionState

MODEL_NAME = "xbgoose/hubert-large-speech-emotion-recognition-russian-dusha-finetuned"
TARGET_SR = 16000

_LABEL_MAP: dict[str, Emotion] = {
    "neutral": Emotion.NEUTRAL,
    "angry": Emotion.ANGRY,
    "positive": Emotion.POSITIVE,
    "sad": Emotion.SAD,
    "other": Emotion.OTHER,
}

_LA_TAU = 0.9
_CLASS_PRIOR = {
    "neutral": 0.57,
    "angry": 0.16,
    "positive": 0.13,
    "sad": 0.11,
    "other": 0.03,
}

_VA_BY_LABEL: dict[Emotion, tuple[float, float]] = {
    Emotion.NEUTRAL: (0.30, 0.00),
    Emotion.ANGRY: (0.80, -0.70),
    Emotion.POSITIVE: (0.60, 0.70),
    Emotion.SAD: (0.30, -0.50),
    Emotion.OTHER: (0.50, 0.00),
}


@dataclass
class SERResult:
    state: EmotionState
    probs: dict[str, float]
    inference_ms: int
    duration_ms: int


class EmotionRecognizer:
    """Ленивый singleton для SER."""

    _instance: EmotionRecognizer | None = None
    _lock = Lock()

    def __new__(cls) -> EmotionRecognizer:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._model = None
        self._extractor = None
        self._id2label: dict[int, str] = {}
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialized = True

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        logger.info(f"Loading SER model: {MODEL_NAME} on {self._device}")
        t0 = time.perf_counter()
        self._extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
        model.to(self._device).eval()
        self._model = model
        self._id2label = model.config.id2label
        logger.info(
            f"SER loaded in {time.perf_counter() - t0:.1f}s, "
            f"labels={list(self._id2label.values())}"
        )

    @staticmethod
    def load_audio(audio: bytes | str | np.ndarray) -> tuple[np.ndarray, int]:
        """Принимает bytes/path/уже-numpy и возвращает (waveform 1-d float32, sr)."""
        if isinstance(audio, np.ndarray):
            return audio.astype(np.float32), TARGET_SR
        if isinstance(audio, (bytes, bytearray)):
            wav, sr = librosa.load(io.BytesIO(audio), sr=TARGET_SR, mono=True)
        else:
            wav, sr = librosa.load(audio, sr=TARGET_SR, mono=True)
        return wav.astype(np.float32), sr

    def predict(self, audio: bytes | str | np.ndarray) -> SERResult:
        self._ensure_loaded()
        assert self._model is not None and self._extractor is not None

        wav, sr = self.load_audio(audio)
        duration_ms = int(len(wav) / sr * 1000)

        if len(wav) > 10 * sr:
            wav = wav[: 10 * sr]

        t0 = time.perf_counter()
        inputs = self._extractor(
            wav, sampling_rate=sr, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.inference_mode():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        inference_ms = int((time.perf_counter() - t0) * 1000)

        probs_by_label = {
            self._id2label[i].lower(): float(p) for i, p in enumerate(probs)
        }
        import math

        def _adj_score(label: str, p: float) -> float:
            prior = _CLASS_PRIOR.get(label, 0.05)
            return math.log(max(p, 1e-9)) - _LA_TAU * math.log(prior)

        top_label = max(probs_by_label, key=lambda lab: _adj_score(lab, probs_by_label[lab]))
        emotion = _LABEL_MAP.get(top_label, Emotion.OTHER)
        confidence = probs_by_label[top_label]
        arousal, valence = _VA_BY_LABEL[emotion]
        if emotion is Emotion.ANGRY:
            arousal = min(1.0, arousal + 0.15 * (confidence - 0.5))

        state = EmotionState(
            label=emotion,
            confidence=round(confidence, 3),
            arousal=round(arousal, 3),
            valence=round(valence, 3),
            escalation_risk=(emotion is Emotion.ANGRY and confidence > 0.55),
        )

        return SERResult(
            state=state,
            probs={k: round(v, 4) for k, v in probs_by_label.items()},
            inference_ms=inference_ms,
            duration_ms=duration_ms,
        )


@lru_cache(maxsize=1)
def get_recognizer() -> EmotionRecognizer:
    return EmotionRecognizer()
