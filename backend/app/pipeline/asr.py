"""ASR с переключаемым движком: Whisper turbo/medium, GigaAM, T-one."""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from functools import lru_cache
from threading import Lock

import numpy as np
from loguru import logger

ENGINES: dict[str, tuple[str, str, str]] = {
    "whisper-turbo": ("whisper", "large-v3-turbo", "Whisper large-v3-turbo"),
    "whisper-medium": ("whisper", "medium", "Whisper medium"),
    "gigaam-v2": ("onnx", "gigaam-v2-ctc", "GigaAM-v2 (Сбер)"),
    "t-one": ("onnx", "t-tech/t-one", "T-one (Т-банк)"),
}
DEFAULT_ENGINE = "whisper-turbo"
COMPUTE_TYPE = "int8_float16"
TARGET_SR = 16000


@dataclass
class ASRResult:
    text: str
    language: str
    language_prob: float
    duration_sec: float
    inference_ms: int


class SpeechRecognizer:
    _instance: SpeechRecognizer | None = None
    _lock = Lock()

    def __new__(cls) -> SpeechRecognizer:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._models: dict[str, tuple[str, object]] = {}
        from app.core.config import settings
        self._current = getattr(settings, "asr_engine", None) or DEFAULT_ENGINE
        if self._current not in ENGINES:
            self._current = DEFAULT_ENGINE
        self._initialized = True

    def set_engine(self, name: str) -> bool:
        if name in ENGINES:
            self._current = name
            return True
        return False

    @property
    def current(self) -> str:
        return self._current

    def _load(self, name: str) -> tuple[str, object]:
        if name in self._models:
            return self._models[name]
        etype, model_id, _ = ENGINES[name]
        t0 = time.perf_counter()
        if etype == "whisper":
            from faster_whisper import WhisperModel
            m = WhisperModel(model_id, device="cuda", compute_type=COMPUTE_TYPE)
        else:
            import onnx_asr
            m = onnx_asr.load_model(model_id)
        logger.info(f"ASR engine '{name}' loaded in {time.perf_counter() - t0:.1f}s")
        self._models[name] = (etype, m)
        return self._models[name]

    def _ensure_loaded(self) -> None:
        self._load(self._current)

    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None = "ru",
        beam_size: int = 5,
    ) -> ASRResult:
        """Распознать одну реплику. audio = float32 numpy 1-D, sr=16k."""
        etype, model = self._load(self._current)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        peak = float(np.max(np.abs(audio))) or 1.0
        if peak > 1.5:
            audio = audio / peak

        t0 = time.perf_counter()
        if etype == "whisper":
            segments, info = model.transcribe(
                audio,
                language=language,
                beam_size=beam_size,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
                condition_on_previous_text=False,
            )
            text = " ".join(s.text.strip() for s in segments if s.text.strip()).strip()
            ms = int((time.perf_counter() - t0) * 1000)
            return ASRResult(text, info.language, info.language_probability, info.duration, ms)
        text = (model.recognize(audio) or "").strip()
        ms = int((time.perf_counter() - t0) * 1000)
        return ASRResult(text, "ru", 1.0, len(audio) / TARGET_SR, ms)


@lru_cache(maxsize=1)
def get_asr() -> SpeechRecognizer:
    return SpeechRecognizer()


def decode_audio_blob(blob: bytes) -> np.ndarray:
    """Декодировать webm/opus/wav/mp3 → float32 numpy 16k mono."""
    import av

    container = av.open(io.BytesIO(blob))
    try:
        stream = next(s for s in container.streams if s.type == "audio")
    except StopIteration as exc:
        raise ValueError("no audio stream in blob") from exc

    resampler = av.AudioResampler(format="flt", layout="mono", rate=TARGET_SR)
    samples: list[np.ndarray] = []
    for frame in container.decode(stream):
        for resampled in resampler.resample(frame):
            arr = resampled.to_ndarray()
            if arr.ndim == 2:
                arr = arr.mean(axis=0)
            samples.append(arr.astype(np.float32))
    for resampled in resampler.resample(None):
        arr = resampled.to_ndarray()
        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        samples.append(arr.astype(np.float32))
    container.close()

    if not samples:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(samples).astype(np.float32)
