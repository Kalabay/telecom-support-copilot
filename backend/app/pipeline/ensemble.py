"""Высокоточный SER: ансамбль GigaAM + HuBERT + наша дообученная wav2vec2 (стекинг)."""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from loguru import logger

from app.models.schemas import Emotion, EmotionState
from app.pipeline.ser import _VA_BY_LABEL, SERResult, get_recognizer

CLASSES = ["neutral", "angry", "positive", "sad"]
_LM = {"neutral": Emotion.NEUTRAL, "angry": Emotion.ANGRY,
       "positive": Emotion.POSITIVE, "sad": Emotion.SAD}
STACK_PATH = Path(__file__).resolve().parent / "artifacts" / "ser_stack.json"
_OURS_CANDIDATES = [Path(r"K:\.caches\ser_finetuned\best"), Path(r"K:\.caches\ser_finetuned")]


def _align(probs_by_name: dict[str, float]) -> np.ndarray:
    low = {str(k).lower(): float(v) for k, v in probs_by_name.items()}
    return np.array([low.get(c, 0.0) for c in CLASSES], dtype=np.float32)


@dataclass
class FusionResult:
    state: EmotionState
    probs: dict[str, float]
    gate: dict[str, float]
    inference_ms: int
    duration_ms: int


class EnsembleRecognizer:
    """Стекинг трёх экспертов поверх их вероятностей. Fallback на HuBERT."""

    _instance: EnsembleRecognizer | None = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._stack = None
        self._gigaam = None
        self._ours = None
        self._ours_fe = None
        self._ours_id2label: dict[int, str] = {}
        self._lock_predict = Lock()
        self._available = STACK_PATH.exists()
        self._initialized = True

    def _load_stack(self) -> bool:
        if self._stack is not None:
            return True
        if not STACK_PATH.exists():
            return False
        d = json.loads(STACK_PATH.read_text(encoding="utf-8"))
        self._stack = (np.array(d["coef"], dtype=np.float32), np.array(d["intercept"], dtype=np.float32))
        return True

    def _probs_gigaam(self, wav: np.ndarray) -> np.ndarray:
        import soundfile as sf
        if self._gigaam is None:
            _orig = torch.load
            torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
            import gigaam
            self._gigaam = gigaam.load_model("emo")
            torch.load = _orig
            self._ga_tmp = Path(tempfile.mkdtemp(prefix="ens_ga_")) / "x.wav"
        sf.write(self._ga_tmp, wav, 16000, subtype="PCM_16")
        return _align(self._gigaam.get_probs(str(self._ga_tmp)))

    def _probs_hubert(self, audio) -> np.ndarray:  # noqa: ANN001
        return _align(get_recognizer().predict(audio).probs)

    def _probs_ours(self, wav: np.ndarray) -> np.ndarray:
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        if self._ours is None:
            path = next((p for p in _OURS_CANDIDATES if (p / "config.json").exists()), None)
            if path is None:
                raise FileNotFoundError("ser_finetuned not found")
            self._ours_fe = AutoFeatureExtractor.from_pretrained(str(path))
            self._ours = AutoModelForAudioClassification.from_pretrained(str(path)).to(self._device).eval()
            self._ours_id2label = {int(k): v for k, v in self._ours.config.id2label.items()}
        inp = self._ours_fe(wav, sampling_rate=16000, return_tensors="pt")
        inp = {k: v.to(self._device) for k, v in inp.items()}
        with torch.inference_mode():
            logits = self._ours(**inp).logits
        probs = torch.softmax(logits, -1).cpu().numpy()[0]
        return _align({self._ours_id2label[i]: float(p) for i, p in enumerate(probs)})

    def predict(self, audio) -> SERResult | FusionResult:  # noqa: ANN001
        rec = get_recognizer()
        if not (self._available and self._load_stack()):
            return rec.predict(audio)
        try:
            wav, sr = rec.load_audio(audio)
            if sr != 16000:
                import librosa
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            duration_ms = int(len(wav) / 16000 * 1000)
            if len(wav) > 10 * 16000:
                wav = wav[: 10 * 16000]
            t0 = time.perf_counter()
            with self._lock_predict:
                x = np.concatenate([self._probs_gigaam(wav), self._probs_hubert(audio), self._probs_ours(wav)])
                coef, intercept = self._stack
                logits = coef @ x + intercept
                probs = np.exp(logits - logits.max())
                probs = probs / probs.sum()
            inference_ms = int((time.perf_counter() - t0) * 1000)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"ensemble predict failed: {exc}; fallback to HuBERT")
            return rec.predict(audio)

        pbl = {CLASSES[i]: float(p) for i, p in enumerate(probs)}
        top = max(pbl, key=pbl.get)
        emotion = _LM[top]
        conf = pbl[top]
        arousal, valence = _VA_BY_LABEL[emotion]
        if emotion is Emotion.ANGRY:
            arousal = min(1.0, arousal + 0.15 * (conf - 0.5))
        state = EmotionState(
            label=emotion, confidence=round(conf, 3),
            arousal=round(arousal, 3), valence=round(valence, 3),
            escalation_risk=(emotion is Emotion.ANGRY and conf > 0.55),
        )
        return FusionResult(
            state=state, probs={k: round(v, 4) for k, v in pbl.items()},
            gate={"gigaam": 0.33, "hubert": 0.33, "ours": 0.34},
            inference_ms=inference_ms, duration_ms=duration_ms,
        )


@lru_cache(maxsize=1)
def get_ensemble_recognizer() -> EnsembleRecognizer:
    return EnsembleRecognizer()
