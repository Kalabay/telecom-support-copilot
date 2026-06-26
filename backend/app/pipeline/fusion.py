"""Multimodal fusion SER: обучаемое слияние замороженных экспертов."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from app.models.schemas import Emotion, EmotionState
from app.pipeline.ser import (
    _VA_BY_LABEL,
    SERResult,
    get_recognizer,
)

ART_PATH = Path(__file__).resolve().parent / "artifacts" / "fusion_head.pt"
CLASSES = ["neutral", "angry", "positive", "sad"]
_LABEL_MAP = {
    "neutral": Emotion.NEUTRAL,
    "angry": Emotion.ANGRY,
    "positive": Emotion.POSITIVE,
    "sad": Emotion.SAD,
}
CEDR_MODEL = "seara/rubert-base-cased-russian-emotion-detection-cedr"


def _make_pre_hook(store: dict):
    """forward-pre-hook: кладёт вход модуля (pooled-эмбеддинг) в store['e']."""
    def hook(module, inp):  # noqa: ANN001
        store["e"] = inp[0].detach().cpu().numpy()
    return hook


class _GMUFusion(nn.Module):
    """GMU поверх frozen-эмбеддингов. Совпадает с eval/train_fusion_artifact.py."""

    def __init__(self, dims, d=256, n_cls=4):
        super().__init__()
        self.proj = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(x), nn.Linear(x, d)) for x in dims])
        self.gate = nn.Linear(sum(dims), d * len(dims))
        self.nm = len(dims)
        self.d = d
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                   nn.Dropout(0.3), nn.Linear(d, n_cls))

    def forward(self, xs, ret_gate=False):
        h = torch.stack([torch.tanh(self.proj[i](xs[i])) for i in range(self.nm)], 1)
        g = torch.softmax(self.gate(torch.cat(xs, -1)).view(-1, self.nm, self.d), 1)
        fused = (g * h).sum(1)
        out = self.head(fused)
        return (out, g.mean(dim=(0, 2))) if ret_gate else out


@dataclass
class FusionResult:
    state: EmotionState
    probs: dict[str, float]
    gate: dict[str, float]
    inference_ms: int
    duration_ms: int


class FusionRecognizer:
    """Ленивый singleton multimodal-fusion SER. Fallback на обычный SER."""

    _instance: FusionRecognizer | None = None
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
        self._predict_lock = Lock()
        self._head: _GMUFusion | None = None
        self._meta: dict | None = None
        self._gigaam = None
        self._hubert_hook_store: dict = {}
        self._cedr = None
        self._cedr_tok = None
        self._asr = None
        self._available = ART_PATH.exists()
        self._initialized = True

    def _load(self) -> bool:
        if self._head is not None:
            return True
        if not ART_PATH.exists():
            logger.warning(f"fusion artifact not found at {ART_PATH}; fallback to SER")
            return False
        try:
            ckpt = torch.load(ART_PATH, map_location=self._device, weights_only=False)
            self._meta = ckpt
            self._head = _GMUFusion(ckpt["dims"]).to(self._device).eval()
            self._head.load_state_dict(ckpt["state_dict"])
            logger.info(f"fusion head loaded: mods={ckpt['mods']} "
                        f"(val macro-F1={ckpt.get('test_macro_f1', '?')})")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error(f"fusion head load failed: {exc}; fallback to SER")
            return False

    def _emb_gigaam(self, wav: np.ndarray) -> np.ndarray:
        import tempfile

        import soundfile as sf
        if self._gigaam is None:
            _orig = torch.load
            torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
            import gigaam
            self._gigaam = gigaam.load_model("emo")
            torch.load = _orig
            self._ga_store: dict = {}
            self._gigaam.head.register_forward_pre_hook(_make_pre_hook(self._ga_store))
            self._ga_tmp = Path(tempfile.mkdtemp(prefix="fus_ga_")) / "x.wav"
        sf.write(self._ga_tmp, wav, 16000, subtype="PCM_16")
        self._gigaam.get_probs(str(self._ga_tmp))
        e = self._ga_store["e"]
        return (e[0] if e.ndim == 2 else e).astype("float32")

    def _emb_hubert(self, wav: np.ndarray) -> np.ndarray:
        rec = get_recognizer()
        rec._ensure_loaded()  # noqa: SLF001
        model, extractor, dev = rec._model, rec._extractor, rec._device  # noqa: SLF001
        if not self._hubert_hook_store.get("registered"):
            model.classifier.register_forward_pre_hook(_make_pre_hook(self._hubert_hook_store))
            self._hubert_hook_store["registered"] = True
        enc = extractor(wav, sampling_rate=16000, return_tensors="pt", padding=True)
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.inference_mode():
            model(**enc)
        e = self._hubert_hook_store["e"]
        return (e[0] if e.ndim == 2 else e).astype("float32")

    def _emb_text(self, wav: np.ndarray) -> np.ndarray:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if self._cedr is None:
            from app.pipeline.asr import get_asr
            self._asr = get_asr()
            self._cedr_tok = AutoTokenizer.from_pretrained(CEDR_MODEL)
            self._cedr = AutoModelForSequenceClassification.from_pretrained(
                CEDR_MODEL).to(self._device).eval()
            self._cedr_store: dict = {}
            self._cedr.classifier.register_forward_pre_hook(_make_pre_hook(self._cedr_store))
        text = self._asr.transcribe(wav, "ru").text.strip() or "."
        enc = self._cedr_tok([text], padding=True, truncation=True, max_length=128,
                             return_tensors="pt").to(self._device)
        with torch.inference_mode():
            self._cedr(**enc)
        return self._cedr_store["e"][0].astype("float32")

    def _embed(self, mod: str, wav: np.ndarray) -> np.ndarray:
        if mod == "ga":
            return self._emb_gigaam(wav)
        if mod == "hu":
            return self._emb_hubert(wav)
        if mod == "text":
            return self._emb_text(wav)
        raise ValueError(mod)

    def predict(self, audio) -> SERResult | FusionResult:
        rec = get_recognizer()
        if not self._load():
            return rec.predict(audio)

        wav, sr = rec.load_audio(audio)
        if sr != 16000:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        duration_ms = int(len(wav) / 16000 * 1000)
        if len(wav) > 10 * 16000:
            wav = wav[: 10 * 16000]

        t0 = time.perf_counter()
        try:
            mods = self._meta["mods"]
            zs = self._meta["zscore"]
            with self._predict_lock:
                xs = []
                for m in mods:
                    e = self._embed(m, wav)
                    mu, sd = zs[m]
                    xs.append(torch.tensor(((e - mu[0]) / sd[0])[None, :],
                                            dtype=torch.float32, device=self._device))
                with torch.inference_mode():
                    logits, gate = self._head(xs, ret_gate=True)
                    probs = torch.softmax(logits, -1).cpu().numpy()[0]
                gate_v = gate.cpu().numpy()
        except Exception as exc:  # noqa: BLE001
            logger.error(f"fusion predict failed: {exc}; fallback to SER")
            return rec.predict(audio)
        inference_ms = int((time.perf_counter() - t0) * 1000)

        probs_by_label = {CLASSES[i]: float(p) for i, p in enumerate(probs)}
        top = max(probs_by_label, key=probs_by_label.get)
        emotion = _LABEL_MAP[top]
        conf = probs_by_label[top]
        arousal, valence = _VA_BY_LABEL[emotion]
        if emotion is Emotion.ANGRY:
            arousal = min(1.0, arousal + 0.15 * (conf - 0.5))
        state = EmotionState(
            label=emotion,
            confidence=round(conf, 3),
            arousal=round(arousal, 3),
            valence=round(valence, 3),
            escalation_risk=(emotion is Emotion.ANGRY and conf > 0.55),
        )
        return FusionResult(
            state=state,
            probs={k: round(v, 4) for k, v in probs_by_label.items()},
            gate={m: round(float(g), 3) for m, g in zip(self._meta["mods"], gate_v)},
            inference_ms=inference_ms,
            duration_ms=duration_ms,
        )


@lru_cache(maxsize=1)
def get_fusion_recognizer() -> FusionRecognizer:
    return FusionRecognizer()
