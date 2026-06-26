"""Озвучивает client-реплики из data/synthetic/dialogs.json через Silero TTS."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import app  # noqa: F401,E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from scipy.signal import butter, sosfiltfilt  # noqa: E402

DIALOGS_FILE = PROJECT_ROOT / "data" / "synthetic" / "dialogs.json"
AUDIO_DIR = PROJECT_ROOT / "data" / "synthetic" / "audio"
MANIFEST = PROJECT_ROOT / "data" / "synthetic" / "audio_manifest.json"

SR_TTS = 24000
SR_OUT = 16000
PHONE_LOW = 300.0
PHONE_HIGH = 3400.0

SPEAKERS = ["aidar", "baya", "kseniya", "xenia"]


def telephony_filter(audio: np.ndarray, sr: int) -> np.ndarray:
    """Butterworth bandpass 300-3400 Hz — стандартный голосовой канал."""
    sos = butter(
        N=4,
        Wn=[PHONE_LOW / (sr / 2), PHONE_HIGH / (sr / 2)],
        btype="bandpass",
        output="sos",
    )
    return sosfiltfilt(sos, audio).astype(np.float32)


def downsample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Простой polyphase resampling через scipy."""
    from scipy.signal import resample_poly

    from math import gcd

    g = gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    return resample_poly(audio, up, down).astype(np.float32)


def main() -> None:
    if not DIALOGS_FILE.exists():
        sys.exit(f"Run scripts/synth_dialogs.py first; not found: {DIALOGS_FILE}")

    dialogs = json.loads(DIALOGS_FILE.read_text(encoding="utf-8"))
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading Silero TTS ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language="ru",
        speaker="v4_ru",
        trust_repo=True,
    )
    model.to(device)

    manifest: list[dict] = []
    total_clips = 0
    for dialog in dialogs:
        spk = SPEAKERS[hash(dialog["dialog_id"]) % len(SPEAKERS)]
        turn_idx = 0
        for turn in dialog["turns"]:
            if turn["speaker"] != "customer":
                continue
            text = turn["text"].strip()
            if not text or len(text) > 800:
                continue
            file_id = f"{dialog['dialog_id']}_t{turn_idx:02d}.wav"
            audio_path = AUDIO_DIR / file_id

            try:
                wav = model.apply_tts(
                    text=text,
                    speaker=spk,
                    sample_rate=SR_TTS,
                    put_accent=True,
                    put_yo=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {file_id}: TTS failed ({exc})")
                turn_idx += 1
                continue

            audio = wav.detach().cpu().numpy().astype(np.float32)
            audio = downsample(audio, SR_TTS, SR_OUT)
            audio = telephony_filter(audio, SR_OUT)

            peak = float(np.max(np.abs(audio))) or 1.0
            audio = (audio / peak) * 0.7

            sf.write(audio_path, audio, SR_OUT, subtype="PCM_16")
            manifest.append(
                {
                    "file": file_id,
                    "dialog_id": dialog["dialog_id"],
                    "scenario": dialog["scenario"],
                    "label": turn["emotion"],
                    "text": text,
                    "speaker": spk,
                    "duration_sec": round(len(audio) / SR_OUT, 2),
                }
            )
            total_clips += 1
            turn_idx += 1
            if total_clips % 10 == 0:
                print(f"  ... {total_clips} clips done")

    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSaved {total_clips} clips -> {AUDIO_DIR}")
    print(f"Manifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
