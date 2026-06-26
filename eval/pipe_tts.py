"""Фаза 1 сквозной проверки: озвучить телеком-кейсы через Silero TTS."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import app  # noqa: F401,E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

CASES = PROJECT_ROOT / "eval" / "e2e_cases.json"
SR = 24000
SPEAKERS = ["aidar", "baya", "kseniya", "xenia", "eugene"]


def telephone(au: np.ndarray, sr: int, rng) -> np.ndarray:
    """Телефонный канал: полоса 8кГц + μ-law + шум (реализм колл-центра)."""
    import librosa
    x = au.astype("float32")
    x8 = librosa.resample(x, orig_sr=sr, target_sr=8000)
    mu = 255.0
    x8 = np.clip(x8 / (np.abs(x8).max() + 1e-6), -1, 1)
    comp = np.sign(x8) * np.log1p(mu * np.abs(x8)) / np.log1p(mu)
    q = np.round(comp * 127) / 127
    decomp = np.sign(q) * (1 / mu) * ((1 + mu) ** np.abs(q) - 1)
    x = librosa.resample(decomp.astype("float32"), orig_sr=8000, target_sr=sr)
    snr = rng.uniform(18, 28)
    p_sig = np.mean(x ** 2) + 1e-9
    p_noise = p_sig / (10 ** (snr / 10))
    x = x + rng.randn(len(x)).astype("float32") * np.sqrt(p_noise)
    return (x / (np.abs(x).max() + 1e-6) * 0.95).astype("float32")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path(r"K:\.caches\pipe_audio"))
    args = ap.parse_args()

    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    by_comp: dict[str, list] = {}
    for c in cases:
        by_comp.setdefault(c["company"], []).append(c)
    per = max(1, args.n // len(by_comp))
    pick = []
    for comp, lst in by_comp.items():
        step = max(1, len(lst) // per)
        pick.extend(lst[::step][:per])
    pick = pick[: args.n]

    wav_dir = args.out / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(7)

    model, _ = torch.hub.load("snakers4/silero-models", "silero_tts",
                               language="ru", speaker="v4_ru")
    model.to("cpu")

    man = args.out / "manifest.jsonl"
    n = 0
    with man.open("w", encoding="utf-8") as mf:
        for i, c in enumerate(pick):
            sp = SPEAKERS[i % len(SPEAKERS)]
            try:
                au = model.apply_tts(text=c["text"], speaker=sp, sample_rate=SR).numpy()
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {c['id']}: {exc}", flush=True)
                continue
            au = telephone(au, SR, rng)
            wp = wav_dir / f"{c['id']}.wav"
            sf.write(wp, au.astype("float32"), SR, subtype="PCM_16")
            mf.write(json.dumps({**c, "path": str(wp), "speaker": sp},
                                 ensure_ascii=False) + "\n")
            n += 1
    print(f"озвучено {n} кейсов -> {wav_dir}", flush=True)


if __name__ == "__main__":
    main()
