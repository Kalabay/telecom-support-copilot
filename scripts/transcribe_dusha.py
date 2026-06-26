"""Транскрибирует Dusha (стрим) текущим ASR и сохраняет (text, emotion) в jsonl."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import app  # noqa: F401,E402

from datasets import Audio, load_dataset  # noqa: E402
from rich.console import Console  # noqa: E402

from app.pipeline.asr import decode_audio_blob, get_asr  # noqa: E402
from app.pipeline.ser import get_recognizer  # noqa: E402

CLASSES = {"neutral", "angry", "positive", "sad", "other"}
OUT_DIR = PROJECT_ROOT / "data" / "dusha_text"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--n", type=int, default=400)
    args = ap.parse_args()

    console = Console()
    console.rule(f"[bold cyan]Транскрипция Dusha {args.split}[/]  n={args.n}")

    asr = get_asr()
    console.log("Гружу ASR…")
    asr._ensure_loaded()  # noqa: SLF001
    rec = get_recognizer()

    ds = load_dataset("xbgoose/dusha", split=args.split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.split}.jsonl"
    done = 0
    t0 = time.perf_counter()
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, item in enumerate(ds):
            if done >= args.n:
                break
            emo = str(item["emotion"]).lower().strip()
            if emo not in CLASSES:
                continue
            raw = item["audio"].get("bytes")
            if not raw:
                continue
            wav, _ = rec.load_audio(raw)
            if wav.size < 16000 * 0.3:
                continue
            res = asr.transcribe(wav, "ru")
            rec_obj = {
                "idx": idx,
                "emotion": emo,
                "text": res.text,
                "asr_ms": res.inference_ms,
            }
            f.write(json.dumps(rec_obj, ensure_ascii=False) + "\n")
            done += 1
            if done % 50 == 0:
                console.log(f"  {done}/{args.n}  посл.текст: «{res.text[:50]}»")

    console.log(f"[bold green]Готово[/] {done} строк за {time.perf_counter()-t0:.0f}s "
                f"-> {out_path}")


if __name__ == "__main__":
    main()
