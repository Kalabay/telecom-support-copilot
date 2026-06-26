"""Фаза 2: СКВОЗНОЙ прогон полного пайплайна на синтетическом аудио."""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import app  # noqa: F401,E402

import sentence_transformers  # noqa: F401,E402
import torch  # noqa: F401,E402

import soundfile as sf  # noqa: E402

R = PROJECT_ROOT / "eval" / "results"
MAN = Path(r"K:\.caches\pipe_audio\manifest.jsonl")


def norm(t: str) -> list[str]:
    t = t.lower().replace("ё", "е")
    return [w for w in re.sub(r"[^\wа-я ]+", " ", t).split() if w]


def wer(ref, hyp) -> float:
    n, m = len(ref), len(hyp)
    if n == 0:
        return 1.0 if m else 0.0
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (0 if ref[i-1] == hyp[j-1] else 1))
            prev = cur
    return dp[m] / n


def main() -> None:
    rows = [json.loads(l) for l in MAN.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"кейсов: {len(rows)}", flush=True)

    from app.pipeline.asr import get_asr
    from app.pipeline.ser import get_recognizer
    from app.pipeline.rag import get_retriever
    from app.pipeline.llm import get_generator
    asr = get_asr(); asr._ensure_loaded()
    ser = get_recognizer()
    rag = get_retriever()
    gen = get_generator(); gen._ensure_loaded()
    print("все модели загружены", flush=True)

    out = []
    t0 = time.perf_counter()
    for i, r in enumerate(rows, 1):
        wav, sr = sf.read(r["path"], dtype="float32")
        if sr != 16000:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        wav = wav.astype("float32")
        asr_text = asr.transcribe(wav, "ru").text.strip()
        ser_res = ser.predict(wav)
        ser_emo = ser_res.state.label.value
        rag_audio = rag.search(asr_text or ".", k=3, company=r["company"])
        top_audio = rag_audio.sources[0].doc_id if rag_audio.sources else ""
        sug_audio = gen.generate(
            [{"speaker": "customer", "text": asr_text or r["text"]}],
            ser_res.state, rag_audio.sources, max_tokens=200).suggestions
        rag_ref = rag.search(r["text"], k=3, company=r["company"])
        top_ref = rag_ref.sources[0].doc_id if rag_ref.sources else ""
        m = {
            "asr_wer": round(wer(norm(r["text"]), norm(asr_text)), 3),
            "ser_pred": ser_emo, "ser_true": r["emotion"],
            "ser_correct": int(ser_emo == r["emotion"]),
            "top_audio": top_audio, "top_ref": top_ref,
            "rag_same_top": int(top_audio == top_ref and top_audio != ""),
        }
        out.append({"id": r["id"], "company": r["company"], "emotion": r["emotion"],
                    "ref_text": r["text"], "asr_text": asr_text,
                    "metrics": m, "sug_audio": sug_audio})
        if i % 5 == 0:
            print(f"  {i}/{len(rows)}  {time.perf_counter()-t0:.0f}s", flush=True)

    n = len(out)
    summ = {
        "n": n,
        "asr_wer_mean": round(sum(o["metrics"]["asr_wer"] for o in out) / n, 3),
        "ser_accuracy": round(sum(o["metrics"]["ser_correct"] for o in out) / n, 3),
        "rag_same_top_rate": round(sum(o["metrics"]["rag_same_top"] for o in out) / n, 3),
        "ser_confusion": dict(Counter(f"{o['metrics']['ser_true']}->{o['metrics']['ser_pred']}"
                                       for o in out)),
    }
    print("\n=== СКВОЗНОЙ ПАЙПЛАЙН (аудио -> ASR -> SER -> RAG -> LLM) ===")
    print(f"  кейсов: {n}")
    print(f"  ASR WER (TTS+телефон): {summ['asr_wer_mean']}")
    print(f"  SER точность (TTS-голос vs истинная эмоция): {summ['ser_accuracy']}")
    print(f"  RAG top-1 совпал с эталоном: {summ['rag_same_top_rate']}")
    print("  SER путаница (истина->предсказание):")
    for k, v in sorted(summ["ser_confusion"].items(), key=lambda x: -x[1]):
        print(f"    {k}: {v}")
    (R / "pipe_full.json").write_text(
        json.dumps({"summary": summ, "rows": out}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\nsaved -> {R / 'pipe_full.json'}")


if __name__ == "__main__":
    main()
