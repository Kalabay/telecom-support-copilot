"""E2E-прогон многоходовых диалогов: аудио клиента -> ASR -> SER -> RAG -> LLM."""
from __future__ import annotations

import argparse
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

DIALOGUES = PROJECT_ROOT / "eval" / "e2e_dialogues.json"
AUDIO_DIR = PROJECT_ROOT / "eval" / "audio" / "e2e"
RESULTS = PROJECT_ROOT / "eval" / "results" / "e2e_dialogues_result.json"

PROMISE = ["бонус", "подар", "компенс", "верну", "вернём", "вернем", "скидк",
           "процент", "нулев", "в подарок"]
HEDGE = ["проверю", "проверим", "уточню", "уточним", "если", "запрош", "посмотрю",
         "детализац", "разберёмся", "разберемся", "проверять"]


def norm(t: str) -> list[str]:
    t = t.lower().replace("ё", "е")
    return [w for w in re.sub(r"[^\wа-я ]+", " ", t).split() if w]


def wer(ref: list[str], hyp: list[str]) -> float:
    n, m = len(ref), len(hyp)
    if n == 0:
        return 1.0 if m else 0.0
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (0 if ref[i - 1] == hyp[j - 1] else 1))
            prev = cur
    return dp[m] / n


def nums(t: str) -> set[str]:
    return set(re.findall(r"\d+", t))


def hallucination(suggestion: str, kb_text: str) -> tuple[bool, str]:
    kb_n = nums(kb_text)
    s = suggestion.lower()
    invented = [x for x in nums(s) if x not in kb_n and int(x) > 5]
    if invented:
        return True, f"числа вне KB: {sorted(invented, key=int)}"
    if any(p in s for p in PROMISE) and not any(p in kb_text.lower() for p in PROMISE) \
            and not any(h in s for h in HEDGE):
        return True, "обещание выгоды без опоры в KB"
    return False, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogues", default=str(DIALOGUES))
    ap.add_argument("--audio", default=str(AUDIO_DIR))
    ap.add_argument("--results", default=str(RESULTS))
    args = ap.parse_args()
    audio_dir = Path(args.audio)
    results_path = Path(args.results)

    dialogues = json.loads(Path(args.dialogues).read_text(encoding="utf-8"))
    manifest = json.loads((audio_dir / "manifest.json").read_text(encoding="utf-8"))
    audio = {(m["dialogue_id"], m["turn_idx"]): m for m in manifest if m.get("ok")}
    print(f"диалогов: {len(dialogues)}, озвучено реплик: {len(audio)}", flush=True)

    from app.pipeline.asr import decode_audio_blob, get_asr
    from app.pipeline.ser import get_recognizer
    from app.pipeline.rag import get_retriever
    from app.pipeline.llm import get_generator
    asr = get_asr(); asr._ensure_loaded()
    ser = get_recognizer()
    rag = get_retriever()
    gen = get_generator(); gen._ensure_loaded()
    print("модели загружены", flush=True)

    turn_rows = []
    t0 = time.perf_counter()
    for d in dialogues:
        did, company = d["dialogue_id"], d["company"]
        transcript: list[dict] = []
        turns = d["turns"]
        for i, turn in enumerate(turns):
            if turn["role"] == "operator":
                transcript.append({"speaker": "operator", "text": turn["ideal_text"]})
                continue
            key = (did, turn["idx"])
            man = audio.get(key)
            clean = turn["text"]
            if man and (audio_dir / man["file"]).exists():
                wav = decode_audio_blob((audio_dir / man["file"]).read_bytes())
                asr_text = asr.transcribe(wav, "ru").text.strip()
                ser_state = ser.predict(wav).state
            else:
                asr_text, wav = clean, None
                ser_state = ser.predict
            ser_emo = ser_state.label.value if wav is not None else "(no-audio)"

            transcript.append({"speaker": "customer", "text": asr_text})
            prev_cust = next((t["text"] for t in reversed(transcript[:-1])
                              if t["speaker"] == "customer"), "")
            query = f"{prev_cust} {asr_text}".strip()
            rag_res = rag.search(query, k=3, company=company)
            top_ids = [s.doc_id for s in rag_res.sources]
            gold = turn.get("gold_doc_ids", [])
            rag_hit = int(any(g in top_ids for g in gold))

            emotion_for_llm = ser_state if wav is not None else ser.predict(
                __import__("numpy").zeros(16000, dtype="float32")).state
            sug = gen.generate(list(transcript), emotion_for_llm, rag_res.sources,
                               max_tokens=200).suggestions
            best = sug[0] if sug else ""
            kb_text = " ".join(s.snippet for s in rag_res.sources)
            halluc, why = hallucination(best, kb_text)

            ideal = ""
            if i + 1 < len(turns) and turns[i + 1]["role"] == "operator":
                ideal = turns[i + 1]["ideal_text"]

            turn_rows.append({
                "dialogue_id": did, "company": company, "turn_idx": turn["idx"],
                "emotion_true": turn["emotion"],
                "emotion_pred": ser_emo,
                "ser_ok": int(ser_emo == turn["emotion"]),
                "asr_text": asr_text, "clean_text": clean,
                "wer": round(wer(norm(clean), norm(asr_text)), 3) if wav is not None else None,
                "gold_doc_ids": gold, "rag_top": top_ids, "rag_hit": rag_hit,
                "suggestion": best, "all_suggestions": sug, "ideal_text": ideal,
                "hallucination": halluc, "halluc_why": why,
            })
        print(f"  {did} ({company}) ✓  {time.perf_counter() - t0:.0f}s", flush=True)

    with_audio = [r for r in turn_rows if r["wer"] is not None]
    n = len(with_audio)
    summ = {
        "n_dialogues": len(dialogues),
        "n_client_turns": len(turn_rows),
        "n_with_audio": n,
        "ser_accuracy": round(sum(r["ser_ok"] for r in with_audio) / n, 3) if n else None,
        "asr_wer_mean": round(sum(r["wer"] for r in with_audio) / n, 3) if n else None,
        "rag_hit_rate": round(sum(r["rag_hit"] for r in turn_rows) / len(turn_rows), 3),
        "hallucination_rate": round(sum(r["hallucination"] for r in turn_rows) / len(turn_rows), 3),
        "ser_confusion": dict(Counter(f"{r['emotion_true']}->{r['emotion_pred']}"
                                      for r in with_audio)),
        "ser_per_emotion": {},
    }
    for emo in ("angry", "sad", "neutral", "positive"):
        sub = [r for r in with_audio if r["emotion_true"] == emo]
        if sub:
            summ["ser_per_emotion"][emo] = {
                "n": len(sub), "acc": round(sum(r["ser_ok"] for r in sub) / len(sub), 3)}

    excl = PROJECT_ROOT / "eval" / "ser_exclude.json"
    exset = set()
    if excl.exists():
        for e in json.loads(excl.read_text(encoding="utf-8")):
            exset.add((e["dialogue_id"], e["turn_idx"]))

    def _merge(e: str) -> str:
        return "calm" if e in ("neutral", "positive") else e

    def _acc(rows, merge: bool):
        if not rows:
            return None
        f = _merge if merge else (lambda x: x)
        return round(sum(1 for r in rows if f(r["emotion_pred"]) == f(r["emotion_true"])) / len(rows), 3)

    cur = [r for r in with_audio if (r["dialogue_id"], r["turn_idx"]) not in exset]
    summ["ser_acc_4class_curated"] = _acc(cur, False)
    summ["ser_acc_3class"] = _acc(with_audio, True)
    summ["ser_acc_3class_curated"] = _acc(cur, True)
    summ["n_excluded_false_angry"] = len(with_audio) - len(cur)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps({"summary": summ, "turns": turn_rows},
                                       ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== E2E ДИАЛОГИ (аудио -> ASR -> SER -> RAG -> LLM) ===")
    print(f"  диалогов: {summ['n_dialogues']}, клиентских реплик: {summ['n_client_turns']} "
          f"(с аудио: {n})")
    print(f"  SER 4 класса:                  raw {summ['ser_accuracy']} | "
          f"curated {summ['ser_acc_4class_curated']}")
    print(f"  SER 3 класса (neutral+positive): raw {summ['ser_acc_3class']} | "
          f"curated {summ['ser_acc_3class_curated']}  "
          f"(исключено ложного гнева: {summ['n_excluded_false_angry']})")
    print(f"  ASR WER: {summ['asr_wer_mean']}")
    print(f"  RAG hit@3 (gold в топе): {summ['rag_hit_rate']}")
    print(f"  Галлюцинации в подсказках: {summ['hallucination_rate']}")
    print(f"  SER по эмоциям: {summ['ser_per_emotion']}")
    print(f"\nsaved -> {results_path}")


if __name__ == "__main__":
    main()
