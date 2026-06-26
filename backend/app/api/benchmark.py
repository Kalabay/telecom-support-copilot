"""REST: данные синтетического E2E-бенчмарка для просмотра в UI."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIO_ROOT = PROJECT_ROOT / "eval" / "audio"
AUDIO_E2E = AUDIO_ROOT / "e2e"
DIALOGUES = PROJECT_ROOT / "eval" / "e2e_dialogues.json"
MANIFEST = AUDIO_E2E / "manifest.json"
E2E_REAL = PROJECT_ROOT / "eval" / "results" / "e2e_real_result.json"

_SAFE = re.compile(r"^[\w.\-]+\.mp3$")


@lru_cache(maxsize=1)
def _real_dialogues() -> dict[str, list[dict]]:
    """Диалоги реальных операторов связи из e2e_real_result (без аудио), сгруппированные по диалогу."""
    if not E2E_REAL.exists():
        return {}
    rows = json.loads(E2E_REAL.read_text(encoding="utf-8")).get("turns", [])
    by: dict[str, list[dict]] = {}
    for t in rows:
        by.setdefault(t["dialogue_id"], []).append(t)
    return {did: sorted(ts, key=lambda x: x.get("turn_idx", 0)) for did, ts in by.items()}


@router.get("/benchmark/dialogues")
def dialogues() -> dict:
    if not DIALOGUES.exists():
        return {"dialogues": [], "count": 0, "client_audio": 0}
    data = json.loads(DIALOGUES.read_text(encoding="utf-8"))
    man = {}
    if MANIFEST.exists():
        for m in json.loads(MANIFEST.read_text(encoding="utf-8")):
            man[(m["dialogue_id"], m["turn_idx"])] = m

    out, n_audio = [], 0
    for d in data:
        did = d["dialogue_id"]
        turns = []
        for t in d["turns"]:
            if t.get("role") == "client":
                mm = man.get((did, t["idx"]))
                has_audio = bool(mm and mm.get("ok") and (AUDIO_E2E / mm["file"]).exists())
                if has_audio:
                    n_audio += 1
                turns.append({
                    "role": "client", "idx": t["idx"],
                    "text": (mm or {}).get("clean_text") or t["text"],
                    "emotion": t["emotion"],
                    "gold_doc_ids": t.get("gold_doc_ids", []),
                    "audio_url": f"/api/benchmark/audio/{mm['file']}" if has_audio else None,
                    "voice_name": (mm or {}).get("voice_name"),
                    "stability": (mm or {}).get("stability"),
                })
            else:
                turns.append({"role": "operator", "idx": t["idx"], "text": t["ideal_text"]})
        out.append({
            "dialogue_id": did, "company": d.get("company", ""),
            "scenario": d.get("scenario", ""), "turns": turns,
        })
    for rdid, rows in _real_dialogues().items():
        turns = []
        for t in rows:
            turns.append({
                "role": "client", "idx": t["turn_idx"],
                "text": t.get("clean_text") or t.get("asr_text", ""),
                "emotion": t.get("emotion_true"),
                "gold_doc_ids": t.get("gold_doc_ids", []),
                "audio_url": None, "voice_name": None, "stability": None,
            })
            turns.append({"role": "operator", "idx": t["turn_idx"], "text": t.get("ideal_text", "")})
        out.append({
            "dialogue_id": rdid, "company": rows[0].get("company", ""),
            "scenario": "реальный оператор связи", "turns": turns,
        })
    return {"dialogues": out, "count": len(out), "client_audio": n_audio}


@router.get("/benchmark/samples")
def samples() -> dict:
    """Одиночные реплики SER-датасета из batch*_gen.json / batch*_manifest.json."""
    out = []
    for jf in sorted(AUDIO_ROOT.glob("batch*.json")):
        try:
            rows = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for e in rows:
            file = e.get("file") or f"{e.get('id', '')}.mp3"
            if not file or not (AUDIO_ROOT / file).exists():
                continue
            out.append({
                "id": e.get("id"), "batch": e.get("batch"),
                "emotion": e.get("emotion"),
                "voice_name": e.get("voice_name") or e.get("voice"),
                "stability": e.get("stability"),
                "clean_text": e.get("clean_text", ""),
                "send_text": e.get("send_text", ""),
                "ser_pred": e.get("ser_pred"),
                "ser_ok": e.get("ser_ok"),
                "audio_url": f"/api/benchmark/audio/{file}",
            })
    # клипы из диалогов тоже доступны как одиночные
    if MANIFEST.exists():
        for m in json.loads(MANIFEST.read_text(encoding="utf-8")):
            file = m.get("file")
            if not file or not m.get("ok") or not (AUDIO_E2E / file).exists():
                continue
            out.append({
                "id": f"{m['dialogue_id']}_t{m['turn_idx']}", "batch": None,
                "emotion": m.get("emotion"),
                "voice_name": m.get("voice_name"), "stability": m.get("stability"),
                "clean_text": m.get("clean_text", ""), "send_text": m.get("send_text", ""),
                "ser_pred": None, "ser_ok": None,
                "audio_url": f"/api/benchmark/audio/{file}",
            })
    by_emotion: dict[str, int] = {}
    for s in out:
        by_emotion[s["emotion"]] = by_emotion.get(s["emotion"], 0) + 1
    return {"samples": out, "count": len(out), "by_emotion": by_emotion}


@router.get("/benchmark/simulate-turn")
def simulate_turn(dialogue_id: str, turn_idx: int) -> dict:
    """Прогнать одну клиентскую реплику диалога через ВЕСЬ конвейер (для симуляции в UI)."""
    from app.pipeline.asr import decode_audio_blob, get_asr
    from app.pipeline.llm import get_generator
    from app.pipeline.rag import get_retriever
    from app.pipeline.ser import get_recognizer

    man = {}
    if MANIFEST.exists():
        for m in json.loads(MANIFEST.read_text(encoding="utf-8")):
            man[(m["dialogue_id"], m["turn_idx"])] = m
    vek = None
    if DIALOGUES.exists():
        vek = next((d for d in json.loads(DIALOGUES.read_text(encoding="utf-8"))
                    if d["dialogue_id"] == dialogue_id), None)

    transcript, prev_customer = [], ""
    target_text, target_emo = "", "neutral"
    if vek:
        company = vek.get("company", "vektor")
        for t in vek["turns"]:
            if t["role"] == "client":
                txt = (man.get((dialogue_id, t["idx"])) or {}).get("clean_text") or t.get("text", "")
                if t["idx"] == turn_idx:
                    target_text, target_emo = txt, t.get("emotion", "neutral")
                    break
                transcript.append({"speaker": "customer", "text": txt})
                prev_customer = txt
            else:
                transcript.append({"speaker": "operator", "text": t.get("ideal_text", "")})
    else:
        rows = _real_dialogues().get(dialogue_id)
        if not rows:
            raise HTTPException(404, "dialogue not found")
        company = rows[0].get("company", "")
        for t in rows:
            if t["turn_idx"] == turn_idx:
                target_text, target_emo = t.get("clean_text", ""), t.get("emotion_true", "neutral")
                break
            transcript.append({"speaker": "customer", "text": t.get("clean_text", "")})
            transcript.append({"speaker": "operator", "text": t.get("ideal_text", "")})
            prev_customer = t.get("clean_text", "")

    mm = man.get((dialogue_id, turn_idx))
    has_audio = bool(mm and mm.get("ok") and (AUDIO_E2E / mm["file"]).exists())
    if has_audio:
        wav = decode_audio_blob((AUDIO_E2E / mm["file"]).read_bytes())
        asr = get_asr().transcribe(wav, "ru")
        asr_text = asr.text.strip()
        ser = get_recognizer().predict(wav)
        emotion = ser.state
        asr_ms, ser_ms = asr.inference_ms, ser.inference_ms
    else:
        from app.models.schemas import Emotion, EmotionState
        from app.pipeline.ser import _VA_BY_LABEL
        asr_text = target_text
        emo_enum = {"angry": Emotion.ANGRY, "sad": Emotion.SAD, "positive": Emotion.POSITIVE,
                    "neutral": Emotion.NEUTRAL}.get(target_emo, Emotion.NEUTRAL)
        ar, val = _VA_BY_LABEL[emo_enum]
        emotion = EmotionState(label=emo_enum, confidence=0.85, arousal=ar, valence=val,
                               escalation_risk=(emo_enum is Emotion.ANGRY))
        asr_ms, ser_ms = 0, 0

    transcript.append({"speaker": "customer", "text": asr_text})
    rag_query = f"{prev_customer} {asr_text}".strip() if prev_customer else asr_text
    rag = get_retriever().search(rag_query, k=3, company=company)
    gen = get_generator().generate(transcript, emotion, rag.sources, max_tokens=320)

    return {
        "dialogue_id": dialogue_id, "turn_idx": turn_idx, "company": company,
        "asr_text": asr_text,
        "emotion": {
            "label": emotion.label.value, "confidence": round(emotion.confidence, 2),
            "arousal": round(emotion.arousal, 2), "escalation": emotion.escalation_risk,
        },
        "suggestions": gen.suggestions,
        "sources": [{"doc_id": s.doc_id, "title": s.title, "snippet": s.snippet[:200]}
                    for s in rag.sources],
        "timings_ms": {"asr": asr_ms, "ser": ser_ms, "rag": rag.total_ms, "llm": gen.total_ms},
    }


@router.get("/benchmark/recognize")
def recognize(file: str) -> dict:
    """Лёгкая проверка одного клипа: только распознавание речи и эмоции, без RAG/LLM."""
    if not _SAFE.match(file):
        raise HTTPException(400, "bad filename")
    path = next((base / file for base in (AUDIO_E2E, AUDIO_ROOT) if (base / file).exists()), None)
    if path is None:
        raise HTTPException(404, "audio not found")
    from app.pipeline.asr import decode_audio_blob, get_asr
    from app.pipeline.ser import get_recognizer

    wav = decode_audio_blob(path.read_bytes())
    asr = get_asr().transcribe(wav, "ru")
    ser = get_recognizer().predict(wav)
    emo = ser.state
    return {
        "asr_text": asr.text.strip(),
        "emotion": {
            "label": emo.label.value, "confidence": round(emo.confidence, 2),
            "arousal": round(emo.arousal, 2), "escalation": emo.escalation_risk,
        },
        "timings_ms": {"asr": asr.inference_ms, "ser": ser.inference_ms},
    }


@router.get("/benchmark/audio/{filename}")
def audio(filename: str) -> FileResponse:
    if not _SAFE.match(filename):
        raise HTTPException(400, "bad filename")
    for base in (AUDIO_E2E, AUDIO_ROOT):
        path = base / filename
        if path.exists():
            return FileResponse(str(path), media_type="audio/mpeg")
    raise HTTPException(404, "audio not found")
