"""Генерация аудио КЛИЕНТСКИХ реплик для E2E-бенчмарка через ElevenLabs v3."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.elevenlabs.io/v1/text-to-speech"
KEY = os.environ.get("ELEVENLABS_API_KEY", "")

EMOTION = {
    "angry": {"tags": "[furious] [shouting] ", "stability": 0.30},
    "neutral": {"tags": "", "stability": 0.40},
    "positive": {"tags": "[happy] [excited] ", "stability": 0.35},
    "sad": {"tags": "[sad] [crying] ", "stability": 0.70},
}

VOICE_SAD = ("Marina Soft RU", "ymDCYd8puC7gYjxIamPt")
VOICES_ANGRY = [("Nikolay Confident RU", "3EuKHIEZbSzrHGNmdYsx"),
                ("Sergey Deep RU", "XuEV9VY3VUASYgJVNBh0")]
VOICES_POS = [("Larisa Actrisa RU", "AB9XsbSA4eLG12t2myjN"),
              ("Nadia Energetic RU", "gedzfqL7OGdPbwm0ynTP")]
VOICES_NEU = [("Maxim Calm RU", "HcaxAsrhw4ByUo4CBCBN"),
              ("Georgy Calm RU", "MYw0upsxdtxs1n97djly")]


def pick_dialogue_voice(did: str, emotions: set) -> tuple[str, str]:
    if "sad" in emotions:
        return VOICE_SAD
    if "angry" in emotions:
        return VOICES_ANGRY[stable_pick(did, len(VOICES_ANGRY))]
    if "positive" in emotions:
        return VOICES_POS[stable_pick(did, len(VOICES_POS))]
    return VOICES_NEU[stable_pick(did, len(VOICES_NEU))]


def stable_pick(seed: str, n: int) -> int:
    return sum(ord(c) for c in seed) % n


def tts(voice_id: str, text: str, stability: float) -> bytes:
    body = json.dumps({
        "text": text, "model_id": "eleven_v3",
        "voice_settings": {"stability": stability},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{API}/{voice_id}", data=body, method="POST",
        headers={"xi-api-key": KEY, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main() -> None:
    dialogues = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    outdir = Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for d in dialogues:
        did = d["dialogue_id"]
        emos = {t["emotion"] for t in d["turns"] if t.get("role") == "client"}
        vname, vid = pick_dialogue_voice(did, emos)
        for t in d["turns"]:
            if t.get("role") != "client":
                continue
            emo = t["emotion"]
            rec = EMOTION[emo]
            clean = t["text"]
            send = rec["tags"] + clean
            fname = f"{did}_t{t['idx']}.mp3"
            ok, err = True, ""
            try:
                (outdir / fname).write_bytes(tts(vid, send, rec["stability"]))
            except urllib.error.HTTPError as e:
                ok, err = False, f"{e.code} {e.read()[:200]!r}"
            except Exception as e:  # noqa: BLE001
                ok, err = False, str(e)[:200]
            manifest.append({
                "dialogue_id": did, "company": d.get("company", ""), "turn_idx": t["idx"],
                "emotion": emo, "voice_name": vname, "voice_id": vid,
                "stability": rec["stability"], "send_text": send, "clean_text": clean,
                "gold_doc_ids": t.get("gold_doc_ids", []), "file": fname,
                "ok": ok, "error": err,
            })
            print(("ok  " if ok else "ERR ") + f"{fname} [{emo}] {vname}"
                  + ("" if ok else " :: " + err), flush=True)
            time.sleep(0.3)
    (outdir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    good = sum(1 for m in manifest if m["ok"])
    print(f"\nwrote {good}/{len(manifest)} audio + manifest.json to {outdir}")


if __name__ == "__main__":
    if not KEY:
        print("set ELEVENLABS_API_KEY")
        sys.exit(1)
    main()
