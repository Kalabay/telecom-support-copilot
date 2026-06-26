"""Один пример диалога клиент↔оператор через ElevenLabs v3 Text-to-Dialogue API."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests

BASE = "https://api.elevenlabs.io"
OUT = Path(r"K:\.caches\el_dialogue.mp3")

DIALOGUE = [
    ("client", "[раздражённо] Да сколько можно?! Третий день нет сети вообще, "
               "ни позвонить, ни в интернет зайти! На работе из-за вас проблемы!"),
    ("operator", "[спокойно, с эмпатией] Понимаю, как это раздражает — три дня без "
                 "связи это точно не нормально. Сейчас проверю, нет ли аварии у вас "
                 "по адресу, и сразу скажу, что делать."),
    ("client", "[устало] Ну давайте уже, сколько можно ждать."),
    ("operator", "[уверенно] Вижу плановых работ в вашем районе нет — оформляю "
                 "срочную заявку, мастер приедет сегодня. За простой сделаем перерасчёт."),
]


def headers(key: str) -> dict:
    return {"xi-api-key": key, "Content-Type": "application/json"}


def list_voices(key: str) -> None:
    r = requests.get(f"{BASE}/v1/voices", headers={"xi-api-key": key}, timeout=30)
    r.raise_for_status()
    voices = r.json().get("voices", [])
    print(f"Доступно голосов: {len(voices)}\n")
    for v in voices:
        labels = v.get("labels", {})
        print(f"  {v['voice_id']}  | {v['name']:20s} | "
              f"{labels.get('gender','?')}, {labels.get('language', labels.get('accent','?'))}")


def make_dialogue(key: str, client_voice: str, operator_voice: str) -> None:
    voice_of = {"client": client_voice, "operator": operator_voice}
    inputs = [{"text": text, "voice_id": voice_of[who]} for who, text in DIALOGUE]
    total_chars = sum(len(t) for _, t in DIALOGUE)
    print(f"Реплик: {len(inputs)}, символов всего: {total_chars} "
          f"(≈{total_chars} кредитов по сайтовому тарифу, через API меньше)\n", flush=True)

    body = {
        "inputs": inputs,
        "model_id": "eleven_v3",
        "settings": {"stability": 0.5},
    }
    r = requests.post(f"{BASE}/v1/text-to-dialogue", headers=headers(key),
                      json=body, timeout=120)
    if r.status_code != 200:
        print(f"ОШИБКА {r.status_code}: {r.text[:500]}", flush=True)
        sys.exit(1)
    OUT.write_bytes(r.content)
    print(f"saved -> {OUT}  ({len(r.content)} байт)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=os.environ.get("ELEVENLABS_API_KEY", ""))
    ap.add_argument("--list", action="store_true", help="показать голоса и выйти")
    ap.add_argument("--client", help="voice_id для клиента")
    ap.add_argument("--operator", help="voice_id для оператора")
    args = ap.parse_args()

    if not args.key:
        print("Нет ключа. Передай --key ВАШ_КЛЮЧ или set ELEVENLABS_API_KEY=...")
        sys.exit(1)
    if args.list:
        list_voices(args.key)
        return
    if not args.client or not args.operator:
        print("Укажи --client VOICE_ID и --operator VOICE_ID "
              "(сначала --list чтобы узнать id)")
        sys.exit(1)
    make_dialogue(args.key, args.client, args.operator)


if __name__ == "__main__":
    main()
