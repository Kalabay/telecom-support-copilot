"""Multi-agent синтетический генератор диалогов оператор↔клиент."""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import app  # noqa: F401,E402

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
OUT_FILE = PROJECT_ROOT / "data" / "synthetic" / "dialogs.json"

SCENARIOS = [
    {
        "id": "no_internet_resolved",
        "issue": "diagnose_general",
        "trajectory": ["neutral", "neutral", "neutral", "positive"],
        "opening": "Алло, у меня не работает интернет, помогите разобраться.",
    },
    {
        "id": "no_internet_escalating",
        "issue": "diagnose_general",
        "trajectory": ["neutral", "angry", "angry"],
        "opening": "Третий день не работает интернет, что вы вообще делаете?",
    },
    {
        "id": "router_reboot_works",
        "issue": "router_reboot",
        "trajectory": ["neutral", "neutral", "positive"],
        "opening": "На моём роутере мигает красная лампочка, что это значит?",
    },
    {
        "id": "router_reboot_fails",
        "issue": "router_reboot",
        "trajectory": ["neutral", "angry", "angry"],
        "opening": "Уже пять раз перезагружал, ничего не помогает.",
    },
    {
        "id": "cable_damage",
        "issue": "cable_check",
        "trajectory": ["neutral", "sad", "neutral"],
        "opening": "Кот перегрыз кабель от роутера, что теперь делать?",
    },
    {
        "id": "outage_check",
        "issue": "outage_map",
        "trajectory": ["neutral", "neutral"],
        "opening": "Соседи говорят что у них тоже нет интернета, у вас авария?",
    },
    {
        "id": "planned_works",
        "issue": "planned_works",
        "trajectory": ["neutral", "angry"],
        "opening": "Интернет отключили без предупреждения! Это безобразие!",
    },
    {
        "id": "wifi_weak",
        "issue": "wifi_weak_signal",
        "trajectory": ["neutral", "neutral"],
        "opening": "Wi-Fi плохо ловит в дальней комнате, что посоветуете?",
    },
    {
        "id": "wifi_no_connect",
        "issue": "wifi_no_connection",
        "trajectory": ["neutral", "neutral"],
        "opening": "Телефон не подключается к Wi-Fi, пишет неверный пароль.",
    },
    {
        "id": "slow_speed",
        "issue": "slow_speed",
        "trajectory": ["neutral", "neutral", "angry"],
        "opening": "Интернет тормозит ужасно, скорость в два раза ниже тарифа.",
    },
    {
        "id": "mobile_apn",
        "issue": "apn_settings",
        "trajectory": ["neutral", "neutral"],
        "opening": "Купил новый телефон, мобильный интернет не работает.",
    },
    {
        "id": "tariff_limit",
        "issue": "tariff_limit",
        "trajectory": ["neutral", "neutral"],
        "opening": "Резко стал работать медленно, что случилось с интернетом?",
    },
    {
        "id": "balance_zero",
        "issue": "balance_check",
        "trajectory": ["neutral", "sad", "neutral"],
        "opening": "Интернет отключили, наверное оплата не прошла, что делать?",
    },
    {
        "id": "sim_replacement",
        "issue": "sim_replacement",
        "trajectory": ["neutral"],
        "opening": "Телефон не видит SIM-карту, можно её заменить?",
    },
    {
        "id": "pppoe_new_router",
        "issue": "pppoe_setup",
        "trajectory": ["neutral", "neutral"],
        "opening": "Купил новый роутер, не могу настроить интернет.",
    },
    {
        "id": "mac_binding",
        "issue": "mac_binding",
        "trajectory": ["neutral", "neutral", "angry"],
        "opening": "Старый роутер сломался, поставил новый — интернета нет.",
    },
    {
        "id": "factory_reset",
        "issue": "router_factory_reset",
        "trajectory": ["neutral", "neutral"],
        "opening": "Не могу войти в админку роутера, забыл пароль.",
    },
    {
        "id": "tech_visit_needed",
        "issue": "technician_visit",
        "trajectory": ["neutral", "neutral", "positive"],
        "opening": "Кабель повреждён в подъезде, можете прислать мастера?",
    },
    {
        "id": "iptv_broken",
        "issue": "iptv_no_signal",
        "trajectory": ["neutral", "angry"],
        "opening": "Телевизор показывает что нет сигнала, что случилось?",
    },
    {
        "id": "escalation_2nd_line",
        "issue": "escalation_l2",
        "trajectory": ["angry", "angry", "neutral"],
        "opening": "Я звоню уже третий раз, никто не может ничего сделать!",
    },
    {
        "id": "compensation_request",
        "issue": "compensation",
        "trajectory": ["angry", "angry", "neutral"],
        "opening": "Три дня без интернета, верните мне деньги за этот месяц!",
    },
    {
        "id": "threat_to_leave",
        "issue": "retention_offer",
        "trajectory": ["angry", "angry", "neutral"],
        "opening": "Всё, надоело, расторгайте договор, ухожу к Билайну!",
    },
    {
        "id": "polite_complaint",
        "issue": "compensation",
        "trajectory": ["sad", "sad", "neutral"],
        "opening": "Грустно, что после стольких лет с вами вот такие проблемы.",
    },
    {
        "id": "relocation",
        "issue": "relocation_setup",
        "trajectory": ["neutral", "neutral", "positive"],
        "opening": "Я переезжаю в другую квартиру, можно перенести интернет?",
    },
    {
        "id": "firmware_curious",
        "issue": "router_firmware",
        "trajectory": ["neutral"],
        "opening": "Стоит ли обновить прошивку роутера? Безопасно?",
    },
]

CLIENT_PROMPT_TMPL = """Ты — клиент российского телеком-оператора, который звонит в техподдержку.

Контекст этого звонка:
- проблема: {issue}
- твоё текущее эмоциональное состояние: {emotion}

Только что оператор сказал тебе:
«{operator_reply}»

Ответь оператору одной короткой репликой (1-2 предложения), как живой человек. \
Не выдумывай детали, которых не было раньше в разговоре. \
Эмоция должна быть {emotion} (не пиши слово 'эмоция', просто говори с этим тоном). \
Без кавычек, без markdown."""


EMOTION_HINTS_CLIENT = {
    "neutral": "спокойно, по делу",
    "angry": "раздражённо, с возмущением, можешь использовать слова 'надоело', 'безобразие'",
    "positive": "благодарно, удовлетворённо",
    "sad": "расстроенно, с тревогой",
}


def _emotion_state(label: str, escal: bool = False) -> dict:
    presets = {
        "neutral": {"label": "neutral", "confidence": 0.92, "arousal": 0.30, "valence": 0.0},
        "angry": {"label": "angry", "confidence": 0.78, "arousal": 0.85, "valence": -0.70},
        "positive": {"label": "positive", "confidence": 0.85, "arousal": 0.55, "valence": 0.70},
        "sad": {"label": "sad", "confidence": 0.75, "arousal": 0.30, "valence": -0.55},
        "other": {"label": "other", "confidence": 0.70, "arousal": 0.50, "valence": 0.0},
    }
    s = presets.get(label, presets["other"])
    return {**s, "escalation_risk": escal or (label == "angry")}


def _operator_reply(
    client: httpx.Client,
    transcript: list[str],
    emotion_label: str,
    sources: list[dict],
) -> str:
    """Сгенерировать одну реплику оператора через бэкендный /api/generate."""
    r = client.post(
        f"{API_URL}/api/generate",
        json={
            "transcript": transcript,
            "emotion": _emotion_state(emotion_label, escal=(emotion_label == "angry")),
            "sources": sources,
            "max_tokens": 180,
        },
        timeout=120.0,
    )
    r.raise_for_status()
    out = r.json()
    sug = out["suggestions"]
    return (sug[0] if sug else out["raw_completion"]).strip()


def _client_reply(
    client: httpx.Client,
    issue: str,
    emotion_label: str,
    operator_last: str,
) -> str:
    """Сгенерировать одну реплику клиента (тоже через тот же /api/generate)."""
    user_prompt = CLIENT_PROMPT_TMPL.format(
        issue=issue,
        emotion=EMOTION_HINTS_CLIENT.get(emotion_label, "нейтрально"),
        operator_reply=operator_last,
    )
    r = client.post(
        f"{API_URL}/api/generate",
        json={
            "transcript": [user_prompt],
            "emotion": _emotion_state(emotion_label),
            "sources": [],
            "max_tokens": 100,
        },
        timeout=120.0,
    )
    r.raise_for_status()
    out = r.json()
    text = (out["suggestions"][0] if out["suggestions"] else out["raw_completion"]).strip()
    if text.startswith(("1.", "2.")):
        text = text[2:].lstrip(" .-—")
    return text


def _retrieve_sources(client: httpx.Client, query: str) -> list[dict]:
    return []


def generate_dialogs(n_per_scenario: int = 2, seed: int = 17) -> list[dict]:
    rng = random.Random(seed)
    dialogs: list[dict] = []
    with httpx.Client() as http:
        for sc in SCENARIOS:
            for variant in range(n_per_scenario):
                turns: list[dict] = []
                transcript: list[str] = []
                customer_opening = sc["opening"]
                transcript.append(customer_opening)
                turns.append(
                    {
                        "speaker": "customer",
                        "text": customer_opening,
                        "emotion": sc["trajectory"][0],
                    }
                )

                for i, emo in enumerate(sc["trajectory"]):
                    print(f"  [{sc['id']}-{variant}] turn {i+1}/{len(sc['trajectory'])}…", flush=True)
                    op_reply = _operator_reply(
                        http,
                        transcript=transcript,
                        emotion_label=emo,
                        sources=_retrieve_sources(http, transcript[-1]),
                    )
                    turns.append({"speaker": "operator", "text": op_reply})
                    transcript.append(op_reply)

                    if i + 1 < len(sc["trajectory"]):
                        next_emo = sc["trajectory"][i + 1]
                        client_text = _client_reply(
                            http,
                            issue=sc["issue"],
                            emotion_label=next_emo,
                            operator_last=op_reply,
                        )
                        transcript.append(client_text)
                        turns.append(
                            {
                                "speaker": "customer",
                                "text": client_text,
                                "emotion": next_emo,
                            }
                        )

                dialog = {
                    "dialog_id": f"{sc['id']}_v{variant}",
                    "scenario": sc["id"],
                    "expected_intent": sc["issue"],
                    "trajectory": sc["trajectory"],
                    "turns": turns,
                }
                dialogs.append(dialog)

    rng.shuffle(dialogs)
    return dialogs


def main() -> None:
    print(f"Generating dialogs via {API_URL}/api/generate ...")
    r = httpx.get(f"{API_URL}/api/health", timeout=5)
    r.raise_for_status()
    print(f"Backend OK: {r.json().get('app')}")

    dialogs = generate_dialogs(n_per_scenario=2)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(
        json.dumps(dialogs, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    n_turns = sum(len(d["turns"]) for d in dialogs)
    n_customer = sum(1 for d in dialogs for t in d["turns"] if t["speaker"] == "customer")
    print(f"\nSaved {len(dialogs)} dialogs ({n_turns} turns, {n_customer} customer) -> {OUT_FILE}")


if __name__ == "__main__":
    main()
