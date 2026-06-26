"""Абляция: влияет ли блок эмоции в промпте на ответы LLM."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import app  # noqa: F401,E402

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

API = "http://127.0.0.1:8000"
OUT = PROJECT_ROOT / "eval" / "results" / "ablation_emotion.json"

EMPATHY = ["извин", "прощени", "понимаю", "сожале", "к сожалению", "неприят",
           "сочувств", "приношу", "жаль"]
ACTION = ["компенс", "приоритет", "проверю", "проверим", "оформлю", "оформим",
          "верн", "перезагруз", "техник", "сейчас", "сразу"]

TESTSET = [
    ("angry",   "Я плачу вам каждый месяц, а интернета нет уже третий день!"),
    ("angry",   "Сколько можно?! Это просто безобразие, я в бешенстве!"),
    ("angry",   "Если сегодня не почините, расторгаю договор и ухожу к Билайну!"),
    ("angry",   "Вы издеваетесь? Звоню четвёртый раз, никто не помогает!"),
    ("angry",   "Верните мне деньги за этот месяц немедленно!"),
    ("angry",   "Достали ваши отписки, дайте нормального специалиста!"),
    ("sad",     "Так обидно, я с вами десять лет, а тут такие проблемы..."),
    ("sad",     "У меня дома ребёнок на удалёнке, а интернета нет, я в отчаянии."),
    ("sad",     "Я уже не знаю, что делать, ничего не помогает."),
    ("sad",     "Грустно, что приходится столько раз звонить из-за одного и того же."),
    ("neutral", "Здравствуйте, у меня не работает интернет, подскажите что делать."),
    ("neutral", "На роутере горит красная лампочка, это нормально?"),
    ("neutral", "Хочу уточнить, нет ли в моём районе плановых работ."),
    ("neutral", "Подскажите, как настроить интернет на новом роутере."),
    ("neutral", "Можно ли перенести интернет при переезде в другую квартиру?"),
]

EMO_PRESETS = {
    "angry":   {"label": "angry", "confidence": 0.82, "arousal": 0.85, "valence": -0.7,
                "escalation_risk": True},
    "sad":     {"label": "sad", "confidence": 0.76, "arousal": 0.30, "valence": -0.55,
                "escalation_risk": False},
    "neutral": {"label": "neutral", "confidence": 0.92, "arousal": 0.30, "valence": 0.0,
                "escalation_risk": False},
}


def _has(markers: list[str], text: str) -> bool:
    t = text.lower()
    return any(m in t for m in markers)


def _gen(client: httpx.Client, text: str, emo: dict, use_emotion: bool) -> dict:
    r = client.post(
        f"{API}/api/generate",
        json={
            "transcript": [text],
            "emotion": emo,
            "sources": [],
            "max_tokens": 200,
            "use_emotion": use_emotion,
        },
        timeout=120.0,
    )
    r.raise_for_status()
    return r.json()


def _score(suggestions: list[str]) -> dict:
    joined = " ".join(suggestions)
    return {
        "empathy": _has(EMPATHY, joined),
        "action": _has(ACTION, joined),
        "len": len(joined),
        "n": len(suggestions),
    }


def main() -> None:
    console = Console()
    httpx.get(f"{API}/api/health", timeout=5).raise_for_status()
    console.rule(f"[bold cyan]Ablation: emotion ON vs OFF[/]  {len(TESTSET)} реплик")

    rows = []
    with httpx.Client() as c:
        for i, (emo_label, text) in enumerate(TESTSET, 1):
            emo = EMO_PRESETS[emo_label]
            console.log(f"[{i}/{len(TESTSET)}] {emo_label}: {text[:50]}…")
            on = _gen(c, text, emo, use_emotion=True)
            off = _gen(c, text, emo, use_emotion=False)
            rows.append({
                "emotion": emo_label,
                "text": text,
                "on": {"suggestions": on["suggestions"], **_score(on["suggestions"]),
                       "ms": on["total_ms"]},
                "off": {"suggestions": off["suggestions"], **_score(off["suggestions"]),
                        "ms": off["total_ms"]},
            })

    def agg(subset: list[dict], side: str) -> dict:
        n = len(subset) or 1
        return {
            "empathy_rate": round(sum(r[side]["empathy"] for r in subset) / n, 2),
            "action_rate": round(sum(r[side]["action"] for r in subset) / n, 2),
            "avg_len": int(sum(r[side]["len"] for r in subset) / n),
        }

    groups = {
        "angry+sad (негатив)": [r for r in rows if r["emotion"] in ("angry", "sad")],
        "neutral": [r for r in rows if r["emotion"] == "neutral"],
        "все": rows,
    }

    table = Table(title="Эмпатия / действие: ON vs OFF по группам")
    table.add_column("группа", style="cyan")
    table.add_column("empathy ON", justify="right", style="green")
    table.add_column("empathy OFF", justify="right")
    table.add_column("action ON", justify="right", style="green")
    table.add_column("action OFF", justify="right")
    table.add_column("len ON/OFF", justify="right")
    summary = {}
    for name, subset in groups.items():
        on_a, off_a = agg(subset, "on"), agg(subset, "off")
        summary[name] = {"on": on_a, "off": off_a, "n": len(subset)}
        table.add_row(
            f"{name} (n={len(subset)})",
            f"{on_a['empathy_rate']:.0%}", f"{off_a['empathy_rate']:.0%}",
            f"{on_a['action_rate']:.0%}", f"{off_a['action_rate']:.0%}",
            f"{on_a['avg_len']}/{off_a['avg_len']}",
        )
    console.print(table)

    console.rule("[bold]Примеры (angry)")
    for r in [x for x in rows if x["emotion"] == "angry"][:2]:
        console.print(f"[yellow]Клиент:[/] {r['text']}")
        console.print(f"  [green]ON :[/] {r['on']['suggestions'][0] if r['on']['suggestions'] else '—'}")
        console.print(f"  [red]OFF:[/] {r['off']['suggestions'][0] if r['off']['suggestions'] else '—'}")
        console.print("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    console.log(f"[bold green]Saved[/] → {OUT}")


if __name__ == "__main__":
    main()
