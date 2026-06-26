"""REST: аналитика звонков для дашборда QA (читает data/call_logs/calls.jsonl)."""

from __future__ import annotations

from collections import Counter, defaultdict

from fastapi import APIRouter

from app.core.call_log import read_events

router = APIRouter()


@router.get("/analytics")
def analytics() -> dict:
    events = read_events()
    client = [e for e in events if e.get("type") == "client_turn"]
    operator = [e for e in events if e.get("type") == "operator_turn"]

    sessions: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        sessions[e.get("session_id", "?")].append(e)

    emotion_counts = Counter(
        (e.get("emotion") or {}).get("label", "?") for e in client
    )
    escalation_count = sum(
        1 for evs in sessions.values()
        if any((e.get("emotion") or {}).get("escalation_risk")
               for e in evs if e.get("type") == "client_turn")
    )
    company_counts = Counter(e.get("company") or "—" for e in client)
    lats = [
        (e.get("latency") or {}).get("total_ms", 0)
        for e in client
        if e.get("latency")
    ]
    avg_latency_ms = int(sum(lats) / len(lats)) if lats else 0

    recent = []
    for sid, evs in sessions.items():
        cts = [e for e in evs if e.get("type") == "client_turn"]
        ots = [e for e in evs if e.get("type") == "operator_turn"]
        started = min((e.get("ts", "") for e in evs), default="")
        recent.append(
            {
                "session_id": sid,
                "started": started,
                "company": cts[-1].get("company") if cts else None,
                "client_turns": len(cts),
                "operator_turns": len(ots),
                "last_emotion": (cts[-1].get("emotion") or {}).get("label") if cts else None,
                "escalated": any((e.get("emotion") or {}).get("escalation_risk") for e in cts),
            }
        )
    recent.sort(key=lambda r: r["started"], reverse=True)

    return {
        "total_sessions": len(sessions),
        "total_client_turns": len(client),
        "total_operator_turns": len(operator),
        "escalation_count": escalation_count,
        "avg_latency_ms": avg_latency_ms,
        "emotion_counts": dict(emotion_counts),
        "company_counts": dict(company_counts),
        "recent_sessions": recent[:50],
    }


@router.get("/analytics/session/{session_id}")
def session_detail(session_id: str) -> dict:
    events = [e for e in read_events() if e.get("session_id") == session_id]
    events.sort(key=lambda e: e.get("ts", ""))
    return {"session_id": session_id, "events": events}
