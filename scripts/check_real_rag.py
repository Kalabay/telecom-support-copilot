"""Прогон демо через WS с выводом реальных doc_id и score'ов первой sources-карточки."""

import asyncio
import json

import websockets

REAL_KB_DOC_IDS = {
    "diagnose_general", "router_reboot", "cable_check", "router_indicators",
    "wifi_no_connection", "wifi_weak_signal", "slow_speed", "mobile_no_internet",
    "balance_check", "tariff_limit", "sim_replacement", "apn_settings",
    "outage_map", "planned_works", "pppoe_setup", "mac_binding",
    "router_factory_reset", "router_firmware", "technician_visit",
    "iptv_no_signal", "escalation_l2", "apology_script", "compensation",
    "retention_offer", "relocation_setup",
}


async def main() -> None:
    async with websockets.connect("ws://127.0.0.1:8000/ws/copilot") as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "demo_trigger"}))

        ready_count = 0
        for _ in range(40):
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=20.0))
            if msg.get("pipeline_stage") != "ready":
                continue
            ready_count += 1
            last_segment = msg["transcript"][-1]["text"]
            print(f"\n=== Turn {ready_count} ===")
            print(f"  Customer said: {last_segment}")
            emo = msg.get("emotion") or {}
            if emo:
                escal = " ⚠ ESCALATION" if emo.get("escalation_risk") else ""
                print(
                    f"  Emotion: {emo['label']:>8}  conf={emo['confidence']:.2f}  "
                    f"arousal={emo['arousal']:.2f}  valence={emo['valence']:.2f}{escal}"
                )
            print("  ----- Suggestions -----")
            for s in msg["suggestions"]:
                print(f"  #{s['rank']}: {s['text']}")
            sources = msg["suggestions"][0]["sources"] if msg["suggestions"] else []
            print("  ----- Sources -----")
            for s in sources:
                marker = "✓ real" if s["doc_id"] in REAL_KB_DOC_IDS else "✗ HARDCODED"
                print(
                    f"  [{marker}] {s['doc_id']:>20}  "
                    f"score={s['score']:.3f}  | {s['title']}"
                )
            lat = msg["latency"]
            print(
                f"  Latency: SER={lat['ser_ms']}ms  "
                f"retrieval={lat['retrieval_ms']}ms  "
                f"LLM={lat['llm_ms']}ms  total={lat['total_ms']}ms"
            )
            if ready_count >= 4:
                break


if __name__ == "__main__":
    asyncio.run(main())
