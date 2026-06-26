"""Quick smoke test for the copilot WebSocket: triggers demo and prints first 4 updates."""

import asyncio
import json

import websockets


async def main() -> None:
    async with websockets.connect("ws://127.0.0.1:8000/ws/copilot") as ws:
        first = json.loads(await ws.recv())
        print(f"[init] stage={first['pipeline_stage']} transcript={len(first['transcript'])}")

        await ws.send(json.dumps({"type": "demo_trigger"}))

        for _ in range(20):
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
            stage = msg.get("pipeline_stage")
            n_seg = len(msg.get("transcript", []))
            emo = msg.get("emotion")
            sugg = len(msg.get("suggestions") or [])
            print(
                f"[{stage:>12}] segs={n_seg} emotion="
                f"{emo['label'] if emo else '—':>8} sugg={sugg}"
                + (
                    f" escalation_risk={emo['escalation_risk']}"
                    if emo and emo.get("escalation_risk")
                    else ""
                )
            )
            if stage == "idle" and n_seg > 0:
                break


if __name__ == "__main__":
    asyncio.run(main())
