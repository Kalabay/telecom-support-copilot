import asyncio
import json
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from app.core.call_log import log_event
from app.models.schemas import ClientMessage, CopilotUpdate, KBSource, TranscriptSegment
from app.pipeline.live import run_voice_input
from app.pipeline.mock import run_demo

router = APIRouter()


@router.websocket("/ws/copilot")
async def copilot_socket(ws: WebSocket) -> None:
    """WS-канал: текст (управление демо/stop/ping) и бинарь (запись микрофона)."""
    await ws.accept()
    logger.info("WS connected")

    transcript: list[TranscriptSegment] = []
    voice_queue: asyncio.Queue[tuple[bytes, str, str | None]] = asyncio.Queue(maxsize=8)
    demo_task: asyncio.Task | None = None
    current_speaker = "customer"
    current_company: str | None = None
    session_id = uuid.uuid4().hex[:12]
    logged_turns = {"n": 0}

    async def send_update(update: CopilotUpdate) -> None:
        """Отправить апдейт во фронт и залогировать завершённую клиентскую реплику."""
        await ws.send_text(update.model_dump_json())
        if update.pipeline_stage == "ready" and update.emotion is not None:
            cnt = sum(1 for s in update.transcript if s.speaker == "customer")
            if cnt > logged_turns["n"]:
                logged_turns["n"] = cnt
                last = next(
                    (s.text for s in reversed(update.transcript) if s.speaker == "customer"),
                    "",
                )
                log_event({
                    "session_id": session_id,
                    "type": "client_turn",
                    "text": last,
                    "emotion": update.emotion.model_dump(mode="json"),
                    "suggestions": [s.text for s in update.suggestions],
                    "latency": update.latency.model_dump() if update.latency else None,
                    "company": current_company,
                })

    async def stream_demo() -> None:
        nonlocal transcript
        transcript = []
        try:
            async for update in run_demo():
                transcript = list(update.transcript)
                await send_update(update)
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Demo stream stopped: {exc}")

    async def voice_worker() -> None:
        """Единственный потребитель очереди — гарантирует последовательность."""
        while True:
            blob, spk, comp = await voice_queue.get()
            try:
                async for update in run_voice_input(blob, transcript, spk, comp):
                    await send_update(update)
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                logger.exception(f"Voice pipeline error: {exc}")
            finally:
                voice_queue.task_done()

    worker = asyncio.create_task(voice_worker())

    try:
        await ws.send_text(CopilotUpdate(transcript=[]).model_dump_json())

        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break

            if msg.get("text") is not None:
                try:
                    cm = ClientMessage(**json.loads(msg["text"]))
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"Bad client message: {exc}")
                    continue

                logger.info(f"<- {cm.type}")

                if cm.type == "demo_trigger":
                    if demo_task and not demo_task.done():
                        demo_task.cancel()
                    demo_task = asyncio.create_task(stream_demo())

                elif cm.type == "stop":
                    if demo_task and not demo_task.done():
                        demo_task.cancel()
                    while not voice_queue.empty():
                        try:
                            voice_queue.get_nowait()
                            voice_queue.task_done()
                        except asyncio.QueueEmpty:
                            break
                    transcript = []
                    logged_turns["n"] = 0
                    await ws.send_text(CopilotUpdate(transcript=[]).model_dump_json())

                elif cm.type == "operator_said":
                    payload = cm.payload or {}
                    text = payload.get("text", "").strip()
                    raw_sources = payload.get("sources") or []
                    if text:
                        start = transcript[-1].end_ms + 300 if transcript else 0
                        sources: list[KBSource] = []
                        for s in raw_sources:
                            try:
                                sources.append(KBSource(**s))
                            except Exception:  # noqa: BLE001
                                pass
                        transcript.append(
                            TranscriptSegment(
                                text=text,
                                speaker="operator",
                                start_ms=start,
                                end_ms=start + len(text) * 50,
                                confidence=1.0,
                                sources=sources,
                            )
                        )
                        await ws.send_text(
                            CopilotUpdate(
                                transcript=list(transcript), pipeline_stage="ready"
                            ).model_dump_json()
                        )
                        log_event({
                            "session_id": session_id,
                            "type": "operator_turn",
                            "text": text,
                            "sources": [s.doc_id for s in sources],
                            "company": current_company,
                        })

                elif cm.type == "voice_speaker":
                    spk = (cm.payload or {}).get("speaker", "customer")
                    current_speaker = "operator" if spk == "operator" else "customer"
                    logger.info(f"voice speaker = {current_speaker}")

                elif cm.type == "set_company":
                    comp = (cm.payload or {}).get("company") or None
                    current_company = comp if comp and comp != "all" else None
                    logger.info(f"KB company = {current_company}")

                elif cm.type == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))

            elif msg.get("bytes") is not None:
                blob: bytes = msg["bytes"]
                logger.info(
                    f"<- voice blob {len(blob)}b speaker={current_speaker} "
                    f"(queue={voice_queue.qsize()})"
                )
                try:
                    voice_queue.put_nowait((blob, current_speaker, current_company))
                except asyncio.QueueFull:
                    logger.warning("Voice queue full, dropping oldest")
                    try:
                        voice_queue.get_nowait()
                        voice_queue.task_done()
                        voice_queue.put_nowait((blob, current_speaker, current_company))
                    except (asyncio.QueueEmpty, asyncio.QueueFull):
                        pass

    except WebSocketDisconnect:
        logger.info("WS disconnected")
    finally:
        worker.cancel()
        if demo_task and not demo_task.done():
            demo_task.cancel()
