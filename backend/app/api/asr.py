"""REST: список ASR-движков и переключение текущего."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.pipeline.asr import ENGINES, get_asr

router = APIRouter()


class EngineReq(BaseModel):
    engine: str


@router.get("/asr/engines")
def list_engines() -> dict:
    """Список доступных ASR-движков и текущий."""
    return {
        "engines": [{"id": k, "label": v[2]} for k, v in ENGINES.items()],
        "current": get_asr().current,
    }


@router.post("/asr/engine")
def set_engine(req: EngineReq) -> dict:
    if not get_asr().set_engine(req.engine):
        raise HTTPException(400, f"unknown engine: {req.engine}")
    return {"ok": True, "current": get_asr().current}
