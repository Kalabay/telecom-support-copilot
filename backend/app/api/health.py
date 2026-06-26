from fastapi import APIRouter

from app.core.config import settings

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "app": settings.app_name,
        "mock_mode": settings.mock_mode,
        "models": {
            "asr": settings.asr_model,
            "ser": settings.ser_model,
            "embed": settings.embed_model,
            "llm": settings.llm_model,
        },
    }
