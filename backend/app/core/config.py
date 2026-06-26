from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Telecom Copilot"
    debug: bool = True

    data_dir: Path = PROJECT_ROOT / "data"
    kb_dir: Path = PROJECT_ROOT / "data" / "kb"
    audio_dir: Path = PROJECT_ROOT / "data" / "audio_samples"
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma"

    mock_mode: bool = True

    use_fusion: bool = False

    asr_model: str = "T-one"
    ser_model: str = "xbgoose/hubert-large-speech-emotion-recognition-russian-dusha-finetuned"
    embed_model: str = "BAAI/bge-m3"
    llm_model: str = "QuantFactory/T-lite-instruct-0.1-GGUF (Q4_K_M)"

    llm_backend: str = "tlite21"

    embed_backend: str = "frida"

    asr_backend: str = "turbo"

    cors_origins: list[str] = [
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ]


settings = Settings()
