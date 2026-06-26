"""Python-SDK телеком-копилота: импортируемый пакет."""

from .client import AssistResult, CopilotClient, Source, Suggestion

__all__ = ["CopilotClient", "AssistResult", "Suggestion", "Source"]
