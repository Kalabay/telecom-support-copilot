"""Обратная совместимость: старый путь импорта `from copilot_client import ...`."""

from telecom_copilot import AssistResult, CopilotClient, Source, Suggestion

__all__ = ["CopilotClient", "AssistResult", "Suggestion", "Source"]
