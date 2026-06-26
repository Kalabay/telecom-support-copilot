"""Тонкий Python-SDK для встраивания копилота в чужую систему."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class Source:
    doc_id: str
    title: str
    snippet: str
    score: float


@dataclass
class Suggestion:
    text: str
    rank: int
    sources: list[Source]


@dataclass
class AssistResult:
    suggestions: list[Suggestion]
    sources: list[Source]
    emotion: dict | None
    retrieval_ms: int
    llm_ms: int
    total_ms: int


class CopilotClient:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def assist(
        self,
        text: str,
        company: str | None = None,
        history: list[dict] | None = None,
        emotion: dict | None = None,
        k: int = 3,
    ) -> AssistResult:
        payload: dict[str, Any] = {"text": text, "k": k}
        if company:
            payload["company"] = company
        if history:
            payload["history"] = history
        if emotion:
            payload["emotion"] = emotion

        r = requests.post(f"{self.base_url}/api/assist", json=payload, timeout=self.timeout)
        r.raise_for_status()
        d = r.json()

        def _src(s: dict) -> Source:
            return Source(s["doc_id"], s["title"], s["snippet"], s["score"])

        return AssistResult(
            suggestions=[
                Suggestion(s["text"], s["rank"], [_src(x) for x in s["sources"]])
                for s in d["suggestions"]
            ],
            sources=[_src(s) for s in d["sources"]],
            emotion=d.get("emotion"),
            retrieval_ms=d["retrieval_ms"],
            llm_ms=d["llm_ms"],
            total_ms=d["total_ms"],
        )

    def ser(self, wav_bytes: bytes, filename: str = "audio.wav") -> dict:
        """Распознать эмоцию по аудио (вернёт dict с label/arousal/valence/...)."""
        files = {"file": (filename, wav_bytes, "audio/wav")}
        r = requests.post(f"{self.base_url}/api/ser", files=files, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def kb_doc(self, doc_id: str) -> dict:
        """Полный текст статьи базы знаний по doc_id."""
        r = requests.get(
            f"{self.base_url}/api/kb/doc", params={"doc_id": doc_id}, timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()


if __name__ == "__main__":
    cop = CopilotClient()
    res = cop.assist("Третий день нет интернета, ничего не работает!", company="orbita")
    print(f"эмоция: {res.emotion}")
    print(f"RAG {res.retrieval_ms} мс, LLM {res.llm_ms} мс, всего {res.total_ms} мс")
    for s in res.suggestions:
        print(f"\n[{s.rank}] {s.text}")
        for src in s.sources:
            print(f"     └ {src.title} ({src.doc_id}, {src.score:.0%})")
