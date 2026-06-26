"""Тесты парсера KB: фронтматтер + нарезка на чанки по секциям."""

from __future__ import annotations

from pathlib import Path

from app.pipeline.kb_loader import parse_kb_file

MD = """\
---
doc_id: no_internet
title: Нет интернета — диагностика
company: orbita
intents: [no_internet, troubleshooting]
---

# Нет интернета — диагностика

Короткое вступление: что делать, когда у клиента пропал интернет.

## Первые шаги

Попросите клиента перезагрузить роутер и подождать две минуты.

## Готовые фразы

Понимаю, как это неприятно. Давайте вместе разберёмся и всё восстановим.
"""


def test_parse_kb_file(tmp_path: Path) -> None:
    md_path = tmp_path / "no_internet.md"
    md_path.write_text(MD, encoding="utf-8")

    chunks = parse_kb_file(md_path)

    assert len(chunks) == 3

    assert all(c.doc_id == "no_internet" for c in chunks)
    assert all(c.company == "orbita" for c in chunks)
    assert all(c.title == "Нет интернета — диагностика" for c in chunks)

    sections = [c.section for c in chunks]
    assert sections == ["intro", "Первые шаги", "Готовые фразы"]

    named = [c for c in chunks if c.section != "intro"]
    assert [c.section for c in named] == ["Первые шаги", "Готовые фразы"]

    assert chunks[0].intents == ["no_internet", "troubleshooting"]

    first_steps = next(c for c in chunks if c.section == "Первые шаги")
    emb = first_steps.to_embedding_text()
    assert "Нет интернета — диагностика" in emb
    assert "Первые шаги" in emb
