"""Парсинг markdown-документов KB с YAML-фронтматтером и нарезкой на чанки по секциям."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
YAML_LIST_RE = re.compile(r"\[(.*?)\]")


@dataclass
class KBChunk:
    """Один индексируемый кусок документа — отдельная секция."""

    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    intents: list[str] = field(default_factory=list)
    emotion_context: str = "all"
    company: str = ""

    def to_embedding_text(self) -> str:
        """Текст для эмбеддинга: заголовок документа + название секции + тело."""
        return f"{self.title}\n\n{self.section}\n\n{self.text}"


def _parse_simple_yaml(yaml_text: str) -> dict:
    """Минимальный YAML-парсер для нашего формата (scalar/list-of-strings)."""
    result: dict = {}
    for line in yaml_text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, raw = line.partition(":")
        key, raw = key.strip(), raw.strip()
        if raw.startswith("[") and raw.endswith("]"):
            items = [x.strip().strip("'\"") for x in raw[1:-1].split(",") if x.strip()]
            result[key] = items
        else:
            result[key] = raw.strip("'\"")
    return result


def _slugify(text: str) -> str:
    slug = re.sub(r"[^\w\-]+", "_", text.lower(), flags=re.UNICODE).strip("_")
    return slug[:60] or "section"


def parse_kb_file(path: Path) -> list[KBChunk]:
    """Прочитать один markdown-файл и вернуть список чанков по секциям."""
    raw = path.read_text(encoding="utf-8")

    fm_match = FRONTMATTER_RE.match(raw)
    if not fm_match:
        raise ValueError(f"{path.name}: missing YAML frontmatter")

    meta = _parse_simple_yaml(fm_match.group(1))
    body = raw[fm_match.end() :].strip()

    doc_id = meta.get("doc_id") or path.stem
    title = meta.get("title") or doc_id
    intents = meta.get("intents") or []
    emotion_context = meta.get("emotion_context") or "all"
    company = meta.get("company") or ""

    body = re.sub(r"^#\s+[^\n]+\n+", "", body, count=1)

    sections = re.split(r"\n##\s+", body)
    chunks: list[KBChunk] = []

    if len(sections) == 1:
        chunks.append(
            KBChunk(
                chunk_id=f"{doc_id}#all",
                doc_id=doc_id,
                title=title,
                section="all",
                text=body.strip(),
                intents=intents,
                emotion_context=emotion_context,
                company=company,
            )
        )
        return chunks

    intro = sections[0].strip()
    if intro:
        chunks.append(
            KBChunk(
                chunk_id=f"{doc_id}#intro",
                doc_id=doc_id,
                title=title,
                section="intro",
                text=intro,
                intents=intents,
                emotion_context=emotion_context,
                company=company,
            )
        )

    for sec in sections[1:]:
        sec = sec.strip()
        if not sec:
            continue
        section_name, _, section_body = sec.partition("\n")
        chunks.append(
            KBChunk(
                chunk_id=f"{doc_id}#{_slugify(section_name)}",
                doc_id=doc_id,
                title=title,
                section=section_name.strip(),
                text=section_body.strip(),
                intents=intents,
                emotion_context=emotion_context,
                company=company,
            )
        )

    return chunks


def load_kb(kb_dir: Path, recursive: bool = False) -> list[KBChunk]:
    """Прочитать .md файлы (кроме README.md) и собрать чанки."""
    chunks: list[KBChunk] = []
    glob = kb_dir.rglob("*.md") if recursive else kb_dir.glob("*.md")
    files = sorted(p for p in glob if p.name.lower() != "readme.md")
    for path in files:
        chunks.extend(parse_kb_file(path))
    return chunks
