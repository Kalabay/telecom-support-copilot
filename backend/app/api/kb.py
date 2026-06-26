"""REST: GET /api/kb/doc?doc_id=... — полный текст KB-статьи для читалки в UI."""

from __future__ import annotations

import json
import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger
from pydantic import BaseModel

from app.pipeline.rag import get_retriever

router = APIRouter()


class KBDocResponse(BaseModel):
    doc_id: str
    title: str
    company: str
    text: str
    instruction: str = ""
    original: str = ""
    source_url: str = ""


@lru_cache(maxsize=1)
def _playbook() -> dict[str, list[str]]:
    """Готовые ответы оператора по документам (плейбук) для вкладки «Инструкция»."""
    p = Path(__file__).resolve().parents[3] / "eval" / "results" / "doc_answers.json"
    try:
        return {o["doc_id"]: o.get("answers", []) for o in json.loads(p.read_text(encoding="utf-8"))}
    except Exception:  # noqa: BLE001
        return {}


@lru_cache(maxsize=1)
def _originals() -> dict[str, dict]:
    """Изначальные (до переработки) тексты документов и ссылки на источник для вкладки «Оригинал»."""
    out: dict[str, dict] = {}
    folder = Path(__file__).resolve().parents[3] / "data" / "kb_original"
    if folder.exists():
        for f in folder.glob("*.json"):
            try:
                out.update(json.loads(f.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                pass
    return out


@router.get("/kb/list")
def kb_list() -> dict:
    """Список всех документов базы знаний для браузера в UI."""
    try:
        docs = get_retriever().list_documents()
    except Exception as exc:  # noqa: BLE001
        logger.exception("kb list failed")
        raise HTTPException(500, f"kb list failed: {exc}") from exc
    return {"documents": docs}


@router.post("/kb/upload")
async def kb_upload(
    file: UploadFile = File(...),
    company: str | None = Form(None),
) -> dict:
    """Загрузить .md-документ(ы) базы знаний через UI и проиндексировать сразу."""
    from app.pipeline.kb_loader import load_kb

    if not (file.filename or "").endswith(".md"):
        raise HTTPException(400, "нужен .md файл в формате базы знаний")
    content = await file.read()
    try:
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / file.filename).write_bytes(content)
            chunks = load_kb(Path(td), recursive=True)
    except Exception as exc:  # noqa: BLE001
        logger.exception("kb upload parse failed")
        raise HTTPException(400, f"не разобрал файл: {exc}") from exc
    if not chunks:
        raise HTTPException(400, "в файле не нашлось статей (проверьте формат frontmatter)")
    if company:
        for c in chunks:
            c.company = company
    try:
        n = get_retriever().index_chunks(chunks)
    except Exception as exc:  # noqa: BLE001
        logger.exception("kb upload index failed")
        raise HTTPException(500, f"ошибка индексации: {exc}") from exc
    return {
        "ok": True, "chunks_added": n,
        "doc_ids": sorted({c.doc_id for c in chunks}),
        "companies": sorted({c.company for c in chunks if c.company}),
    }


@router.get("/kb/doc", response_model=KBDocResponse)
def kb_doc(doc_id: str) -> KBDocResponse:
    try:
        d = get_retriever().get_document(doc_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("kb doc fetch failed")
        raise HTTPException(500, f"kb doc failed: {exc}") from exc
    if not d.get("text"):
        raise HTTPException(404, f"doc '{doc_id}' not found")
    answers = _playbook().get(doc_id, [])
    instruction = "\n\n".join(answers)
    orig = _originals().get(doc_id, {})
    return KBDocResponse(**d, instruction=instruction,
                         original=orig.get("original", ""),
                         source_url=orig.get("source_url", ""))
