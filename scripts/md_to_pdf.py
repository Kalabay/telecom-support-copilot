"""Универсальный конвертер Markdown -> PDF (кириллица, fpdf2)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

from fpdf import FPDF

FONTS = Path("C:/Windows/Fonts")
INK = (33, 37, 41)
MUTED = (110, 118, 129)
ACCENT = (10, 110, 180)
H2COL = (20, 40, 70)
CODE_BG = (245, 246, 248)
QUOTE_BG = (245, 248, 252)
RULE = (210, 215, 222)
TH_BG = (10, 110, 180)
ZEBRA = (245, 247, 250)


class Doc(FPDF):
    def __init__(self, title: str = "") -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self.title_text = title
        self.set_auto_page_break(auto=True, margin=16)
        self.set_margins(16, 14, 16)
        self.add_font("ui", "", str(FONTS / "arial.ttf"))
        self.add_font("ui", "B", str(FONTS / "arialbd.ttf"))
        self.add_font("ui", "I", str(FONTS / "ariali.ttf"))
        self.add_font("ui", "BI", str(FONTS / "arialbi.ttf"))
        self.add_font("mono", "", str(FONTS / "cour.ttf"))

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("ui", "", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 8, str(self.page_no()), align="C")


_GLYPHS = {
    "✓": "[пров]",
    "◐": "[знаю]",
    "⚠": "[свер]",
    "✅": "[да]",
    "\U0001f536": "[опц]",
    "⬜": "[фон]",
    "✗": "нет",
    "\U0001f3af": "",
    "⬆": "", "️": "",
    "\U0001f621": "", "\U0001f622": "", "\U0001f610": "",
    "\U0001f60a": "", "\U0001f52c": "", "\U0001f3a7": "",
}


def _clean_inline(s: str) -> str:
    s = s.replace("`", "")
    for k, v in _GLYPHS.items():
        if k in s:
            s = s.replace(k, v)
    return s


def content_width(doc: Doc) -> float:
    return doc.w - doc.l_margin - doc.r_margin


def heading(doc: Doc, level: int, text: str) -> None:
    text = _clean_inline(text)
    sizes = {1: 16, 2: 12.5, 3: 11, 4: 10.5}
    if level == 1:
        if doc.get_y() > doc.t_margin + 4:
            doc.ln(3)
        doc.set_font("ui", "B", sizes[1])
        doc.set_text_color(*ACCENT)
        doc.multi_cell(0, 8, text, markdown=True)
        y = doc.get_y() + 0.5
        doc.set_draw_color(*ACCENT)
        doc.set_line_width(0.4)
        doc.line(doc.l_margin, y, doc.w - doc.r_margin, y)
        doc.ln(3)
    else:
        doc.ln(1.5)
        doc.set_font("ui", "B", sizes.get(level, 10.5))
        doc.set_text_color(*(ACCENT if level == 2 else H2COL))
        doc.multi_cell(0, 6, text, markdown=True)
        doc.ln(1)


def paragraph(doc: Doc, text: str) -> None:
    doc.set_font("ui", "", 10)
    doc.set_text_color(*INK)
    doc.multi_cell(0, 5.4, _clean_inline(text), markdown=True)
    doc.ln(1.6)


def listing(doc: Doc, items: list[str]) -> None:
    doc.set_text_color(*INK)
    indent = 4
    bw = 4
    inner = content_width(doc) - indent - bw
    for marker, txt in items:
        if doc.get_y() > doc.h - 22:
            doc.add_page()
        y = doc.get_y()
        doc.set_xy(doc.l_margin + indent, y)
        doc.set_font("ui", "B", 10)
        doc.cell(bw, 5.4, marker)
        doc.set_font("ui", "", 10)
        doc.set_xy(doc.l_margin + indent + bw, y)
        doc.multi_cell(inner, 5.4, _clean_inline(txt), markdown=True)
    doc.ln(1.6)


def code_block(doc: Doc, lines: list[str]) -> None:
    doc.ln(1)
    doc.set_font("mono", "", 8.5)
    h = 4.6 * len(lines) + 3
    left, width = doc.l_margin, content_width(doc)
    if doc.get_y() + h > doc.h - 18:
        doc.add_page()
    y0 = doc.get_y()
    doc.set_fill_color(*CODE_BG)
    doc.set_draw_color(*RULE)
    doc.rect(left, y0, width, h, style="DF")
    doc.set_xy(left + 2.5, y0 + 1.5)
    doc.set_text_color(*INK)
    for ln in lines:
        doc.set_x(left + 2.5)
        doc.cell(0, 4.6, ln[:130])
        doc.ln(4.6)
    doc.set_y(y0 + h)
    doc.ln(2)


def quote_block(doc: Doc, text: str) -> None:
    doc.ln(0.5)
    left, width = doc.l_margin, content_width(doc)
    doc.set_font("ui", "I", 9.5)
    lines = doc.multi_cell(width - 8, 5, _clean_inline(text), dry_run=True,
                           output="LINES", markdown=True)
    h = 5 * len(lines) + 3
    if doc.get_y() + h > doc.h - 18:
        doc.add_page()
    y0 = doc.get_y()
    doc.set_fill_color(*QUOTE_BG)
    doc.rect(left, y0, width, h, style="F")
    doc.set_fill_color(*ACCENT)
    doc.rect(left, y0, 1.5, h, style="F")
    doc.set_xy(left + 5, y0 + 1.5)
    doc.set_text_color(*MUTED)
    doc.multi_cell(width - 8, 5, _clean_inline(text), markdown=True)
    doc.set_y(y0 + h)
    doc.ln(2)


def render_image(doc: Doc, path: Path) -> None:
    if not path.exists():
        paragraph(doc, f"[изображение не найдено: {path.name}]")
        return
    from PIL import Image

    with Image.open(path) as im:
        iw, ih = im.size
    ar = ih / iw
    max_w = content_width(doc)
    max_h = doc.h - doc.t_margin - 22
    w = max_w
    h = w * ar
    if h > max_h:
        h = max_h
        w = h / ar
    if doc.get_y() + h > doc.h - 16:
        doc.add_page()
    doc.ln(1)
    y0 = doc.get_y()
    x = (doc.w - w) / 2
    doc.image(str(path), x=x, y=y0, w=w, h=h)
    doc.set_y(y0 + h)
    doc.ln(3)


def render_table(doc: Doc, rows: list[list[str]]) -> None:
    if not rows:
        return
    doc.ln(1)
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    header, body = rows[0], rows[1:]
    total = content_width(doc)
    w = total / ncol
    fs = 8.5 if ncol <= 3 else (7.5 if ncol <= 5 else 6.6)
    lh = fs * 0.5 + 1.2

    def cell_lines(txt: str, fw: float) -> list[str]:
        doc.set_font("ui", "", fs)
        return doc.multi_cell(fw - 2, lh, _clean_inline(txt), dry_run=True,
                              output="LINES", markdown=True, wrapmode="CHAR") or [""]

    def draw_row(cells: list[str], head: bool, zebra: bool) -> None:
        doc.set_font("ui", "B" if head else "", fs)
        nlines = max(len(cell_lines(c, w)) for c in cells)
        rh = lh * nlines + 1.6
        if doc.get_y() + rh > doc.h - 16:
            doc.add_page()
        y0 = doc.get_y()
        x0 = doc.l_margin
        if head:
            doc.set_fill_color(*TH_BG); doc.set_text_color(255, 255, 255)
        else:
            doc.set_fill_color(*(ZEBRA if zebra else (255, 255, 255)))
            doc.set_text_color(*INK)
        for c in cells:
            doc.rect(x0, y0, w, rh, style="F")
            doc.set_xy(x0 + 1, y0 + 0.8)
            doc.multi_cell(w - 2, lh, _clean_inline(c), markdown=True,
                           align="L", wrapmode="CHAR")
            x0 += w
            doc.set_xy(x0, y0)
        doc.set_draw_color(*RULE)
        doc.set_line_width(0.2)
        doc.rect(doc.l_margin, y0, total, rh)
        doc.set_y(y0 + rh)

    draw_row(header, True, False)
    for i, r in enumerate(body):
        draw_row(r, False, i % 2 == 1)
    doc.ln(2)


def parse_table_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def is_sep_row(cells: list[str]) -> bool:
    return all(re.fullmatch(r":?-{2,}:?", c.replace(" ", "")) for c in cells if c != "")


def build(md_path: Path, pdf_path: Path, title: str) -> None:
    doc = Doc(title)
    doc.add_page()
    if title:
        doc.set_font("ui", "B", 18)
        doc.set_text_color(*ACCENT)
        doc.multi_cell(0, 9, title)
        doc.ln(2)

    lines = md_path.read_text(encoding="utf-8").splitlines()
    i = 0
    para_buf: list[str] = []
    list_buf: list[tuple[str, str]] = []

    def flush_para() -> None:
        if para_buf:
            paragraph(doc, " ".join(para_buf))
            para_buf.clear()

    def flush_list() -> None:
        if list_buf:
            listing(doc, list(list_buf))
            list_buf.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_para(); flush_list()
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i]); i += 1
            code_block(doc, buf)
            i += 1
            continue

        im = re.match(r"^!\[[^\]]*\]\(([^)]+)\)\s*$", stripped)
        if im:
            flush_para(); flush_list()
            render_image(doc, md_path.parent / im.group(1))
            i += 1
            continue

        if stripped.startswith("|") and "|" in stripped[1:]:
            flush_para(); flush_list()
            trows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = parse_table_row(lines[i])
                if not is_sep_row(cells):
                    trows.append(cells)
                i += 1
            render_table(doc, trows)
            continue

        if not stripped:
            flush_para(); flush_list()
            i += 1
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            flush_para(); flush_list()
            heading(doc, len(m.group(1)), m.group(2))
            i += 1
            continue

        if re.match(r"^[-*]{3,}$", stripped):
            flush_para(); flush_list()
            doc.ln(1)
            doc.set_draw_color(*RULE); doc.set_line_width(0.2)
            yy = doc.get_y()
            doc.line(doc.l_margin, yy, doc.w - doc.r_margin, yy)
            doc.ln(2)
            i += 1
            continue

        if stripped.startswith(">"):
            flush_para(); flush_list()
            qbuf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                qbuf.append(lines[i].strip()[1:].strip()); i += 1
            quote_block(doc, " ".join(qbuf))
            continue

        lm = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if lm:
            flush_para()
            list_buf.append((lm.group(1) + ".", lm.group(2)))
            i += 1
            continue
        if re.match(r"^[-*]\s+", stripped):
            flush_para()
            list_buf.append((chr(0x2022), stripped[2:]))
            i += 1
            continue

        flush_list()
        para_buf.append(stripped)
        i += 1

    flush_para(); flush_list()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc.output(str(pdf_path))
    print(f"Saved -> {pdf_path}  ({pdf_path.stat().st_size // 1024} KB)")


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: md_to_pdf.py input.md [output.pdf] [title]")
        sys.exit(1)
    md = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else md.with_suffix(".pdf")
    title = sys.argv[3] if len(sys.argv) > 3 else ""
    build(md, out, title)


if __name__ == "__main__":
    main()
