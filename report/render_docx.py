"""Render report.md to a compact three-page report.docx.

Deliberately small-scoped: it handles exactly the Markdown constructs used in
report.md (ATX headings, paragraphs, bullet and numbered lists, pipe tables, and
**bold** / *italic* / `code` spans) rather than pretending to be a general converter.

Usage:  python report/render_docx.py [--md report/report.md] [--out report/report.docx]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
ACCENT = RGBColor(0x16, 0x32, 0x4A)

# Point sizes tuned against a MEASURED page count, not an estimate. An earlier version of
# this file guessed pagination from word counts and was out by more than a page. Word's own
# ComputeStatistics is the source of truth now - see report/page_count.ps1.
# Measured with the current report.md: 8.6pt spills to 4 pages, 8.4pt fits 3. 8.2pt is one
# notch below that cliff so small edits do not silently push it over.
BODY_PT = 8.2
H1_PT = 11.5
H2_PT = 8.8
TABLE_PT = 6.4


def _style(doc: Document) -> None:
    n = doc.styles["Normal"]
    n.font.name = "Calibri"
    n.font.size = Pt(BODY_PT)
    n.paragraph_format.space_after = Pt(1.2)
    n.paragraph_format.space_before = Pt(0)
    n.paragraph_format.line_spacing = 0.98
    rpr = n.element.get_or_add_rPr().get_or_add_rFonts()
    rpr.set(qn("w:eastAsia"), "Calibri")

    for sec in doc.sections:
        sec.top_margin = sec.bottom_margin = Inches(0.38)
        sec.left_margin = sec.right_margin = Inches(0.42)


_TOKEN = re.compile(r"(\*\*.+?\*\*|(?<!\*)\*[^*]+?\*(?!\*)|`[^`]+?`)")


def _add_runs(par, text: str, base_bold: bool = False, size: float | None = None) -> None:
    """Emit runs honouring **bold**, *italic* and `code` spans."""
    for part in _TOKEN.split(text):
        if not part:
            continue
        bold, italic, mono = base_bold, False, False
        if part.startswith("**") and part.endswith("**"):
            part, bold = part[2:-2], True
        elif part.startswith("*") and part.endswith("*"):
            part, italic = part[1:-1], True
        elif part.startswith("`") and part.endswith("`"):
            part, mono = part[1:-1], True
        r = par.add_run(part)
        r.bold = bold
        r.italic = italic
        if mono:
            r.font.name = "Consolas"
            r.font.size = Pt((size or BODY_PT) - 0.5)
        elif size:
            r.font.size = Pt(size)


def _heading(doc: Document, text: str, level: int):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    if level == 1:
        pf.space_before, pf.space_after = Pt(0), Pt(3)
        _add_runs(p, text, base_bold=True, size=H1_PT)
        for r in p.runs:
            r.font.color.rgb = ACCENT
    else:
        pf.space_before, pf.space_after = Pt(4), Pt(1.5)
        pf.keep_with_next = True
        _add_runs(p, text, base_bold=True, size=H2_PT)
        for r in p.runs:
            r.font.color.rgb = ACCENT
    return p


def _table(doc: Document, rows: list[list[str]]):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            c = t.cell(i, j)
            c.text = ""
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(0.2)
            p.paragraph_format.space_before = Pt(0.2)
            if j > 0:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _add_runs(p, cell, base_bold=(i == 0), size=TABLE_PT)
    return t


def _is_table_sep(line: str) -> bool:
    s = line.strip().strip("|")
    return bool(s) and set(s.replace(":", "").replace("-", "").replace("|", "").strip()) == set()


def render(md_path: Path, out_path: Path, figures: list[Path] | None = None) -> None:
    doc = Document()
    _style(doc)
    lines = md_path.read_text(encoding="utf-8").splitlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # pipe table
        if stripped.startswith("|") and i + 1 < len(lines) and _is_table_sep(lines[i + 1]):
            rows = []
            header = [c.strip() for c in stripped.strip("|").split("|")]
            rows.append(header)
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                cells += [""] * (len(header) - len(cells))
                rows.append(cells[: len(header)])
                i += 1
            _table(doc, rows)
            doc.add_paragraph().paragraph_format.space_after = Pt(0.5)
            continue

        # inline image:  ![width_inches](path/to.png)
        img = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", stripped)
        if img:
            path = (md_path.parent / img.group(2)).resolve()
            if not path.exists():
                path = (Path.cwd() / img.group(2)).resolve()
            if path.exists():
                try:
                    width = Inches(float(img.group(1)))
                except ValueError:
                    width = Inches(7.0)
                doc.add_picture(str(path), width=width)
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                doc.paragraphs[-1].paragraph_format.space_after = Pt(3)
            else:
                print(f"  [warn] figure not found, skipped: {img.group(2)}")
            i += 1
            continue

        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            _heading(doc, stripped[level:].strip(), 1 if level == 1 else 2)
            i += 1
            continue

        m = re.match(r"^(\d+)\.\s+", stripped)
        if m:
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.17)
            p.paragraph_format.first_line_indent = Inches(-0.17)
            p.paragraph_format.space_after = Pt(0.8)
            body, i = _join_wrapped(lines, i, prefix_len=len(m.group(0)))
            _add_runs(p, f"{m.group(1)}. {body}")
            continue

        if stripped.startswith("- "):
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.left_indent = Inches(0.2)
            p.paragraph_format.space_after = Pt(0.8)
            text, i = _join_wrapped(lines, i, prefix_len=2)
            _add_runs(p, text)
            for r in p.runs:
                r.font.size = Pt(BODY_PT)
            continue

        text, i = _join_wrapped(lines, i)
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        _add_runs(p, text)

    if figures:
        for fig in figures:
            if fig.exists():
                doc.add_picture(str(fig), width=Inches(7.1))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_path)
    print("wrote", out_path)


def _join_wrapped(lines: list[str], i: int, prefix_len: int = 0) -> tuple[str, int]:
    """Join a soft-wrapped Markdown block into one logical line."""
    buf = [lines[i].strip()[prefix_len:]]
    j = i + 1
    while j < len(lines):
        nxt = lines[j].strip()
        if (not nxt or nxt.startswith("#") or nxt.startswith("|")
                or nxt.startswith("- ") or re.match(r"^\d+\.\s", nxt)):
            break
        buf.append(nxt)
        j += 1
    return " ".join(buf), j


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=str(HERE / "report.md"))
    ap.add_argument("--out", default=str(HERE / "report.docx"))
    ap.add_argument("--figures", nargs="*", default=None)
    a = ap.parse_args()
    figs = [Path(f) for f in a.figures] if a.figures else None
    render(Path(a.md), Path(a.out), figs)
