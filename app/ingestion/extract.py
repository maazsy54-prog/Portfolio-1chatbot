"""Text extraction for PDF and DOCX.

Each document becomes a list of ExtractedSection objects carrying the text
plus where it came from (page for PDFs, heading for DOCX), so every chunk
can later be cited precisely.
"""
import io
import re
import unicodedata
from dataclasses import dataclass, field

import pymupdf
from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.errors import AppError, BadRequestError, PayloadTooLargeError

# A document with fewer characters than this is treated as image-only/scanned.
MIN_TOTAL_CHARS = 50


@dataclass(frozen=True)
class ExtractedSection:
    """A contiguous piece of text with its location in the source file."""

    text: str
    page: int | None = None  # PDF page number (1-based)
    section: str | None = None  # DOCX heading the text sits under


@dataclass(frozen=True)
class ExtractedDocument:
    """All text extracted from one file."""

    file_type: str
    sections: list[ExtractedSection] = field(default_factory=list)
    page_count: int | None = None

    @property
    def total_chars(self) -> int:
        return sum(len(s.text) for s in self.sections)


def clean_text(text: str) -> str:
    """Normalize unicode (fixes ligatures), drop NULs, tidy whitespace."""
    text = unicodedata.normalize("NFKC", text).replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _require_text(doc: ExtractedDocument) -> None:
    """Reject documents with no usable text (typically scanned PDFs)."""
    if doc.total_chars < MIN_TOTAL_CHARS:
        raise BadRequestError(
            "No extractable text found. The file may be scanned or image-only."
        )


def extract_pdf(
    data: bytes, *, max_pages: int = 300
) -> ExtractedDocument:
    """Extract text page by page from a PDF."""
    try:
        pdf = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise BadRequestError("Could not read the PDF; it may be corrupt.") from exc

    try:
        with pdf:
            if pdf.needs_pass:
                raise BadRequestError("Password-protected PDFs are not supported.")
            if pdf.page_count > max_pages:
                raise PayloadTooLargeError(
                    f"PDF has {pdf.page_count} pages; the limit is {max_pages}."
                )
            sections: list[ExtractedSection] = []
            for number, page in enumerate(pdf, start=1):
                text = clean_text(page.get_text("text", sort=True))
                if text:
                    sections.append(ExtractedSection(text=text, page=number))
            result = ExtractedDocument("pdf", sections, pdf.page_count)
    except AppError:
        raise
    except Exception as exc:
        raise BadRequestError("Could not read the PDF; it may be corrupt.") from exc

    _require_text(result)
    return result


def _table_to_text(table: Table) -> str:
    """Serialize a table as 'Header: value | Header: value' lines.

    Keeping the header next to every value lets each row stand alone when
    it is split into chunks and embedded.
    """
    rows: list[list[str]] = []
    for row in table.rows:
        seen: set[int] = set()
        cells: list[str] = []
        for cell in row.cells:
            if id(cell._tc) in seen:  # merged cells repeat; keep one
                continue
            seen.add(id(cell._tc))
            cells.append(clean_text(cell.text).replace("\n", " "))
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    if len(rows) == 1:
        return " | ".join(rows[0])
    header, body = rows[0], rows[1:]
    lines = []
    for row in body:
        pairs = [
            f"{header[i]}: {value}" if i < len(header) and header[i] else value
            for i, value in enumerate(row)
        ]
        lines.append(" | ".join(pairs))
    return "\n".join(lines)


def extract_docx(data: bytes) -> ExtractedDocument:
    """Extract paragraphs and tables from a DOCX, grouped under headings."""
    try:
        document = Document(io.BytesIO(data))
    except Exception as exc:
        raise BadRequestError("Could not read the DOCX; it may be corrupt.") from exc

    sections: list[ExtractedSection] = []
    heading: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        text = clean_text("\n".join(buffer))
        if text:
            sections.append(ExtractedSection(text=text, section=heading))
        buffer.clear()

    try:
        for child in document.element.body.iterchildren():
            if child.tag == qn("w:p"):
                paragraph = Paragraph(child, document)
                style = (paragraph.style.name or "") if paragraph.style else ""
                text = paragraph.text.strip()
                if not text:
                    continue
                if style.startswith("Heading") or style == "Title":
                    flush()
                    heading = text
                    buffer.append(text)  # keep the title in the text too
                else:
                    buffer.append(text)
            elif child.tag == qn("w:tbl"):
                table_text = _table_to_text(Table(child, document))
                if table_text:
                    buffer.append(table_text)
        flush()
    except Exception as exc:
        raise BadRequestError("Could not read the DOCX; it may be corrupt.") from exc

    result = ExtractedDocument("docx", sections, None)
    _require_text(result)
    return result


def extract_text(file_type: str, data: bytes, *, max_pages: int = 300) -> ExtractedDocument:
    """Dispatch to the extractor for a validated file type."""
    if file_type == "pdf":
        return extract_pdf(data, max_pages=max_pages)
    if file_type == "docx":
        return extract_docx(data)
    raise BadRequestError(f"No extractor for file type '{file_type}'.")
