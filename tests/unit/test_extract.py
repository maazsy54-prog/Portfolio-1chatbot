"""Unit tests for PDF/DOCX text extraction. Fixtures are built in memory."""
import io

import pymupdf
import pytest
from docx import Document

from app.errors import BadRequestError, PayloadTooLargeError
from app.ingestion.extract import clean_text, extract_docx, extract_pdf

LONG = "The refund policy allows returns within thirty days of purchase. "


def make_pdf(pages: list[str], password: str | None = None) -> bytes:
    pdf = pymupdf.open()
    for text in pages:
        page = pdf.new_page()
        if text:
            page.insert_text((72, 72), text)
        else:
            page.draw_rect(pymupdf.Rect(50, 50, 200, 200))  # graphics only
    if password:
        return pdf.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256,
                           owner_pw=password, user_pw=password)
    return pdf.tobytes()


def make_docx(build) -> bytes:
    doc = Document()
    build(doc)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def test_pdf_pages_are_numbered() -> None:
    result = extract_pdf(make_pdf([LONG + "Page one.", LONG + "Page two."]))
    assert result.page_count == 2
    assert [s.page for s in result.sections] == [1, 2]
    assert "Page two" in result.sections[1].text


def test_pdf_blank_pages_skipped_but_counted() -> None:
    result = extract_pdf(make_pdf([LONG, "", LONG]))
    assert result.page_count == 3
    assert [s.page for s in result.sections] == [1, 3]


def test_scanned_pdf_without_text_is_400() -> None:
    with pytest.raises(BadRequestError, match="No extractable text"):
        extract_pdf(make_pdf(["", ""]))


def test_encrypted_pdf_is_400() -> None:
    with pytest.raises(BadRequestError, match="Password"):
        extract_pdf(make_pdf([LONG], password="secret"))


def test_corrupt_pdf_is_400() -> None:
    with pytest.raises(BadRequestError):
        extract_pdf(b"%PDF-1.4 this is not really a pdf")


def test_pdf_page_limit_is_413() -> None:
    with pytest.raises(PayloadTooLargeError):
        extract_pdf(make_pdf([LONG, LONG]), max_pages=1)


def test_docx_headings_become_sections() -> None:
    def build(doc: Document) -> None:
        doc.add_heading("Returns", level=1)
        doc.add_paragraph(LONG)
        doc.add_heading("Shipping", level=1)
        doc.add_paragraph("Shipping takes five business days within the country.")

    result = extract_docx(make_docx(build))
    assert [s.section for s in result.sections] == ["Returns", "Shipping"]
    assert "thirty days" in result.sections[0].text


def test_docx_table_rows_keep_their_headers() -> None:
    def build(doc: Document) -> None:
        doc.add_heading("Fees", level=1)
        doc.add_paragraph(LONG)
        table = doc.add_table(rows=3, cols=2)
        for r, (a, b) in enumerate([("Plan", "Price"), ("Basic", "$5"), ("Pro", "$20")]):
            table.cell(r, 0).text, table.cell(r, 1).text = a, b

    text = extract_docx(make_docx(build)).sections[0].text
    assert "Plan: Basic | Price: $5" in text
    assert "Plan: Pro | Price: $20" in text


def test_docx_unicode_urdu_preserved() -> None:
    urdu = "واپسی کی پالیسی تیس دن کے اندر واپسی کی اجازت دیتی ہے۔ " * 2

    result = extract_docx(make_docx(lambda d: d.add_paragraph(urdu)))
    assert "واپسی" in result.sections[0].text


def test_docx_without_text_is_400() -> None:
    with pytest.raises(BadRequestError, match="No extractable text"):
        extract_docx(make_docx(lambda d: d.add_paragraph("   ")))


def test_corrupt_docx_is_400() -> None:
    with pytest.raises(BadRequestError):
        extract_docx(b"PK\x03\x04 definitely not a docx")


def test_clean_text_fixes_ligatures_and_blank_lines() -> None:
    assert clean_text("o\ufb03ce\x00\n\n\n\nnext  ") == "office\n\nnext"
