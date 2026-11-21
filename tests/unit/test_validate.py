"""Unit tests for upload validation."""
import io
import zipfile

import pytest

from app.errors import BadRequestError, PayloadTooLargeError, UnsupportedMediaTypeError
from app.ingestion.validate import sanitize_filename, validate_upload

MB = 1024 * 1024


def make_docx_bytes() -> bytes:
    from docx import Document

    buf = io.BytesIO()
    doc = Document()
    doc.add_paragraph("hello world")
    doc.save(buf)
    return buf.getvalue()


PDF_BYTES = b"%PDF-1.4\n% minimal header for validation tests\n"


def test_valid_pdf_and_docx() -> None:
    pdf = validate_upload("a.PDF", PDF_BYTES, 10 * MB)
    assert pdf.file_type == "pdf" and len(pdf.sha256) == 64
    docx = validate_upload("b.docx", make_docx_bytes(), 10 * MB)
    assert docx.file_type == "docx"


def test_empty_file_is_400() -> None:
    with pytest.raises(BadRequestError):
        validate_upload("a.pdf", b"", MB)


def test_too_large_is_413() -> None:
    with pytest.raises(PayloadTooLargeError):
        validate_upload("a.pdf", PDF_BYTES + b"x" * 200, 100)


def test_unsupported_extension_is_415() -> None:
    with pytest.raises(UnsupportedMediaTypeError):
        validate_upload("malware.exe", b"MZ\x90\x00data", MB)


def test_spoofed_extension_is_415() -> None:
    with pytest.raises(UnsupportedMediaTypeError):
        validate_upload("fake.pdf", b"MZ\x90\x00not a pdf", MB)


def test_zip_that_is_not_docx_is_415() -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("readme.txt", "hi")
    with pytest.raises(UnsupportedMediaTypeError):
        validate_upload("x.docx", buf.getvalue(), MB)


def test_same_content_gives_same_hash() -> None:
    a = validate_upload("one.pdf", PDF_BYTES, MB)
    b = validate_upload("two.pdf", PDF_BYTES, MB)
    assert a.sha256 == b.sha256


@pytest.mark.parametrize("raw,expected", [
    ("../../etc/passwd.pdf", "passwd.pdf"),
    ("C:\\Users\\me\\report.pdf", "report.pdf"),
    ("we<ird>|name?.pdf", "we_ird__name_.pdf"),
    ("....", "document"),
])
def test_sanitize_filename(raw: str, expected: str) -> None:
    assert sanitize_filename(raw) == expected
