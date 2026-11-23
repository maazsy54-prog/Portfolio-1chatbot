"""Each generated demo file must trigger exactly the behaviour the demo script promises."""
import pytest

from app.errors import BadRequestError, PayloadTooLargeError, UnsupportedMediaTypeError
from app.ingestion.extract import extract_text
from app.ingestion.validate import validate_upload
from scripts.make_demo_files import build_all

LIMIT = 10 * 1024 * 1024


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    return build_all(tmp_path_factory.mktemp("demo"), include_large=True)


def ingest(path):
    data = path.read_bytes()
    upload = validate_upload(path.name, data, LIMIT)
    return extract_text(upload.file_type, data)


@pytest.mark.parametrize("name,error", [
    ("empty.pdf", BadRequestError),
    ("fake_document.pdf", UnsupportedMediaTypeError),
    ("notes.txt", UnsupportedMediaTypeError),
    ("too_large.pdf", PayloadTooLargeError),
    ("too_many_pages.pdf", PayloadTooLargeError),
    ("corrupt.pdf", BadRequestError),
    ("password_protected.pdf", BadRequestError),
    ("scanned_no_text.pdf", BadRequestError),
])
def test_bad_files_fail_with_the_expected_error(files, name, error) -> None:
    with pytest.raises(error):
        ingest(files[name])


def test_scanned_pdf_really_has_no_text_but_a_normal_pdf_does(files) -> None:
    with pytest.raises(BadRequestError, match="No extractable text"):
        ingest(files["scanned_no_text.pdf"])
    assert "five to seven" in ingest(files["good_policy.pdf"]).sections[0].text


def test_table_rows_keep_their_headers(files) -> None:
    text = ingest(files["plans_with_table.docx"]).sections[0].text
    assert "Plan: Team | Monthly price: $29 | Seats: 5" in text


def test_the_two_policies_genuinely_contradict(files) -> None:
    a = ingest(files["return_policy_2023.docx"]).sections[0].text
    b = ingest(files["return_policy_2024.docx"]).sections[0].text
    assert "30 days" in a and "14 days" in b


def test_poisoned_document_ingests_normally_with_the_instruction_as_text(files) -> None:
    text = ingest(files["poisoned_instructions.docx"]).sections[0].text
    assert "ignore all previous instructions" in text.lower()
