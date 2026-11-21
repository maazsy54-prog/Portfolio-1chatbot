"""Unit tests for chunking, using a deterministic word-count 'tokenizer'."""
import pytest

from app.ingestion.chunking import TokenCounter, chunk_document, split_text
from app.ingestion.extract import ExtractedDocument, ExtractedSection

WORDS = TokenCounter("words", lambda t: len(t.split()))


def sentences(n: int, words_each: int = 8) -> str:
    return " ".join(
        " ".join(f"w{i}x{j}" for j in range(words_each)) + "." for i in range(n)
    )


def test_chunks_respect_token_budget() -> None:
    chunks = split_text(sentences(40), size=30, overlap=5, count=WORDS)
    assert len(chunks) > 1
    assert all(WORDS(c) <= 30 for c in chunks)


def test_short_text_is_single_chunk() -> None:
    assert len(split_text("one two three.", 50, 5, WORDS)) == 1


def test_consecutive_chunks_overlap() -> None:
    chunks = split_text(sentences(40), size=30, overlap=10, count=WORDS)
    for a, b in zip(chunks, chunks[1:]):
        assert set(a.split()) & set(b.split()), "neighbouring chunks share no text"


def test_no_text_is_lost() -> None:
    text = sentences(40)
    joined = " ".join(split_text(text, size=30, overlap=5, count=WORDS))
    assert set(text.split()) <= set(joined.split())


def test_table_rows_are_not_split_mid_row() -> None:
    rows = [f"Plan: P{i} | Price: ${i} | Notes: standard tier number {i}" for i in range(30)]
    chunks = split_text("\n".join(rows), size=40, overlap=0, count=WORDS)
    for row in rows:
        assert any(row in c for c in chunks), f"row broken across chunks: {row}"


def test_unbroken_string_is_hard_split() -> None:
    chars = TokenCounter("chars", lambda t: len(t))
    chunks = split_text("x" * 500, size=100, overlap=0, count=chars)
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(chunks) == "x" * 500


def test_urdu_sentences_split_on_urdu_full_stop() -> None:
    urdu = " ".join(f"یہ جملہ نمبر {i} ہے۔" for i in range(30))
    chunks = split_text(urdu, size=20, overlap=0, count=WORDS)
    assert len(chunks) > 1
    assert all(c.rstrip().endswith("۔") for c in chunks)


def test_empty_text_gives_no_chunks() -> None:
    assert split_text("   \n ", 10, 2, WORDS) == []


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_parameters_raise(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        split_text("hello", size, overlap, WORDS)


def test_document_chunks_keep_page_metadata_and_positions() -> None:
    doc = ExtractedDocument(
        "pdf",
        [ExtractedSection(sentences(20), page=1), ExtractedSection(sentences(20), page=2)],
        page_count=2,
    )
    chunks = chunk_document(doc, size=30, overlap=5, counter=WORDS)
    assert [c.position for c in chunks] == list(range(len(chunks)))
    assert {c.page for c in chunks} == {1, 2}
    pages = [c.page for c in chunks]
    assert pages == sorted(pages), "a chunk crossed a page boundary"


def test_docx_section_metadata_preserved() -> None:
    doc = ExtractedDocument("docx", [ExtractedSection(sentences(10), section="Returns")])
    chunks = chunk_document(doc, size=20, overlap=3, counter=WORDS)
    assert {c.section for c in chunks} == {"Returns"}
