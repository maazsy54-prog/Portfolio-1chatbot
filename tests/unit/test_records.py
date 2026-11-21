"""Unit tests for turning chunks into vector records."""
import pytest

from app.ingestion.chunking import Chunk
from app.ingestion.records import build_records, make_vector_id


def test_records_carry_citation_metadata() -> None:
    chunks = [Chunk("hello", 0, 3, page=2), Chunk("world", 1, 3, section="Intro")]
    records = build_records("doc1", "a.pdf", chunks, [[0.1], [0.2]])
    assert [r.id for r in records] == [make_vector_id("doc1", 0), make_vector_id("doc1", 1)]
    assert records[0].metadata["page"] == 2 and records[0].metadata["filename"] == "a.pdf"
    assert records[1].metadata["section"] == "Intro" and records[1].metadata["text"] == "world"


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        build_records("d", "a.pdf", [Chunk("x", 0, 1)], [])
