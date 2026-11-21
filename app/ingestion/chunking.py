"""Token-aware recursive chunking.

Text is split on the largest natural boundary that fits (paragraph, line,
sentence, word, finally characters) and pieces are merged greedily up to a
token budget, with a token overlap between neighbouring chunks.

Chunks never cross a page/section boundary, so each chunk maps to exactly
one page (PDF) or heading (DOCX) and citations stay precise.
"""
import logging
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

from app.ingestion.extract import ExtractedDocument

logger = logging.getLogger(__name__)

# Boundaries tried in order. Sentence ends include Arabic/Urdu '۔' and '؟'.
_SEPARATORS: tuple[str, ...] = (
    r"(\n\n+)",
    r"(\n)",
    r"(?<=[.!?\u06d4\u061f])(\s+)",
    r"(\s+)",
)


@dataclass(frozen=True)
class TokenCounter:
    """Callable that counts tokens, with a name recorded for reproducibility."""

    name: str
    fn: Callable[[str], int]

    def __call__(self, text: str) -> int:
        return self.fn(text)


def _approx_tokens(text: str) -> int:
    """Rough fallback: about four characters per token."""
    return math.ceil(len(text) / 4) if text else 0


@lru_cache
def get_token_counter() -> TokenCounter:
    """Return a tiktoken-based counter, or a character heuristic if unavailable.

    tiktoken downloads its vocabulary on first use; if that fails (offline),
    we fall back so ingestion never breaks. Counts are approximate for
    Cohere's own tokenizer either way; the budget only controls chunk size.
    """
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        return TokenCounter(
            "tiktoken:cl100k_base",
            lambda text: len(encoding.encode(text, disallowed_special=())),
        )
    except Exception as exc:  # network, missing vocab, import errors
        logger.warning("tiktoken unavailable (%s); using approximate counter", exc)
        return TokenCounter("approx:chars/4", _approx_tokens)


@dataclass(frozen=True)
class Chunk:
    """One embeddable passage plus the metadata needed to cite it."""

    text: str
    position: int  # 0-based index within the document
    token_count: int
    page: int | None = None
    section: str | None = None


def _split_keep(text: str, pattern: str) -> list[str]:
    """Split on a pattern, attaching each separator to the preceding piece."""
    parts = re.split(pattern, text)
    pieces = [parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")
              for i in range(0, len(parts), 2)]
    return [p for p in pieces if p]


def _hard_split(text: str, size: int, count: TokenCounter) -> list[str]:
    """Last resort for text with no boundaries (e.g. a very long unbroken string)."""
    pieces: list[str] = []
    while text:
        step = max(1, int(len(text) * size / max(count(text), 1)))
        piece = text[:step]
        while count(piece) > size and len(piece) > 1:
            piece = piece[: max(1, len(piece) // 2)]
        pieces.append(piece)
        text = text[len(piece):]
    return pieces


def _atomize(text: str, size: int, count: TokenCounter, level: int = 0) -> list[str]:
    """Break text into pieces that each fit the budget, on natural boundaries."""
    if count(text) <= size:
        return [text]
    if level >= len(_SEPARATORS):
        return _hard_split(text, size, count)
    parts = _split_keep(text, _SEPARATORS[level])
    if len(parts) <= 1:
        return _atomize(text, size, count, level + 1)
    atoms: list[str] = []
    for part in parts:
        atoms.extend(_atomize(part, size, count, level + 1) if count(part) > size else [part])
    return atoms


def split_text(text: str, size: int, overlap: int, count: TokenCounter) -> list[str]:
    """Split text into chunks of at most `size` tokens with `overlap` tokens shared."""
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    current: list[tuple[str, int]] = []  # (piece, tokens)
    current_tokens = 0

    for piece in _atomize(text, size, count):
        tokens = count(piece)
        if current and current_tokens + tokens > size:
            chunks.append("".join(p for p, _ in current).strip())
            # Carry the tail of this chunk forward as overlap.
            tail: list[tuple[str, int]] = []
            tail_tokens = 0
            for item in reversed(current):
                if tail_tokens >= overlap:
                    break
                tail.insert(0, item)
                tail_tokens += item[1]
            while tail and tail_tokens + tokens > size:
                tail_tokens -= tail.pop(0)[1]
            current, current_tokens = tail, tail_tokens
        current.append((piece, tokens))
        current_tokens += tokens

    if current:  # always holds at least one new piece, never just overlap
        chunks.append("".join(p for p, _ in current).strip())
    return [c for c in chunks if c]


def chunk_document(
    doc: ExtractedDocument,
    *,
    size: int,
    overlap: int,
    counter: TokenCounter | None = None,
) -> list[Chunk]:
    """Chunk every section of a document, preserving page/section metadata."""
    count = counter or get_token_counter()
    chunks: list[Chunk] = []
    for section in doc.sections:
        for text in split_text(section.text, size, overlap, count):
            chunks.append(
                Chunk(
                    text=text,
                    position=len(chunks),
                    token_count=count(text),
                    page=section.page,
                    section=section.section,
                )
            )
    return chunks
