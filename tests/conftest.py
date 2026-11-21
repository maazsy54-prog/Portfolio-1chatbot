"""Shared fixtures: in-memory stand-ins for Cohere and Pinecone plus a SQLite database."""
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass

import pymupdf
import pytest
from docx import Document as DocxDocument
from fastapi.testclient import TestClient

from app.clients.cohere_client import LLMResponse
from app.clients.pinecone_client import Match
from app.components import Components
from app.config import Settings
from app.db.session import build_engine, build_session_factory
from app.errors import ServiceUnavailableError
from app.main import create_app

DIM = 64


def _vec(text: str) -> list[float]:
    """Deterministic bag-of-words embedding so similar wording gives similar vectors."""
    v = [0.0] * DIM
    for word in re.findall(r"\w+", text.lower()):
        v[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls, self.fail = 0, False

    async def embed_documents(self, texts):
        self.calls += 1
        if self.fail:
            raise ServiceUnavailableError("Embedding service is temporarily unavailable.")
        return [_vec(t) for t in texts]

    async def embed_query(self, text):
        self.calls += 1
        if self.fail:
            raise ServiceUnavailableError("Embedding service is temporarily unavailable.")
        return _vec(text)


class MemoryStore:
    """In-memory vector store with the same interface as VectorStore."""

    def __init__(self) -> None:
        self.data: dict[str, dict] = {}
        self.ignore_deletes = False  # simulates a vector delete that has not taken effect yet
        self.fail_visible = False

    async def upsert(self, records, namespace=None):
        for r in records:
            self.data[r.id] = r

    async def query(self, vector, *, top_k, namespace=None, metadata_filter=None):
        items = list(self.data.values())
        if metadata_filter:
            allowed = set(metadata_filter["document_id"]["$in"])
            items = [r for r in items if r.metadata.get("document_id") in allowed]
        scored = sorted(((sum(a * b for a, b in zip(vector, r.values)), r) for r in items),
                        key=lambda x: -x[0])
        return [Match(id=r.id, score=s, metadata={k: v for k, v in r.metadata.items() if v is not None})
                for s, r in scored[:top_k]]

    async def delete(self, ids, namespace=None):
        if not self.ignore_deletes:
            for i in ids:
                self.data.pop(i, None)

    async def wait_until_visible(self, record, namespace=None):
        if self.fail_visible:
            raise ServiceUnavailableError("Vector database is temporarily unavailable.")
        return True


class IdentityReranker:
    async def rerank(self, query, documents, top_n):
        return [(i, 1.0 - 0.01 * i) for i in range(min(top_n, len(documents)))]


class ScriptedLLM:
    """Answers from excerpt [1], except for questions containing 'unknowable'."""

    def __init__(self) -> None:
        self.calls = 0

    async def chat_json(self, *, system, user):
        self.calls += 1
        question = user.split("<question>")[1].lower()
        if "unknowable" in question:
            out = {"answerable": False, "answer": "The documents do not cover that.", "citations": []}
        else:
            out = {"answerable": True, "answer": "According to the documents [1].", "citations": [1]}
        return LLMResponse(json.dumps(out), input_tokens=300, output_tokens=20)


@dataclass
class Env:
    client: TestClient
    embedder: FakeEmbedder
    store: MemoryStore
    llm: ScriptedLLM
    settings: Settings


@pytest.fixture
def make_env(tmp_path):
    """Factory: build an isolated app + fakes; pass Settings overrides as keyword arguments."""
    opened: list = []

    def _make(**overrides) -> Env:
        options = dict(
            _env_file=None, embedding_dim=DIM, max_upload_mb=1, rerank_enabled=True,
            database_url=f"sqlite+aiosqlite:///{tmp_path / f'test{len(opened)}.db'}",
            log_level="WARNING", rate_limit_ask_per_minute=1000, rate_limit_upload_per_minute=1000,
        )
        options.update(overrides)
        settings = Settings(**options)
        engine = build_engine(settings.database_url.get_secret_value())
        embedder, store, llm = FakeEmbedder(), MemoryStore(), ScriptedLLM()
        components = Components(engine, build_session_factory(engine), embedder,
                                IdentityReranker(), llm, store)
        client = TestClient(create_app(settings, components))
        client.__enter__()
        opened.append(client)
        return Env(client, embedder, store, llm, settings)

    yield _make
    for client in reversed(opened):
        client.__exit__(None, None, None)


@pytest.fixture
def env(make_env):
    return make_env()


# ---------- file builders ----------

def make_pdf(*pages: str) -> bytes:
    pdf = pymupdf.open()
    for text in pages:
        page = pdf.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 750), text, fontsize=11)
        else:
            page.draw_rect(pymupdf.Rect(50, 50, 200, 200))  # image-only page
    return pdf.tobytes()


def make_docx(heading: str, body: str) -> bytes:
    doc = DocxDocument()
    doc.add_heading(heading, 1)
    doc.add_paragraph(body)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


SHIPPING_PDF = make_pdf(
    "Standard shipping takes five to seven business days within the country. "
    "Express shipping takes two business days and costs extra."
)
RETURNS_DOCX = make_docx(
    "Returns", "Returns are accepted within thirty days of delivery with the original receipt."
)
PDF_MIME, DOCX_MIME = "application/pdf", (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)


def upload(env: Env, name: str, data: bytes, mime: str = PDF_MIME):
    return env.client.post("/documents", files={"file": (name, data, mime)})
