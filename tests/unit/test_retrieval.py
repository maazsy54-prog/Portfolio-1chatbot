"""Unit tests for retrieval (embedding, search, rerank, thresholds) using fakes."""
from app.clients.pinecone_client import Match
from app.config import Settings
from app.retrieval import retrieve


def make_settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def match(i: int, score: float, **md) -> Match:
    base = {"document_id": "D1", "filename": "f.pdf", "position": float(i),
            "text": f"text {i}", "page": 2.0}
    return Match(id=f"D1:{i}", score=score, metadata={**base, **md})


class FakeEmbedder:
    async def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2]


class FakeStore:
    def __init__(self, matches: list[Match]) -> None:
        self.matches, self.last_filter, self.last_top_k = matches, None, None

    async def query(self, vector, *, top_k, namespace=None, metadata_filter=None):
        self.last_filter, self.last_top_k = metadata_filter, top_k
        return self.matches[:top_k]


class FakeReranker:
    """Reverses candidate order and scores them descending from 0.9."""

    def __init__(self) -> None:
        self.called = False

    async def rerank(self, query, documents, top_n):
        self.called = True
        order = list(range(len(documents)))[::-1][:top_n]
        return [(i, 0.9 - 0.1 * rank) for rank, i in enumerate(order)]


async def run(matches, **settings):
    s = make_settings(**settings)
    rr = FakeReranker()
    result = await retrieve("q?", embedder=FakeEmbedder(), store=FakeStore(matches),
                            reranker=rr, settings=s)
    return result, rr


async def test_without_rerank_applies_vector_threshold_and_top_n() -> None:
    matches = [match(i, s) for i, s in enumerate([0.9, 0.8, 0.5, 0.3, 0.2, 0.1])]
    result, rr = await run(matches, rerank_enabled=False, similarity_threshold=0.4, context_top_n=2)
    assert not rr.called and not result.reranked
    assert [c.id for c in result.chunks] == ["D1:0", "D1:1"]
    assert result.fetched == 6  # counted before filtering


async def test_rerank_reorders_and_sets_rerank_score() -> None:
    matches = [match(i, 0.5) for i in range(4)]
    result, rr = await run(matches, rerank_enabled=True, context_top_n=3, rerank_threshold=0.0)
    assert rr.called and result.reranked
    assert [c.id for c in result.chunks] == ["D1:3", "D1:2", "D1:1"]
    assert result.chunks[0].rerank_score == 0.9
    assert result.chunks[0].score == 0.5  # vector score preserved for display


class SteepReranker:
    """One strong chunk followed by weak ones, like a multi-part question's second passage."""

    async def rerank(self, query, documents, top_n):
        scores = [0.9, 0.05, 0.04, 0.03]
        return [(i, scores[i]) for i in range(min(top_n, len(documents)))]


async def test_threshold_judges_the_best_chunk_not_each_chunk() -> None:
    s = make_settings(rerank_enabled=True, rerank_threshold=0.5, context_top_n=3)
    result = await retrieve("q?", embedder=FakeEmbedder(), store=FakeStore([match(i, 0.5) for i in range(4)]),
                            reranker=SteepReranker(), settings=s)
    assert len(result.chunks) == 3, "weak supporting chunks must stay when the best chunk passes"


async def test_best_chunk_below_threshold_refuses_everything() -> None:
    s = make_settings(rerank_enabled=True, rerank_threshold=0.95, context_top_n=3)
    result = await retrieve("q?", embedder=FakeEmbedder(), store=FakeStore([match(i, 0.5) for i in range(4)]),
                            reranker=SteepReranker(), settings=s)
    assert result.chunks == [] and result.fetched == 4


async def test_vector_threshold_also_uses_the_best_chunk_only() -> None:
    matches = [match(0, 0.9), match(1, 0.1), match(2, 0.05)]
    kept, _ = await run(matches, rerank_enabled=False, similarity_threshold=0.5)
    assert [c.id for c in kept.chunks] == ["D1:0", "D1:1", "D1:2"]
    refused, _ = await run([match(0, 0.4), match(1, 0.3)], rerank_enabled=False, similarity_threshold=0.5)
    assert refused.chunks == []


async def test_rerank_threshold_can_empty_the_result() -> None:
    result, _ = await run([match(i, 0.5) for i in range(3)], rerank_threshold=0.95)
    assert result.chunks == [] and result.fetched == 3


async def test_empty_index_gives_no_chunks() -> None:
    result, rr = await run([])
    assert result.chunks == [] and result.fetched == 0
    assert not rr.called and result.top_score is None


async def test_metadata_converted_and_missing_page_is_none() -> None:
    result, _ = await run([match(0, 0.7, page=None, section="Returns")], rerank_enabled=False)
    chunk = result.chunks[0]
    assert chunk.page is None and chunk.section == "Returns" and chunk.position == 0
    result2, _ = await run([match(1, 0.7)], rerank_enabled=False)
    assert result2.chunks[0].page == 2 and isinstance(result2.chunks[0].page, int)


async def test_document_filter_is_passed_to_vector_store() -> None:
    store = FakeStore([match(0, 0.9)])
    await retrieve("q?", embedder=FakeEmbedder(), store=store, reranker=FakeReranker(),
                   settings=make_settings(rerank_enabled=False), document_ids=["D1", "D2"])
    assert store.last_filter == {"document_id": {"$in": ["D1", "D2"]}}
    assert store.last_top_k == 20
