"""The configurations compared by the evaluation.

Chunking is fixed at ingestion time, so each distinct (size, overlap) gets its own
Pinecone namespace; reranking and thresholds are applied at query time.
Edit this list to add experiments (e.g. calibrated thresholds after a first run).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class EvalConfig:
    name: str
    chunk_size: int
    chunk_overlap: int
    rerank: bool
    similarity_threshold: float = 0.0  # applies when rerank is off
    rerank_threshold: float = 0.0  # applies when rerank is on
    description: str = ""

    @property
    def namespace(self) -> str:
        return f"eval-c{self.chunk_size}-o{self.chunk_overlap}"


CONFIGS: list[EvalConfig] = [
    EvalConfig("A", 400, 50, False, description="baseline: 400/50 tokens, vector search only"),
    EvalConfig("B", 400, 50, True, description="400/50 + Cohere rerank (isolates the effect of reranking)"),
    EvalConfig("C", 200, 30, True, description="small chunks + rerank"),
    EvalConfig("D", 800, 100, True, description="large chunks + rerank"),
    EvalConfig("E", 400, 50, True, rerank_threshold=0.10,
               description="B + refuse when the best rerank score is below 0.10 (calibrated from B's scores)"),
]
