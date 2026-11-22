"""End-to-end test of the evaluation pipeline using in-memory fakes."""
import csv
import json
import math
from pathlib import Path

from app.clients.cohere_client import LLMResponse
from app.clients.pinecone_client import Match
from app.config import Settings
from evals.configs import EvalConfig
from evals.judge import Judge
from evals.run import run_all
from evals.runner import Backends, evaluate_config, find_failures, ingest_corpus, summarise
from tests.conftest import FakeEmbedder, IdentityReranker, make_pdf

PDF_TEXT = (
    "Standard shipping costs $4.99 and takes 5 to 7 business days. "
    "Returns are accepted within 30 days of delivery. "
    "Warranty claims are answered within 3 business days."
)
GOLDEN = {
    "corpus": ["facts.pdf"],
    "questions": [
        {"id": "q1", "type": "direct", "question": "What does standard shipping cost?", "answerable": True,
         "document": "facts.pdf", "expected_pages": [1], "evidence": ["$4.99"], "key_facts": ["4.99"],
         "reference_answer": "Standard shipping costs $4.99."},
        {"id": "q2", "type": "multi_part", "question": "Shipping cost and return window?", "answerable": True,
         "document": "facts.pdf", "expected_pages": [1], "evidence": ["$4.99", "within 30 days of delivery"],
         "key_facts": ["4.99", "30 days"], "reference_answer": "$4.99 and 30 days."},
        {"id": "q3", "type": "unanswerable", "question": "Do you sell zzzboots?", "answerable": False,
         "document": None, "expected_pages": [], "evidence": [], "key_facts": [],
         "reference_answer": "The documents do not contain this information."},
    ],
}


class NamespacedStore:
    def __init__(self) -> None:
        self.ns: dict[str, dict] = {}
        self.deleted: list[str] = []

    async def upsert(self, records, namespace=None):
        self.ns.setdefault(namespace, {}).update({r.id: r for r in records})

    async def query(self, vector, *, top_k, namespace=None, metadata_filter=None):
        items = list(self.ns.get(namespace, {}).values())
        scored = sorted(((sum(a * b for a, b in zip(vector, r.values)), r) for r in items), key=lambda x: -x[0])
        return [Match(r.id, s, {k: v for k, v in r.metadata.items() if v is not None}) for s, r in scored[:top_k]]

    async def delete_namespace(self, namespace):
        self.ns.pop(namespace, None)
        self.deleted.append(namespace)

    async def wait_until_visible(self, record, namespace=None):
        return True


class EvalLLM:
    """Answers [1] from the context; the judge call is told apart by its schema."""

    def __init__(self, answer_unanswerable: bool = False, judge_says_correct: bool = True) -> None:
        self.answer_unanswerable, self.judge_ok, self.calls = answer_unanswerable, judge_says_correct, 0

    async def chat_json(self, *, system, user, schema=None, model=None):
        self.calls += 1
        if schema and "correct" in schema["properties"]:
            return LLMResponse(json.dumps({"correct": self.judge_ok, "faithful": True, "reason": "ok"}), 50, 5)
        question = user.split("<question>")[1].lower()
        if "zzzboots" in question and not self.answer_unanswerable:
            out = {"answerable": False, "answer": "Not covered by the documents.", "citations": []}
        else:
            out = {"answerable": True, "answer": "Shipping costs $4.99 and returns take 30 days [1].", "citations": [1]}
        return LLMResponse(json.dumps(out), 200, 20)


def backends(llm: EvalLLM, store: NamespacedStore | None = None) -> Backends:
    return Backends(FakeEmbedder(), IdentityReranker(), llm, store or NamespacedStore(), Judge(llm))


def corpus(tmp_path: Path) -> Path:
    folder = tmp_path / "corpus"
    folder.mkdir()
    (folder / "facts.pdf").write_bytes(make_pdf(PDF_TEXT))
    return folder


SETTINGS = Settings(_env_file=None, embedding_dim=64)
CFG = EvalConfig("T", 40, 5, True)


async def test_ingest_corpus_counts_and_isolates_namespaces(tmp_path) -> None:
    store = NamespacedStore()
    stats = await ingest_corpus(corpus(tmp_path), "ns-a", 40, 5, backends(EvalLLM(), store), SETTINGS)
    assert stats["files"] == 1 and stats["chunks"] >= 1
    assert "ns-a" in store.ns and store.deleted == ["ns-a"]  # cleared first


async def test_evaluation_scores_a_working_system(tmp_path) -> None:
    store, llm = NamespacedStore(), EvalLLM()
    b = backends(llm, store)
    await ingest_corpus(corpus(tmp_path), CFG.namespace, 40, 5, b, SETTINGS)
    results = await evaluate_config(CFG, GOLDEN, b, SETTINGS)
    summary = summarise(results, CFG)
    assert summary["n_answerable"] == 2 and summary["n_unanswerable"] == 1 and summary["n_errors"] == 0
    assert summary["retrieval"]["hit@5"] == 1.0 and summary["retrieval"]["recall@5"] == 1.0
    assert summary["answers"]["correct_rate"] == 1.0 and summary["answers"]["faithful_rate"] == 1.0
    assert summary["answers"]["unanswerable_refused_rate"] == 1.0
    assert summary["answers"]["citation_precision"] is not None
    assert math.isclose(summary["answers"]["fact_recall"], 1.0)
    assert find_failures(results) == []
    assert "vector_score" in summary["calibration"] and "rerank_score" in summary["calibration"]


async def test_hallucination_on_unanswerable_and_wrong_answers_are_flagged(tmp_path) -> None:
    b = backends(EvalLLM(answer_unanswerable=True, judge_says_correct=False))
    await ingest_corpus(corpus(tmp_path), CFG.namespace, 40, 5, b, SETTINGS)
    results = await evaluate_config(CFG, GOLDEN, b, SETTINGS)
    issues = {f["id"]: f["issues"] for f in find_failures(results)}
    assert "hallucinated_on_unanswerable" in issues["q3"]
    assert "wrong_answer" in issues["q1"]
    summary = summarise(results, CFG)
    assert summary["answers"]["unanswerable_hallucinated_rate"] == 1.0
    assert summary["answers"]["correct_rate"] == 0.0


async def test_no_judge_mode_skips_judge_calls(tmp_path) -> None:
    llm = EvalLLM()
    b = backends(llm)
    await ingest_corpus(corpus(tmp_path), CFG.namespace, 40, 5, b, SETTINGS)
    before = llm.calls
    results = await evaluate_config(CFG, GOLDEN, b, SETTINGS, use_judge=False)
    assert llm.calls - before == 3  # one generation call per question, none for judging
    assert summarise(results, CFG)["answers"]["correct_rate"] is None


async def test_run_all_writes_every_output_and_cleans_up(tmp_path) -> None:
    store = NamespacedStore()
    b = backends(EvalLLM(), store)
    configs = [EvalConfig("A", 40, 5, False), EvalConfig("B", 40, 5, True), EvalConfig("C", 80, 10, True)]
    out = tmp_path / "out"
    paths = await run_all(b, GOLDEN, configs, corpus(tmp_path), SETTINGS, out, golden_bytes=b"golden")

    for key in ("json", "summary", "csv", "judge_sample"):
        assert paths[key].exists(), key
    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert set(data["configs"]) == {"A", "B", "C"} and data["meta"]["golden_questions"] == 3
    summary_md = paths["summary"].read_text(encoding="utf-8")
    assert "## Winner" in summary_md and "hit@5" in summary_md and "Threshold calibration" in summary_md
    assert paths["winner"].name in {"A", "B", "C"}
    # A and B share a chunking, so only two namespaces were ingested, and all were cleaned up
    assert sorted(set(store.deleted)) == ["eval-c40-o5", "eval-c80-o10"] and not store.ns
    with paths["judge_sample"].open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows and {"human_correct", "human_faithful"} <= set(rows[0]) and rows[0]["human_correct"] == ""
    with paths["csv"].open(encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == 9  # 3 configs x 3 questions


async def test_keep_namespaces_option(tmp_path) -> None:
    store = NamespacedStore()
    await run_all(backends(EvalLLM(), store), GOLDEN, [CFG], corpus(tmp_path), SETTINGS,
                  tmp_path / "out", keep_namespaces=True)
    assert CFG.namespace in store.ns
