"""Unit tests for evaluation metrics, golden-set validity and judge-agreement maths."""
import json
from pathlib import Path

import pytest

from evals import metrics as m
from evals.judge import parse_verdict
from evals.judge_agreement import agreement
from evals.validate_golden import validate_golden

GOLDEN_PATH = Path(__file__).resolve().parents[2] / "evals" / "golden_set.json"
FAQ_PDF = Path(__file__).resolve().parents[2] / "corpus" / "nimbus_faq.pdf"


def test_normalise_handles_case_whitespace_and_ranges() -> None:
    assert m.normalise("  Takes 5-7\n Business   DAYS ") == "takes 5 to 7 business days"
    assert m.contains("takes 5 to 7 business days", "5-7 business days")
    assert m.contains("Call 1-800-555-0142 now", "1-800-555-0142")  # symmetric, so phone numbers still match


def test_relevance_requires_right_document_and_evidence() -> None:
    assert m.is_relevant("It costs $4.99.", "faq.pdf", "faq.pdf", ["$4.99"])
    assert not m.is_relevant("It costs $4.99.", "other.pdf", "faq.pdf", ["$4.99"])
    assert not m.is_relevant("Nothing here.", "faq.pdf", "faq.pdf", ["$4.99"])
    assert not m.is_relevant("It costs $4.99.", "faq.pdf", None, ["$4.99"])  # unanswerable


def test_hit_rr_and_recall() -> None:
    flags = [False, False, True, False]
    assert (m.hit_at_k(flags, 1), m.hit_at_k(flags, 3)) == (0.0, 1.0)
    assert m.reciprocal_rank(flags) == pytest.approx(1 / 3)
    assert m.reciprocal_rank([False, False]) == 0.0
    texts = ["costs $4.99", "takes 5 to 7 business days", "unrelated"]
    files = ["f.pdf"] * 3
    assert m.recall_at_k(texts, files, "f.pdf", ["$4.99", "5 to 7 business days"], 1) == 0.5
    assert m.recall_at_k(texts, files, "f.pdf", ["$4.99", "5 to 7 business days"], 2) == 1.0
    assert m.recall_at_k(texts, ["x.pdf"] * 3, "f.pdf", ["$4.99"], 3) == 0.0  # wrong document


def test_fact_recall_and_citation_precision() -> None:
    assert m.fact_recall("It takes 5-7 business days and costs $4.99", ["5 to 7", "4.99"]) == 1.0
    assert m.fact_recall("It is quick", ["5 to 7"]) == 0.0
    assert m.citation_precision([True, False]) == 0.5 and m.citation_precision([]) is None


def test_percentile_and_mean() -> None:
    assert m.percentile([1, 2, 3, 4], 50) == 2.5
    assert m.percentile([1, 2, 3, 4, 5], 95) == pytest.approx(4.8)
    assert m.percentile([], 50) is None and m.mean([None, None]) is None and m.mean([1, None, 3]) == 2


def test_cohens_kappa() -> None:
    assert m.cohens_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert m.cohens_kappa([True, True, False, False], [True, False, True, False]) == pytest.approx(0.0)
    assert m.cohens_kappa([True, True], [True, True]) == 1.0
    assert m.cohens_kappa([], []) is None


def test_threshold_calibration_clean_separation() -> None:
    cal = m.calibrate_threshold([0.8, 0.7, 0.6], [0.3, 0.2, 0.1])
    assert cal["balanced_accuracy"] == 1.0 and 0.3 < cal["best_threshold"] < 0.6
    assert cal["keep_all_answerable_threshold"] == 0.6 and cal["unanswerable_rejected_at_keep_all"] == 1.0


def test_threshold_calibration_with_overlap_and_missing_scores() -> None:
    cal = m.calibrate_threshold([0.8, 0.5, 0.4], [0.6, 0.2, None])
    assert cal["balanced_accuracy"] < 1.0
    assert cal["unanswerable_score_range"][0] == -1.0  # no retrieval counts as the lowest score
    assert m.calibrate_threshold([], [0.1]) == {}


def test_judge_verdict_parsing() -> None:
    assert parse_verdict('```json\n{"correct": true, "faithful": false, "reason": "x"}\n```') == (True, False, "x")
    for bad in ["nope", '{"correct": "yes", "faithful": true}', "[]"]:
        with pytest.raises(ValueError):
            parse_verdict(bad)


def test_judge_agreement_report() -> None:
    rows = [
        {"judge_correct": "True", "human_correct": "True", "judge_faithful": "True", "human_faithful": "True"},
        {"judge_correct": "False", "human_correct": "True", "judge_faithful": "True", "human_faithful": "False"},
        {"judge_correct": "True", "human_correct": "", "judge_faithful": "True", "human_faithful": ""},  # unlabelled: ignored
    ]
    report = agreement(rows)
    assert report["correct"]["n"] == 2 and report["correct"]["agreement"] == 0.5


# ---------------------------------------------------------------- golden set

def test_golden_set_structure() -> None:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert validate_golden(golden, None) == []
    qs = golden["questions"]
    assert len(qs) >= 25 and sum(not q["answerable"] for q in qs) >= 5


@pytest.mark.skipif(not FAQ_PDF.exists(), reason="corpus/nimbus_faq.pdf not present")
def test_golden_evidence_matches_the_actual_pdf() -> None:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert validate_golden(golden, FAQ_PDF.parent) == []


def test_validate_golden_catches_problems() -> None:
    bad = {"corpus": [], "questions": [
        {"id": "a", "type": "direct", "question": "q", "answerable": True},
        {"id": "a", "type": "unanswerable", "question": "q", "answerable": False, "evidence": ["x"]},
    ]}
    errors = validate_golden(bad, None)
    text = " ".join(errors)
    assert "at least 25" in text and "duplicate" in text and "must not have evidence" in text
