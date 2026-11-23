"""The judge self-test must expose a lenient judge and pass a sound one."""
import json
import re

from app.clients.cohere_client import LLMResponse
from evals.judge import Judge
from evals.judge_selftest import (
    EXTRA_CLAIM, flip_polarity, make_controls, perturb_numbers, render, run_selftest, sample_questions,
)

GOLDEN = {"questions": [
    {"id": f"q{i}", "answerable": True, "question": f"Question {i}?",
     "reference_answer": ref}
    for i, ref in enumerate([
        "Standard shipping costs $4.99 and takes 5 to 7 business days.",
        "Yes. Exchanges are free within 30 days.",
        "No. The warranty applies to the original purchaser only.",
        "Returns need the original receipt.",
    ] * 4)
] + [{"id": "u1", "answerable": False, "question": "?", "reference_answer": "n/a"}]}


class SoundJudgeLLM:
    """Approves only answers whose numbers/polarity match the reference and with no extra claim."""

    async def chat_json(self, *, system, user, schema=None, model=None):
        ref = re.search(r"<reference_answer>\n(.*?)\n</reference_answer>", user, re.S).group(1)
        cand = re.search(r"<candidate_answer>\n(.*?)\n</candidate_answer>", user, re.S).group(1)
        numbers_ok = re.findall(r"\d+(?:\.\d+)?", cand) == re.findall(r"\d+(?:\.\d+)?", ref[: len(cand)]) \
            or set(re.findall(r"\d+(?:\.\d+)?", cand)) <= set(re.findall(r"\d+(?:\.\d+)?", ref))
        polarity_ok = cand.split()[0:1] == ref.split()[0:1] or not ref.startswith(("Yes", "No"))
        clean = EXTRA_CLAIM not in cand
        out = {"correct": numbers_ok and polarity_ok, "faithful": numbers_ok and polarity_ok and clean, "reason": "ok"}
        return LLMResponse(json.dumps(out), 10, 2)


class LenientJudgeLLM:
    async def chat_json(self, *, system, user, schema=None, model=None):
        return LLMResponse(json.dumps({"correct": True, "faithful": True, "reason": "looks fine"}), 10, 2)


def test_perturbations() -> None:
    assert perturb_numbers("costs $4.99 in 5 days") == "costs $7 in 8 days"
    assert perturb_numbers("no digits here") is None
    assert flip_polarity("Yes. Free.") == "No. Free." and flip_polarity("No way") == "Yes way"
    assert flip_polarity("Maybe") is None


def test_controls_have_known_expectations() -> None:
    controls = {c["kind"]: c for c in make_controls(GOLDEN["questions"][0])}
    assert controls["correct"]["expect_correct"] is True
    assert controls["wrong_fact"]["answer"] != GOLDEN["questions"][0]["reference_answer"]
    assert controls["unsupported_claim"]["expect_faithful"] is False and EXTRA_CLAIM in controls["unsupported_claim"]["answer"]


def test_question_without_numbers_or_polarity_gets_no_wrong_fact_control() -> None:
    kinds = {c["kind"] for c in make_controls(GOLDEN["questions"][3])}
    assert kinds == {"correct", "unsupported_claim"}


def test_sampling_is_deterministic_and_answerable_only() -> None:
    sample = sample_questions(GOLDEN, 6)
    assert len(sample) <= 6 and all(q["answerable"] for q in sample)
    assert sample == sample_questions(GOLDEN, 6)


async def test_sound_judge_scores_perfectly() -> None:
    summary = await run_selftest(Judge(SoundJudgeLLM()), GOLDEN, n=8)
    kinds = summary["by_kind"]
    assert kinds["correct"]["correct_verdict_ok"] == 1.0 and kinds["correct"]["faithful_verdict_ok"] == 1.0
    assert kinds["wrong_fact"]["correct_verdict_ok"] == 1.0
    assert kinds["unsupported_claim"]["faithful_verdict_ok"] == 1.0


async def test_lenient_judge_is_exposed() -> None:
    summary = await run_selftest(Judge(LenientJudgeLLM()), GOLDEN, n=8)
    kinds = summary["by_kind"]
    assert kinds["correct"]["correct_verdict_ok"] == 1.0          # accepts right answers...
    assert kinds["wrong_fact"]["correct_verdict_ok"] == 0.0       # ...but also accepts wrong ones
    assert kinds["unsupported_claim"]["faithful_verdict_ok"] == 0.0
    assert "0%" in render(summary)
