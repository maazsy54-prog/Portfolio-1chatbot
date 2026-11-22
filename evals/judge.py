"""LLM-as-a-judge for answer correctness and faithfulness.

One call judges both properties. The prompt is short and fully visible here (unlike a
framework's hidden prompts), and the judge is validated against human labels
(see evals/judge_agreement.py).
"""
import json
import re
from dataclasses import dataclass
from typing import Any

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean"},
        "faithful": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["correct", "faithful", "reason"],
}

JUDGE_SYSTEM = (
    "You are a strict, impartial evaluator of a document question-answering system. "
    "Judge only what is asked and reply with JSON only."
)


@dataclass(frozen=True)
class JudgeVerdict:
    correct: bool
    faithful: bool
    reason: str
    input_tokens: int = 0
    output_tokens: int = 0


def build_judge_prompt(question: str, reference: str, excerpts: str, answer: str) -> str:
    return (
        f"<question>\n{question}\n</question>\n\n"
        f"<reference_answer>\n{reference}\n</reference_answer>\n\n"
        f"<retrieved_excerpts>\n{excerpts}\n</retrieved_excerpts>\n\n"
        f"<candidate_answer>\n{answer}\n</candidate_answer>\n\n"
        "Evaluate the candidate answer:\n"
        "1. correct: true if it states the key information of the reference answer and does not "
        "contradict it. Different wording and extra correct detail are fine; a missing key fact "
        "or a wrong number makes it false.\n"
        "2. faithful: true if every factual claim in the candidate answer is supported by the "
        "retrieved excerpts. Ignore [n] citation markers and any sentence beginning with "
        "\"General knowledge (not from your documents):\". Unsupported specifics make it false.\n"
        "Reply with JSON: {\"correct\": boolean, \"faithful\": boolean, \"reason\": \"one short sentence\"}"
    )


def parse_verdict(raw: str) -> tuple[bool, bool, str]:
    """Parse the judge's JSON; raises ValueError if malformed."""
    data = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip()))
    if not isinstance(data, dict) or not isinstance(data.get("correct"), bool) \
            or not isinstance(data.get("faithful"), bool):
        raise ValueError("judge reply missing boolean fields")
    return data["correct"], data["faithful"], str(data.get("reason", ""))[:300]


class Judge:
    """Wraps an LLM client; retries once on a malformed verdict."""

    def __init__(self, llm: Any, model: str | None = None) -> None:
        self._llm = llm
        self._model = model or None

    async def evaluate(self, question: str, reference: str, excerpts: str, answer: str) -> JudgeVerdict:
        prompt = build_judge_prompt(question, reference, excerpts, answer)
        tokens_in = tokens_out = 0
        error = ""
        for _ in range(2):
            response = await self._llm.chat_json(
                system=JUDGE_SYSTEM, user=prompt, schema=JUDGE_SCHEMA, model=self._model
            )
            tokens_in += response.input_tokens
            tokens_out += response.output_tokens
            try:
                correct, faithful, reason = parse_verdict(response.text)
                return JudgeVerdict(correct, faithful, reason, tokens_in, tokens_out)
            except ValueError as exc:
                error = str(exc)
        raise ValueError(f"judge returned an invalid verdict twice: {error}")
