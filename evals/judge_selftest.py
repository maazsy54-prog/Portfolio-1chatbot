"""Judge self-test: feed the judge answers whose correct verdict is known in advance.

For each sampled golden question the judge sees three candidate answers over the same excerpt
(the reference answer itself):
  correct            the reference answer            -> must be judged correct AND faithful
  wrong_fact         numbers altered / yes<->no      -> must be judged NOT correct and NOT faithful
  unsupported_claim  reference + an invented sentence -> must be judged NOT faithful
This measures whether the judge is lenient (approves wrong answers) or harsh (rejects right ones)
without needing human labels. Run:  python -m evals.judge_selftest --n 16
"""
import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any

from evals.judge import Judge

EXTRA_CLAIM = "Every customer also receives a free tent with their first order."


def perturb_numbers(text: str) -> str | None:
    """Change every number so the answer is factually wrong; None if there are no numbers."""
    if not re.search(r"\d", text):
        return None
    return re.sub(r"\d+(?:\.\d+)?", lambda m: str(int(float(m.group())) + 3), text)


def flip_polarity(text: str) -> str | None:
    """Turn a leading Yes into No (or No into Yes); None if it starts with neither."""
    if re.match(r"\s*Yes\b", text):
        return re.sub(r"\bYes\b", "No", text, count=1)
    if re.match(r"\s*No\b", text):
        return re.sub(r"\bNo\b", "Yes", text, count=1)
    return None


def make_controls(question: dict) -> list[dict]:
    """Candidate answers with known expected verdicts for one golden question."""
    ref = question["reference_answer"]
    controls = [{"kind": "correct", "answer": ref, "expect_correct": True, "expect_faithful": True}]
    wrong = perturb_numbers(ref) or flip_polarity(ref)
    if wrong and wrong != ref:
        controls.append({"kind": "wrong_fact", "answer": wrong,
                         "expect_correct": False, "expect_faithful": False})
    controls.append({"kind": "unsupported_claim", "answer": f"{ref} {EXTRA_CLAIM}",
                     "expect_correct": None, "expect_faithful": False})
    return controls


def sample_questions(golden: dict, n: int) -> list[dict]:
    """An evenly spread, deterministic sample of answerable questions."""
    answerable = [q for q in golden["questions"] if q["answerable"]]
    step = max(1, len(answerable) // n)
    return answerable[::step][:n]


async def run_selftest(judge: Judge, golden: dict, n: int = 16, concurrency: int = 3) -> dict[str, Any]:
    """Judge every control answer and score the judge against the known verdicts."""
    gate = asyncio.Semaphore(concurrency)
    jobs = [(q, c) for q in sample_questions(golden, n) for c in make_controls(q)]

    async def one(q: dict, c: dict) -> dict:
        async with gate:
            try:
                v = await judge.evaluate(q["question"], q["reference_answer"], q["reference_answer"], c["answer"])
            except ValueError as exc:
                return {"id": q["id"], **c, "error": str(exc)}
            return {"id": q["id"], **c, "judge_correct": v.correct, "judge_faithful": v.faithful, "error": None}

    rows = await asyncio.gather(*(one(q, c) for q, c in jobs))
    summary: dict[str, Any] = {"n_questions": len({r["id"] for r in rows}), "by_kind": {}}
    for kind in ("correct", "wrong_fact", "unsupported_claim"):
        rs = [r for r in rows if r["kind"] == kind and not r["error"]]
        entry: dict[str, Any] = {"n": len(rs), "errors": sum(1 for r in rows if r["kind"] == kind and r["error"])}
        if rs and rs[0]["expect_correct"] is not None:
            entry["correct_verdict_ok"] = sum(r["judge_correct"] == r["expect_correct"] for r in rs) / len(rs)
        if rs:
            entry["faithful_verdict_ok"] = sum(r["judge_faithful"] == r["expect_faithful"] for r in rs) / len(rs)
        summary["by_kind"][kind] = entry
    summary["rows"] = rows
    return summary


def render(summary: dict) -> str:
    kinds = summary["by_kind"]
    def pct(x): return "n/a" if x is None else f"{x:.0%}"
    lines = [f"# Judge self-test ({summary['n_questions']} questions)", "",
             "| Control answer | n | judged correctly: correct flag | judged correctly: faithful flag |", "|---|---|---|---|"]
    meaning = {"correct": "right answer (judge should accept)", "wrong_fact": "wrong numbers/polarity (judge should reject)",
               "unsupported_claim": "right answer + invented claim (judge should flag as unfaithful)"}
    for k, label in meaning.items():
        e = kinds.get(k, {})
        lines.append(f"| {label} | {e.get('n', 0)} | {pct(e.get('correct_verdict_ok'))} | {pct(e.get('faithful_verdict_ok'))} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the LLM judge against controlled answers.")
    parser.add_argument("--golden", default="evals/golden_set.json", type=Path)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--llm-rpm", type=int, default=18, help="chat calls per minute (0 = unlimited)")
    parser.add_argument("--output", default="evals/results/judge_selftest", type=Path)
    args = parser.parse_args()

    from app.clients.llm import make_llm
    from app.config import get_settings

    settings = get_settings().model_copy(update={"max_retries": 6, "retry_backoff_seconds": 2.0,
                                                 "http_timeout_seconds": 90.0,
                                                 "llm_calls_per_minute": args.llm_rpm})
    judge = Judge(make_llm(settings), settings.judge_model or None)
    golden = json.loads(args.golden.read_text(encoding="utf-8"))
    summary = asyncio.run(run_selftest(judge, golden, args.n))
    text = render(summary)
    print(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".md").write_text(text, encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved to {args.output}.md / .json")


if __name__ == "__main__":
    main()
