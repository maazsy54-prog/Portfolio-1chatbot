"""Measure how much the LLM judge agrees with a human.

1. `python -m evals.run` writes evals/results/<run>/judge_sample.csv (20 judged answers).
2. Fill the `human_correct` and `human_faithful` columns with true/false yourself.
3. Run: python -m evals.judge_agreement evals/results/<run>/judge_sample.csv
"""
import csv
import sys
from pathlib import Path

from evals.metrics import cohens_kappa


def _parse(value: str) -> bool | None:
    value = value.strip().lower()
    return True if value in {"true", "yes", "1", "y"} else False if value in {"false", "no", "0", "n"} else None


def agreement(rows: list[dict[str, str]]) -> dict[str, dict]:
    """Agreement and Cohen's kappa for 'correct' and 'faithful' over human-labelled rows."""
    report = {}
    for prop in ("correct", "faithful"):
        pairs = [(_parse(r[f"judge_{prop}"]), _parse(r[f"human_{prop}"])) for r in rows]
        pairs = [(j, h) for j, h in pairs if j is not None and h is not None]
        judge, human = [j for j, _ in pairs], [h for _, h in pairs]
        report[prop] = {
            "n": len(pairs),
            "agreement": (sum(j == h for j, h in pairs) / len(pairs)) if pairs else None,
            "kappa": cohens_kappa(judge, human),
        }
    return report


if __name__ == "__main__":
    path = Path(sys.argv[1])
    with path.open(encoding="utf-8") as fh:
        report = agreement(list(csv.DictReader(fh)))
    for prop, r in report.items():
        agree = "n/a" if r["agreement"] is None else f"{r['agreement']:.0%}"
        kappa = "undefined (one rater used a single label)" if r["kappa"] is None and r["n"] else (
            "n/a" if r["kappa"] is None else f"{r['kappa']:.2f}")
        print(f"{prop:9} n={r['n']:<3} agreement={agree:<5} kappa={kappa}")
    if all(r["n"] == 0 for r in report.values()):
        print("No labelled rows found: fill in the human_correct / human_faithful columns first.")
