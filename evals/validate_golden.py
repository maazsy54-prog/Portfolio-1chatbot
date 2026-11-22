"""Validate the golden set against the corpus.  Run: python -m evals.validate_golden"""
import json
import sys
from pathlib import Path

from app.ingestion.extract import extract_text
from evals.metrics import contains

REQUIRED_TYPES = {"direct", "reworded", "multi_part", "unanswerable"}


def validate_golden(golden: dict, corpus_dir: Path | None) -> list[str]:
    """Return a list of problems (empty = valid)."""
    errors: list[str] = []
    qs = golden.get("questions", [])
    if len(qs) < 25:
        errors.append(f"need at least 25 questions, have {len(qs)}")
    if sum(not q["answerable"] for q in qs) < 5:
        errors.append("need at least 5 unanswerable questions")
    missing = REQUIRED_TYPES - {q["type"] for q in qs}
    if missing:
        errors.append(f"missing question types: {sorted(missing)}")
    ids = [q["id"] for q in qs]
    if len(set(ids)) != len(ids):
        errors.append("duplicate question ids")

    pages: dict[str, dict[int, str]] = {}
    if corpus_dir:
        for name in golden.get("corpus", []):
            path = corpus_dir / name
            if path.exists() and path.suffix.lower() == ".pdf":
                doc = extract_text("pdf", path.read_bytes())
                pages[name] = {s.page: s.text for s in doc.sections}

    for q in qs:
        tag = q["id"]
        if q["answerable"]:
            if not (q.get("document") and q.get("evidence") and q.get("reference_answer")):
                errors.append(f"{tag}: answerable question needs document, evidence and reference_answer")
            doc_pages = pages.get(q.get("document", ""))
            if doc_pages:
                for e in q.get("evidence", []):
                    found = [p for p, t in doc_pages.items() if contains(t, e)]
                    if not found:
                        errors.append(f"{tag}: evidence not found in {q['document']}: {e!r}")
                    elif not set(found) & set(q.get("expected_pages", [])):
                        errors.append(f"{tag}: evidence {e!r} is on pages {found}, not {q['expected_pages']}")
        elif q.get("evidence"):
            errors.append(f"{tag}: unanswerable question must not have evidence")
    return errors


if __name__ == "__main__":
    golden = json.loads(Path("evals/golden_set.json").read_text(encoding="utf-8"))
    problems = validate_golden(golden, Path("corpus"))
    print("golden set OK" if not problems else "\n".join(problems))
    sys.exit(1 if problems else 0)
