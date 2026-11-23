"""Pure metric functions for the evaluation (no network, fully unit-tested)."""
import math
import re
import unicodedata
from collections.abc import Sequence


def normalise(text: str) -> str:
    """Lower-case, unify unicode, collapse whitespace, and write '5-7' as '5 to 7'.

    Applied to both the evidence phrases and the text they are searched in, so
    layout differences (line wraps, dashes) never decide a match.
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"(\d)\s*[-\u2013\u2014]\s*(\d)", r"\1 to \2", text)
    return re.sub(r"\s+", " ", text).strip()


def contains(text: str, phrase: str) -> bool:
    return normalise(phrase) in normalise(text)


def is_relevant(chunk_text: str, chunk_filename: str, document: str | None, evidence: Sequence[str]) -> bool:
    """A chunk is relevant if it comes from the labelled document and holds any evidence phrase."""
    if not document or chunk_filename != document:
        return False
    return any(contains(chunk_text, e) for e in evidence)


def hit_at_k(flags: Sequence[bool], k: int) -> float:
    """1.0 if any of the top-k chunks is relevant."""
    return 1.0 if any(flags[:k]) else 0.0


def reciprocal_rank(flags: Sequence[bool]) -> float:
    """1/rank of the first relevant chunk, 0 if none."""
    for rank, flag in enumerate(flags, start=1):
        if flag:
            return 1.0 / rank
    return 0.0


def recall_at_k(chunk_texts: Sequence[str], chunk_files: Sequence[str],
                document: str | None, evidence: Sequence[str], k: int) -> float:
    """Fraction of the evidence items found within the top-k chunks of the labelled document.

    For multi-part questions this is the share of required facts that were retrieved.
    """
    if not evidence or not document:
        return 0.0
    top = [t for t, f in zip(chunk_texts[:k], chunk_files[:k]) if f == document]
    return sum(any(contains(t, e) for t in top) for e in evidence) / len(evidence)


def fact_recall(answer: str, key_facts: Sequence[str]) -> float:
    """Share of key facts that appear verbatim (normalised) in the answer. A lexical lower bound."""
    if not key_facts:
        return 0.0
    return sum(contains(answer, f) for f in key_facts) / len(key_facts)


def citation_precision(cited_flags: Sequence[bool]) -> float | None:
    """Share of cited chunks that are relevant; None if nothing was cited."""
    return sum(cited_flags) / len(cited_flags) if cited_flags else None


def mean(values: Sequence[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def percentile(values: Sequence[float], p: float) -> float | None:
    """Linear-interpolated percentile (p in 0..100), like numpy's default."""
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def cohens_kappa(a: Sequence[bool], b: Sequence[bool]) -> float | None:
    """Agreement between two binary raters, corrected for chance. None if undefined."""
    if len(a) != len(b) or not a:
        return None
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else None
    return (observed - expected) / (1 - expected)


def calibrate_threshold(answerable: Sequence[float | None], unanswerable: Sequence[float | None]) -> dict:
    """Pick a score threshold separating answerable from unanswerable questions.

    A question is 'kept' (treated as answerable) if its top score >= threshold.
    Reports the balanced-accuracy optimum, and the strictest threshold that still
    keeps every answerable question (no false refusals) with what it rejects.
    """
    a = [-1.0 if s is None else s for s in answerable]
    u = [-1.0 if s is None else s for s in unanswerable]
    if not a or not u:
        return {}
    candidates = sorted(set(a + u))
    points = [candidates[0] - 1e-6] + [(x + y) / 2 for x, y in zip(candidates, candidates[1:])] + [candidates[-1] + 1e-6]

    def stats(t: float) -> tuple[float, float, float]:
        kept = sum(s >= t for s in a) / len(a)
        rejected = sum(s < t for s in u) / len(u)
        return (kept + rejected) / 2, kept, rejected

    best = max(points, key=lambda t: (stats(t)[0], stats(t)[1], -t))
    ba, kept, rejected = stats(best)
    keep_all = min(a)
    _, _, rejected_keep_all = stats(keep_all)
    return {
        "n_answerable": len(a), "n_unanswerable": len(u),
        "best_threshold": round(best, 4), "balanced_accuracy": round(ba, 4),
        "answerable_kept": round(kept, 4), "unanswerable_rejected": round(rejected, 4),
        "keep_all_answerable_threshold": round(keep_all, 4),
        "unanswerable_rejected_at_keep_all": round(rejected_keep_all, 4),
        "answerable_score_range": [round(min(a), 4), round(max(a), 4)],
        "unanswerable_score_range": [round(min(u), 4), round(max(u), 4)],
    }
