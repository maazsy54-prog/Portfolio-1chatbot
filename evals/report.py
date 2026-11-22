"""Write evaluation outputs: results.json, summary.md, per_question.csv, judge_sample.csv."""
import csv
import json
import random
from pathlib import Path

from evals.configs import EvalConfig


def _f(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def pick_winner(summaries: dict[str, dict]) -> str:
    """Documented rule: best correct_rate, then hit@5, then MRR, then lower p95 latency."""
    def key(name: str) -> tuple:
        s = summaries[name]
        return (s["answers"]["correct_rate"] or 0, s["retrieval"]["hit@5"] or 0,
                s["retrieval"]["mrr"] or 0, -(s["latency_s"]["p95"] or 0))
    return max(summaries, key=key)


def render_summary(meta: dict, configs: list[EvalConfig], summaries: dict[str, dict],
                   failures: dict[str, list[dict]]) -> str:
    lines = [f"# Evaluation results ({meta['timestamp']})", "",
             f"Golden set: {meta['golden_questions']} questions (sha256 `{meta['golden_sha256'][:12]}`). "
             f"Embedding `{meta['embedding_model']}` ({meta['embedding_dim']}d), LLM `{meta['llm_model']}`, "
             f"judge `{meta['judge_model']}`, rerank `{meta['rerank_model']}`. Commit `{meta['git_commit']}`.", "",
             "## Configurations", "", "| Config | Chunk size / overlap | Rerank | Description |", "|---|---|---|---|"]
    for c in configs:
        lines.append(f"| {c.name} | {c.chunk_size}/{c.chunk_overlap} | {'yes' if c.rerank else 'no'} | {c.description} |")

    names = [c.name for c in configs if c.name in summaries]
    def row(label: str, getter) -> str:
        return f"| {label} | " + " | ".join(getter(summaries[n]) for n in names) + " |"
    header = "| Metric | " + " | ".join(names) + " |\n|---|" + "---|" * len(names)

    lines += ["", "## Retrieval (answerable questions)", "", header]
    for label, key in [("hit@1", "hit@1"), ("hit@3", "hit@3"), ("hit@5", "hit@5"),
                       ("recall@3", "recall@3"), ("recall@5", "recall@5"), ("MRR", "mrr"),
                       ("top-1 source page correct", "source_page_acc")]:
        lines.append(row(label, lambda s, k=key: _f(s["retrieval"][k])))

    lines += ["", "## Answers", "", header]
    for label, key in [("correct (judge)", "correct_rate"), ("faithful (judge)", "faithful_rate"),
                       ("key-fact recall (lexical)", "fact_recall"), ("citation precision", "citation_precision"),
                       ("false refusals (answerable)", "false_refusal_rate"),
                       ("unanswerable correctly refused", "unanswerable_refused_rate"),
                       ("unanswerable hallucinated", "unanswerable_hallucinated_rate")]:
        lines.append(row(label, lambda s, k=key: _f(s["answers"][k])))

    lines += ["", "## Cost and speed", "", header,
              row("latency p50 (s)", lambda s: _f(s["latency_s"]["p50"], 2)),
              row("latency p95 (s)", lambda s: _f(s["latency_s"]["p95"], 2)),
              row("avg generation tokens in", lambda s: _f(s["tokens_per_question"]["generation_in"], 0)),
              row("avg generation tokens out", lambda s: _f(s["tokens_per_question"]["generation_out"], 0))]

    lines += ["", "## By question type (hit@5 / correct)", "", header]
    types = sorted({t for n in names for t in summaries[n]["by_type"]})
    for t in types:
        lines.append(row(t, lambda s, t=t: (
            f"{_f(s['by_type'][t]['hit@5'], 2)} / {_f(s['by_type'][t]['correct'], 2)}" if t in s["by_type"] else "-")))

    winner = pick_winner({n: summaries[n] for n in names})
    lines += ["", f"## Winner: **{winner}**", "",
              "Rule: highest judge-correct rate, then hit@5, then MRR, then lower p95 latency."]

    lines += ["", "## Threshold calibration (top-1 score, answerable vs unanswerable)", ""]
    for n in names:
        cal = summaries[n].get("calibration", {})
        for kind, c in cal.items():
            if c:
                lines.append(
                    f"- **{n} / {kind}**: best threshold {c['best_threshold']} "
                    f"(balanced accuracy {c['balanced_accuracy']}); keeping every answerable question needs "
                    f"{c['keep_all_answerable_threshold']} and rejects {c['unanswerable_rejected_at_keep_all']:.0%} "
                    f"of unanswerable. Answerable range {c['answerable_score_range']}, "
                    f"unanswerable range {c['unanswerable_score_range']}.")

    lines += ["", "## Failures", ""]
    for n in names:
        lines.append(f"### Config {n}: {len(failures[n])} question(s) with issues")
        for f in failures[n][:25]:
            lines.append(f"- `{f['id']}` ({f['type']}) {', '.join(f['issues'])}: {f['question']}")
        lines.append("")
    return "\n".join(lines)


def write_outputs(out_dir: Path, meta: dict, configs: list[EvalConfig],
                  all_results: dict[str, list[dict]], summaries: dict[str, dict],
                  failures: dict[str, list[dict]]) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"json": out_dir / "results.json", "summary": out_dir / "summary.md",
             "csv": out_dir / "per_question.csv", "judge_sample": out_dir / "judge_sample.csv"}

    paths["json"].write_text(json.dumps({
        "meta": meta,
        "configs": {c.name: {"config": c.__dict__, "summary": summaries[c.name],
                             "failures": failures[c.name], "results": all_results[c.name]}
                    for c in configs if c.name in summaries},
    }, indent=2, default=str), encoding="utf-8")
    paths["summary"].write_text(render_summary(meta, configs, summaries, failures), encoding="utf-8")

    cols = ["config", "id", "type", "answerable", "hit@1", "hit@3", "hit@5", "recall@5", "rr",
            "refused", "refusal_reason", "correct", "faithful", "fact_recall", "citation_precision",
            "latency_s", "error"]
    with paths["csv"].open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for name, rows in all_results.items():
            for r in rows:
                writer.writerow({"config": name, **r})

    judged = [(name, r) for name, rows in all_results.items() for r in rows
              if r.get("correct") is not None and r.get("answer") and not r.get("refused")]
    sample = random.Random(42).sample(judged, min(20, len(judged)))
    with paths["judge_sample"].open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["config", "id", "question", "answer", "judge_correct", "judge_faithful",
                         "judge_reason", "human_correct", "human_faithful"])
        for name, r in sample:
            writer.writerow([name, r["id"], r["question"], r["answer"], r["correct"], r["faithful"],
                             r["judge_reason"], "", ""])
    return paths
