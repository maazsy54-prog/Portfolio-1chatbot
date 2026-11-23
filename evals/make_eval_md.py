"""Generate EVAL.md from a saved evaluation run, so every number in the report is real.

    python -m evals.make_eval_md                       # newest run in evals/results/
    python -m evals.make_eval_md evals/results/<run>   # a specific run

Reads results.json (+ judge_selftest.json and a human-labelled judge_sample.csv if present) and
writes EVAL.md. The 'analysis' parts you must write yourself are marked with TODO.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from evals.judge_agreement import agreement
from evals.report import pick_winner


def _f(v: float | None, d: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{d}f}"


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.0%}"


def latest_run(results_root: Path) -> Path:
    runs = sorted(p for p in results_root.iterdir() if (p / "results.json").exists())
    if not runs:
        raise SystemExit(f"No evaluation runs found in {results_root}. Run: python -m evals.run")
    return runs[-1]


def _table(names: list[str], rows: list[tuple[str, list[str]]]) -> list[str]:
    out = ["| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    out += [f"| {label} | " + " | ".join(vals) + " |" for label, vals in rows]
    return out


def _delta(a: float | None, b: float | None) -> str:
    return "n/a" if a is None or b is None else f"{b - a:+.3f}"


def build_eval_md(run: dict, golden: dict, selftest: dict | None, human: dict | None, run_dir: str) -> str:
    meta, cfgs = run["meta"], run["configs"]
    names = list(cfgs)
    S = {n: cfgs[n]["summary"] for n in names}
    winner = pick_winner(S)
    qs = golden["questions"]
    by_type = Counter(q["type"] for q in qs)
    judge_same = "same as LLM" in meta.get("judge_model", "")

    L: list[str] = ["# Evaluation report", "",
        f"Generated from `{run_dir}` ({meta['timestamp']}, commit `{meta['git_commit']}`). "
        "Every number below is read from the saved results; nothing is typed by hand.", ""]

    L += ["## 1. What was measured and how", "",
          f"**Golden set:** {len(qs)} questions over `{', '.join(golden['corpus'])}`: "
          + ", ".join(f"{n} {t.replace('_', '-')}" for t, n in sorted(by_type.items()))
          + f". {by_type.get('unanswerable', 0)} {'question cannot' if by_type.get('unanswerable', 0) == 1 else 'questions cannot'} "
          "be answered from the corpus. Each answerable question is labelled with the source "
          "document, the page(s), an *evidence phrase* copied from the source text, key facts and a reference answer.", "",
          "**Relevance is evidence-based, not page-based.** A retrieved chunk counts as relevant if it comes from the "
          "labelled document and contains an evidence phrase (case- and whitespace-insensitive). This keeps the numbers "
          "comparable across chunk sizes: page-only labels would reward large chunks simply for covering more of a page.", "",
          "| Metric | Definition |", "|---|---|",
          "| hit@k | 1 if any of the top-k chunks is relevant, averaged over answerable questions |",
          "| recall@k | share of a question's evidence items found in the top-k chunks (matters for multi-part questions) |",
          "| MRR | mean of 1/rank of the first relevant chunk |",
          "| correct (judge) | an LLM judge compares the answer with the reference answer |",
          "| faithful (judge) | every claim in the answer is supported by the retrieved excerpts |",
          "| key-fact recall | share of short key facts found verbatim in the answer (lexical, a lower bound) |",
          "| citation precision | share of cited chunks that are relevant |",
          "| false refusal | answerable question that the system refused |",
          "| unanswerable refused | unanswerable question correctly refused (the rest are hallucinations) |", "",
          "**Why a custom judge instead of RAGAS or Langfuse evals.** RAGAS pulls in heavy dependencies, hides its "
          "prompts, and its provider wiring is version-fragile. A short visible judge prompt (`evals/judge.py`) that "
          "is validated against controlled and human-labelled answers is easier to audit and reproduce. Langfuse "
          "is a tracing tool rather than a judge; tracing is listed as future work in DESIGN.md.", ""]

    L += ["## 2. Judge reliability", ""]
    L.append(f"The judge model is `{meta['judge_model']}`; the answering model is `{meta['llm_model']}`."
             + (" **They are the same model, so the judge may be biased toward its own style of answer.**" if judge_same else
                " They differ, so the judge is not grading its own output."))
    L.append("")
    if selftest:
        k = selftest["by_kind"]
        L += [f"**Self-test with known answers ({selftest['n_questions']} questions).** The judge is shown answers whose "
              "correct verdict is known in advance:", "",
              "| Control answer | n | correct flag right | faithful flag right |", "|---|---|---|---|"]
        for key, label in [("correct", "the reference answer (should be accepted)"),
                           ("wrong_fact", "numbers changed / yes-no flipped (should be rejected)"),
                           ("unsupported_claim", "right answer + an invented sentence (should be flagged unfaithful)")]:
            e = k.get(key, {})
            L.append(f"| {label} | {e.get('n', 0)} | {_pct(e.get('correct_verdict_ok'))} | {_pct(e.get('faithful_verdict_ok'))} |")
        L += ["", "This shows the judge is neither lenient nor harsh on clear-cut cases. It does not prove it is "
              "reliable on subtle ones, which is what the human check below is for.", ""]
    else:
        L += ["*Self-test not run yet: `python -m evals.judge_selftest --n 16`.*", ""]
    if human and any(v["n"] for v in human.values()):
        L += ["**Agreement with a human on real answers** (`judge_sample.csv`, hand-labelled):", "",
              "| Property | n | agreement | Cohen's kappa |", "|---|---|---|---|"]
        for prop, v in human.items():
            kappa = "undefined (one rater used a single label)" if v["kappa"] is None and v["n"] else _f(v["kappa"], 2)
            L.append(f"| {prop} | {v['n']} | {_pct(v['agreement'])} | {kappa} |")
        L.append("")
    else:
        L += ["*Human agreement not measured yet: fill the `human_correct` / `human_faithful` columns of "
              f"`{run_dir}/judge_sample.csv`, then run `python -m evals.judge_agreement {run_dir}/judge_sample.csv` "
              "and regenerate this file.*", ""]

    L += ["## 3. Configurations compared", "", "| Config | Chunk size / overlap | Rerank | Refusal threshold | Description |", "|---|---|---|---|---|"]
    for n in names:
        c = cfgs[n]["config"]
        thr = c["rerank_threshold"] if c["rerank"] else c["similarity_threshold"]
        L.append(f"| {n} | {c['chunk_size']}/{c['chunk_overlap']} | {'yes' if c['rerank'] else 'no'} | {thr} | {c['description']} |")

    L += ["", "## 4. Results", "", "### Retrieval (answerable questions)", ""]
    L += _table(names, [(lab, [_f(S[n]["retrieval"][key]) for n in names]) for lab, key in
        [("hit@1", "hit@1"), ("hit@3", "hit@3"), ("hit@5", "hit@5"), ("recall@3", "recall@3"),
         ("recall@5", "recall@5"), ("MRR", "mrr"), ("top-1 source page correct", "source_page_acc")]])
    L += ["", "### Answers", ""]
    L += _table(names, [(lab, [_f(S[n]["answers"][key]) for n in names]) for lab, key in
        [("correct (judge)", "correct_rate"), ("faithful (judge)", "faithful_rate"),
         ("key-fact recall", "fact_recall"), ("citation precision", "citation_precision"),
         ("false refusals", "false_refusal_rate"), ("unanswerable refused", "unanswerable_refused_rate"),
         ("unanswerable hallucinated", "unanswerable_hallucinated_rate")]])
    L += ["", "### Speed and cost", ""]
    L += _table(names, [("latency p50 (s)", [_f(S[n]["latency_s"]["p50"], 2) for n in names]),
                        ("latency p95 (s)", [_f(S[n]["latency_s"]["p95"], 2) for n in names]),
                        ("generation tokens in / question", [_f(S[n]["tokens_per_question"]["generation_in"], 0) for n in names]),
                        ("generation tokens out / question", [_f(S[n]["tokens_per_question"]["generation_out"], 0) for n in names])])
    types = sorted({t for n in names for t in S[n]["by_type"]})
    L += ["", "### By question type (hit@5 / judge-correct)", ""]
    L += _table(names, [(t, [f"{_f(S[n]['by_type'][t]['hit@5'], 2)} / {_f(S[n]['by_type'][t]['correct'], 2)}"
                             if t in S[n]["by_type"] else "-" for n in names]) for t in types])

    L += ["", f"## 5. Winner: configuration {winner}", "",
          "**Selection rule (fixed before looking at results):** highest judge-correct rate, then hit@5, then MRR, "
          "then lower p95 latency.", ""]
    others = sorted((n for n in names if n != winner), key=lambda n: -(S[n]["answers"]["correct_rate"] or 0))
    if others:
        r = S[others[0]]
        w = S[winner]
        L.append(f"Closest competitor: **{others[0]}** (correct {_f(r['answers']['correct_rate'])} vs "
                 f"{_f(w['answers']['correct_rate'])}, hit@5 {_f(r['retrieval']['hit@5'])} vs {_f(w['retrieval']['hit@5'])}, "
                 f"MRR {_f(r['retrieval']['mrr'])} vs {_f(w['retrieval']['mrr'])}).")
        if all(abs((S[n]["answers"]["correct_rate"] or 0) - (w["answers"]["correct_rate"] or 0)) < 1e-9 for n in names):
            L.append("")
            L.append("**Caution:** all configurations tie on answer correctness, so the golden set may be too easy to "
                     "separate them (a ceiling effect). The winner was then decided by the retrieval tie-breakers.")
    L += ["", "## 6. Effect of each design choice", "",
          "| Comparison | What changed | Δ hit@1 | Δ hit@5 | Δ MRR | Δ correct | Δ false refusals | Δ p95 latency (s) |", "|---|---|---|---|---|---|---|---|"]
    def row(a: str, b: str, what: str) -> None:
        if a in S and b in S:
            A, B = S[a], S[b]
            L.append(f"| {a} → {b} | {what} | {_delta(A['retrieval']['hit@1'], B['retrieval']['hit@1'])} | "
                     f"{_delta(A['retrieval']['hit@5'], B['retrieval']['hit@5'])} | {_delta(A['retrieval']['mrr'], B['retrieval']['mrr'])} | "
                     f"{_delta(A['answers']['correct_rate'], B['answers']['correct_rate'])} | "
                     f"{_delta(A['answers']['false_refusal_rate'], B['answers']['false_refusal_rate'])} | "
                     f"{_delta(A['latency_s']['p95'], B['latency_s']['p95'])} |")
    row("A", "B", "add reranking")
    row("B", "C", "chunks 400 → 200 tokens")
    row("B", "D", "chunks 400 → 800 tokens")
    row("B", "E", "add rerank-score refusal threshold")
    L += ["", "*TODO (write yourself): one or two sentences per row on whether the change was worth its cost.*", ""]

    L += ["## 7. Refusal threshold calibration", ""]
    cal = next(((n, S[n]["calibration"]) for n in names if S[n].get("calibration")), None)
    if cal:
        n, c = cal
        for kind, v in c.items():
            if v:
                L.append(f"- **Config {n}, {kind.replace('_', ' ')}:** answerable top-1 scores span {v['answerable_score_range']}, "
                         f"unanswerable span {v['unanswerable_score_range']}. Best single threshold {v['best_threshold']} "
                         f"(balanced accuracy {v['balanced_accuracy']}); the strictest threshold that still keeps every "
                         f"answerable question is {v['keep_all_answerable_threshold']} and rejects "
                         f"{v['unanswerable_rejected_at_keep_all']:.0%} of unanswerable questions before any LLM call.")
        L += ["", "Overlapping ranges mean a score threshold alone cannot separate the two groups, which is why the "
              "model's own `answerable` judgement remains the final layer. The threshold was tuned on the same "
              "questions it is evaluated on, so its benefit is likely slightly optimistic.", ""]
    else:
        L += ["No calibration data in this run.", ""]

    L += ["## 8. Failure analysis", ""]
    total = 0
    for n in names:
        fails = cfgs[n]["failures"]
        total += len(fails)
        L.append(f"### Configuration {n}: {len(fails)} question(s) with issues")
        if not fails:
            L.append("None.")
        for f in fails[:30]:
            ans = (f.get("answer") or "").replace("\n", " ")[:140]
            L.append(f"- `{f['id']}` ({f['type']}) **{', '.join(f['issues'])}**: {f['question']}"
                     + (f" Answer: “{ans}”" if ans else ""))
        L.append("")
    if total == 0:
        L += ["No configuration had a failing question on this golden set. That is good, but it also means the set "
              "does not exercise the system's weak points. Add harder questions about messy real documents "
              "(tables, scanned-looking layouts, conflicting statements) and re-run.", ""]
    L += ["*TODO (write yourself): for each failure pattern above, explain the cause (retrieval miss vs generation "
          "error vs judge error), show one example, and say what you would change.*", "",
          "## 9. Limitations", "",
          f"- The golden set has {len(qs)} questions over a small, partly synthetic corpus; differences of a few points are within noise.",
          "- Results hold for the models and settings in the run metadata, and the answer model in production may differ.",
          "- Scanned (image-only) PDFs are rejected, not OCR'd, so they are outside this evaluation.",
          "- Repeated identical requests inside a run are cached (temperature 0), which saves calls without changing what is measured.",
          "- Evidence-phrase relevance can miss a correct chunk that words the fact differently.", "",
          "## 10. Reproduce", "", "```powershell", "python -m evals.validate_golden", "python -m evals.run",
          "python -m evals.judge_selftest --n 16", "python -m evals.make_eval_md", "```", "",
          f"Run metadata: embedding `{meta['embedding_model']}` ({meta['embedding_dim']}d), LLM `{meta['llm_model']}`, "
          f"judge `{meta['judge_model']}`, rerank `{meta['rerank_model']}`, temperature {meta['temperature']}, "
          f"top-k {meta['retrieval_top_k']}, context chunks {meta['context_top_n']}, golden set sha256 `{meta['golden_sha256'][:12]}`.", ""]
    return "\n".join(L)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EVAL.md from a saved evaluation run.")
    parser.add_argument("run_dir", nargs="?", type=Path)
    parser.add_argument("--golden", default="evals/golden_set.json", type=Path)
    parser.add_argument("--output", default="EVAL.md", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir or latest_run(Path("evals/results"))
    run = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    golden = json.loads(args.golden.read_text(encoding="utf-8"))
    st_path = Path("evals/results/judge_selftest.json")
    selftest = json.loads(st_path.read_text(encoding="utf-8")) if st_path.exists() else None
    human = None
    sample = run_dir / "judge_sample.csv"
    if sample.exists():
        with sample.open(encoding="utf-8") as fh:
            human = agreement(list(csv.DictReader(fh)))
    args.output.write_text(build_eval_md(run, golden, selftest, human, run_dir.as_posix()), encoding="utf-8")
    print(f"Wrote {args.output} from {run_dir}")


if __name__ == "__main__":
    main()
