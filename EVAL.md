# Evaluation report

Generated from `evals/results/20261122-110908` (2026-11-22 11:38 UTC, commit `3d68484`). Every number below is read from the saved results; nothing is typed by hand.

## 1. What was measured and how

**Golden set:** 40 questions over `nimbus_faq.pdf`: 15 direct, 7 multi-part, 10 reworded, 8 unanswerable. 8 questions cannot be answered from the corpus. Each answerable question is labelled with the source document, the page(s), an *evidence phrase* copied from the source text, key facts and a reference answer.

**Relevance is evidence-based, not page-based.** A retrieved chunk counts as relevant if it comes from the labelled document and contains an evidence phrase (case- and whitespace-insensitive). This keeps the numbers comparable across chunk sizes: page-only labels would reward large chunks simply for covering more of a page.

| Metric | Definition |
|---|---|
| hit@k | 1 if any of the top-k chunks is relevant, averaged over answerable questions |
| recall@k | share of a question's evidence items found in the top-k chunks (matters for multi-part questions) |
| MRR | mean of 1/rank of the first relevant chunk |
| correct (judge) | an LLM judge compares the answer with the reference answer |
| faithful (judge) | every claim in the answer is supported by the retrieved excerpts |
| key-fact recall | share of short key facts found verbatim in the answer (lexical, a lower bound) |
| citation precision | share of cited chunks that are relevant |
| false refusal | answerable question that the system refused |
| unanswerable refused | unanswerable question correctly refused (the rest are hallucinations) |

**Why a custom judge instead of RAGAS or Langfuse evals.** RAGAS pulls in heavy dependencies, hides its prompts, and its provider wiring is version-fragile. A short visible judge prompt (`evals/judge.py`) that is validated against controlled and human-labelled answers is easier to audit and reproduce. Langfuse is a tracing tool rather than a judge; tracing is listed as future work in DESIGN.md.

## 2. Judge reliability

The judge model is `command-a-03-2025 (same as LLM)`; the answering model is `command-a-03-2025`. **They are the same model, so the judge may be biased toward its own style of answer.**

**Self-test with known answers (16 questions).** The judge is shown answers whose correct verdict is known in advance:

| Control answer | n | correct flag right | faithful flag right |
|---|---|---|---|
| the reference answer (should be accepted) | 16 | 100% | 100% |
| numbers changed / yes-no flipped (should be rejected) | 16 | 100% | 100% |
| right answer + an invented sentence (should be flagged unfaithful) | 16 | n/a | 100% |

This shows the judge is neither lenient nor harsh on clear-cut cases. It does not prove it is reliable on subtle ones, which is what the human check below is for.

*Human agreement not measured yet: fill the `human_correct` / `human_faithful` columns of `evals/results/20261122-110908/judge_sample.csv`, then run `python -m evals.judge_agreement evals/results/20261122-110908/judge_sample.csv` and regenerate this file.*

## 3. Configurations compared

| Config | Chunk size / overlap | Rerank | Refusal threshold | Description |
|---|---|---|---|---|
| A | 400/50 | no | 0.0 | baseline: 400/50 tokens, vector search only |
| B | 400/50 | yes | 0.0 | 400/50 + Cohere rerank (isolates the effect of reranking) |
| C | 200/30 | yes | 0.0 | small chunks + rerank |
| D | 800/100 | yes | 0.0 | large chunks + rerank |
| E | 400/50 | yes | 0.1 | B + refusal threshold on the rerank score (calibrated from B's scores) |

## 4. Results

### Retrieval (answerable questions)

| Metric | A | B | C | D | E |
|---|---|---|---|---|---|
| hit@1 | 0.906 | 1.000 | 0.969 | 1.000 | 1.000 |
| hit@3 | 0.969 | 1.000 | 1.000 | 1.000 | 1.000 |
| hit@5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| recall@3 | 0.969 | 1.000 | 1.000 | 1.000 | 1.000 |
| recall@5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| MRR | 0.945 | 1.000 | 0.984 | 1.000 | 1.000 |
| top-1 source page correct | 0.969 | 1.000 | 1.000 | 1.000 | 1.000 |

### Answers

| Metric | A | B | C | D | E |
|---|---|---|---|---|---|
| correct (judge) | 1.000 | 1.000 | 1.000 | 1.000 | 0.938 |
| faithful (judge) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| key-fact recall | 0.984 | 0.953 | 0.953 | 0.950 | 0.891 |
| citation precision | 0.984 | 0.984 | 0.990 | 1.000 | 1.000 |
| false refusals | 0.000 | 0.000 | 0.000 | 0.000 | 0.062 |
| unanswerable refused | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| unanswerable hallucinated | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### Speed and cost

| Metric | A | B | C | D | E |
|---|---|---|---|---|---|
| latency p50 (s) | 2.26 | 4.45 | 2.98 | 6.12 | 3.46 |
| latency p95 (s) | 14.23 | 10.06 | 28.02 | 19.74 | 27.54 |
| generation tokens in / question | 2760 | 2834 | 2008 | 4066 | 2110 |
| generation tokens out / question | 52 | 54 | 47 | 56 | 62 |

### By question type (hit@5 / judge-correct)

| Metric | A | B | C | D | E |
|---|---|---|---|---|---|
| direct | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| multi_part | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 0.86 |
| reworded | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 0.90 |

## 5. Winner: configuration B

**Selection rule (fixed before looking at results):** highest judge-correct rate, then hit@5, then MRR, then lower p95 latency.

Closest competitor: **A** (correct 1.000 vs 1.000, hit@5 1.000 vs 1.000, MRR 0.945 vs 1.000).

## 6. Effect of each design choice

| Comparison | What changed | Δ hit@1 | Δ hit@5 | Δ MRR | Δ correct | Δ false refusals | Δ p95 latency (s) |
|---|---|---|---|---|---|---|---|
| A → B | add reranking | +0.094 | +0.000 | +0.055 | +0.000 | +0.000 | -4.175 |
| B → C | chunks 400 → 200 tokens | -0.031 | +0.000 | -0.016 | +0.000 | +0.000 | +17.956 |
| B → D | chunks 400 → 800 tokens | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +9.681 |
| B → E | add rerank-score refusal threshold | +0.000 | +0.000 | +0.000 | -0.062 | +0.062 | +17.480 |

*TODO (write yourself): one or two sentences per row on whether the change was worth its cost.*

## 7. Refusal threshold calibration

- **Config A, vector score:** answerable top-1 scores span [0.3195, 0.5636], unanswerable span [0.1909, 0.5184]. Best single threshold 0.3387 (balanced accuracy 0.8281); the strictest threshold that still keeps every answerable question is 0.3195 and rejects 38% of unanswerable questions before any LLM call.

Overlapping ranges mean a score threshold alone cannot separate the two groups, which is why the model's own `answerable` judgement remains the final layer. The threshold was tuned on the same questions it is evaluated on, so its benefit is likely slightly optimistic.

## 8. Failure analysis

### Configuration A: 0 question(s) with issues
None.

### Configuration B: 0 question(s) with issues
None.

### Configuration C: 0 question(s) with issues
None.

### Configuration D: 2 question(s) with issues
- `q031` (multi_part) **error**: How do Trail Points work, and how long are gift cards valid?
- `q032` (multi_part) **error**: What is the warranty on sleeping bags, and do you offer a student discount?

### Configuration E: 2 question(s) with issues
- `q021` (reworded) **false_refusal**: How many years is a sleeping bag guaranteed for? Answer: “I couldn't produce an answer that I could verify against your documents.”
- `q031` (multi_part) **false_refusal**: How do Trail Points work, and how long are gift cards valid? Answer: “I couldn't produce an answer that I could verify against your documents.”

*TODO (write yourself): for each failure pattern above, explain the cause (retrieval miss vs generation error vs judge error), show one example, and say what you would change.*

## 9. Limitations

- The golden set has 40 questions over a small, partly synthetic corpus; differences of a few points are within noise.
- Results hold for the models and settings in the run metadata, and the answer model in production may differ.
- Scanned (image-only) PDFs are rejected, not OCR'd, so they are outside this evaluation.
- Repeated identical requests inside a run are cached (temperature 0), which saves calls without changing what is measured.
- Evidence-phrase relevance can miss a correct chunk that words the fact differently.

## 10. Reproduce

```powershell
python -m evals.validate_golden
python -m evals.run
python -m evals.judge_selftest --n 16
python -m evals.make_eval_md
```

Run metadata: embedding `embed-v4.0` (1024d), LLM `command-a-03-2025`, judge `command-a-03-2025 (same as LLM)`, rerank `rerank-v3.5`, temperature 0.0, top-k 20, context chunks 5, golden set sha256 `dd04bfe127ca`.
