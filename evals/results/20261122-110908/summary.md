# Evaluation results (2026-11-22 11:38 UTC)

Golden set: 40 questions (sha256 `dd04bfe127ca`). Embedding `embed-v4.0` (1024d), LLM `command-a-03-2025`, judge `command-a-03-2025 (same as LLM)`, rerank `rerank-v3.5`. Commit `3d68484`.

## Configurations

| Config | Chunk size / overlap | Rerank | Description |
|---|---|---|---|
| A | 400/50 | no | baseline: 400/50 tokens, vector search only |
| B | 400/50 | yes | 400/50 + Cohere rerank (isolates the effect of reranking) |
| C | 200/30 | yes | small chunks + rerank |
| D | 800/100 | yes | large chunks + rerank |
| E | 400/50 | yes | B + refusal threshold on the rerank score (calibrated from B's scores) |

## Retrieval (answerable questions)

| Metric | A | B | C | D | E |
|---|---|---|---|---|---|
| hit@1 | 0.906 | 1.000 | 0.969 | 1.000 | 1.000 |
| hit@3 | 0.969 | 1.000 | 1.000 | 1.000 | 1.000 |
| hit@5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| recall@3 | 0.969 | 1.000 | 1.000 | 1.000 | 1.000 |
| recall@5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| MRR | 0.945 | 1.000 | 0.984 | 1.000 | 1.000 |
| top-1 source page correct | 0.969 | 1.000 | 1.000 | 1.000 | 1.000 |

## Answers

| Metric | A | B | C | D | E |
|---|---|---|---|---|---|
| correct (judge) | 1.000 | 1.000 | 1.000 | 1.000 | 0.938 |
| faithful (judge) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| key-fact recall (lexical) | 0.984 | 0.953 | 0.953 | 0.950 | 0.891 |
| citation precision | 0.984 | 0.984 | 0.990 | 1.000 | 1.000 |
| false refusals (answerable) | 0.000 | 0.000 | 0.000 | 0.000 | 0.062 |
| unanswerable correctly refused | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| unanswerable hallucinated | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Cost and speed

| Metric | A | B | C | D | E |
|---|---|---|---|---|---|
| latency p50 (s) | 2.26 | 4.45 | 2.98 | 6.12 | 3.46 |
| latency p95 (s) | 14.23 | 10.06 | 28.02 | 19.74 | 27.54 |
| avg generation tokens in | 2760 | 2834 | 2008 | 4066 | 2110 |
| avg generation tokens out | 52 | 54 | 47 | 56 | 62 |

## By question type (hit@5 / correct)

| Metric | A | B | C | D | E |
|---|---|---|---|---|---|
| direct | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| multi_part | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 0.86 |
| reworded | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 0.90 |

## Winner: **B**

Rule: highest judge-correct rate, then hit@5, then MRR, then lower p95 latency.

## Threshold calibration (top-1 score, answerable vs unanswerable)

- **A / vector_score**: best threshold 0.3387 (balanced accuracy 0.8281); keeping every answerable question needs 0.3195 and rejects 38% of unanswerable. Answerable range [0.3195, 0.5636], unanswerable range [0.1909, 0.5184].
- **B / vector_score**: best threshold 0.3387 (balanced accuracy 0.8281); keeping every answerable question needs 0.3195 and rejects 38% of unanswerable. Answerable range [0.3195, 0.5636], unanswerable range [0.1909, 0.5184].
- **B / rerank_score**: best threshold 0.2191 (balanced accuracy 0.9688); keeping every answerable question needs 0.1518 and rejects 62% of unanswerable. Answerable range [0.1518, 0.9063], unanswerable range [0.0343, 0.2183].
- **C / vector_score**: best threshold 0.3315 (balanced accuracy 0.8125); keeping every answerable question needs 0.3341 and rejects 62% of unanswerable. Answerable range [0.3341, 0.617], unanswerable range [0.1744, 0.5339].
- **C / rerank_score**: best threshold 0.2002 (balanced accuracy 0.9844); keeping every answerable question needs 0.1045 and rejects 50% of unanswerable. Answerable range [0.1045, 0.9063], unanswerable range [0.0211, 0.1695].
- **D / vector_score**: best threshold 0.3285 (balanced accuracy 0.8083); keeping every answerable question needs 0.2933 and rejects 38% of unanswerable. Answerable range [0.2933, 0.5602], unanswerable range [0.1738, 0.5014].
- **D / rerank_score**: best threshold 0.2666 (balanced accuracy 0.9167); keeping every answerable question needs 0.1117 and rejects 38% of unanswerable. Answerable range [0.1117, 0.9063], unanswerable range [0.0263, 0.2436].

## Failures

### Config A: 0 question(s) with issues

### Config B: 0 question(s) with issues

### Config C: 0 question(s) with issues

### Config D: 2 question(s) with issues
- `q031` (multi_part) error: How do Trail Points work, and how long are gift cards valid?
- `q032` (multi_part) error: What is the warranty on sleeping bags, and do you offer a student discount?

### Config E: 2 question(s) with issues
- `q021` (reworded) false_refusal: How many years is a sleeping bag guaranteed for?
- `q031` (multi_part) false_refusal: How do Trail Points work, and how long are gift cards valid?
