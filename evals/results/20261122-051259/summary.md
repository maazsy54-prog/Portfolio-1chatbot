# Evaluation results (2026-11-22 05:17 UTC)

Golden set: 40 questions (sha256 `dd04bfe127ca`). Embedding `embed-v4.0` (1024d), LLM `command-a-03-2025`, judge `command-a-03-2025 (same as LLM)`, rerank `rerank-v3.5`. Commit `de86da2`.

## Configurations

| Config | Chunk size / overlap | Rerank | Description |
|---|---|---|---|
| B | 400/50 | yes | 400/50 + Cohere rerank (isolates the effect of reranking) |

## Retrieval (answerable questions)

| Metric | B |
|---|---|
| hit@1 | 1.000 |
| hit@3 | 1.000 |
| hit@5 | 1.000 |
| recall@3 | 1.000 |
| recall@5 | 1.000 |
| MRR | 1.000 |
| top-1 source page correct | 1.000 |

## Answers

| Metric | B |
|---|---|
| correct (judge) | 1.000 |
| faithful (judge) | 1.000 |
| key-fact recall (lexical) | 0.984 |
| citation precision | 0.953 |
| false refusals (answerable) | 0.000 |
| unanswerable correctly refused | 1.000 |
| unanswerable hallucinated | 0.000 |

## Cost and speed

| Metric | B |
|---|---|
| latency p50 (s) | 8.03 |
| latency p95 (s) | 32.95 |
| avg generation tokens in | 2851 |
| avg generation tokens out | 54 |

## By question type (hit@5 / correct)

| Metric | B |
|---|---|
| direct | 1.00 / 1.00 |
| multi_part | 1.00 / 1.00 |
| reworded | 1.00 / 1.00 |

## Winner: **B**

Rule: highest judge-correct rate, then hit@5, then MRR, then lower p95 latency.

## Threshold calibration (top-1 score, answerable vs unanswerable)

- **B / vector_score**: best threshold 0.3387 (balanced accuracy 0.8281); keeping every answerable question needs 0.3195 and rejects 38% of unanswerable. Answerable range [0.3195, 0.5639], unanswerable range [0.1909, 0.5184].
- **B / rerank_score**: best threshold 0.219 (balanced accuracy 0.9688); keeping every answerable question needs 0.1518 and rejects 62% of unanswerable. Answerable range [0.1518, 0.9063], unanswerable range [0.0343, 0.2183].

## Failures

### Config B: 0 question(s) with issues
