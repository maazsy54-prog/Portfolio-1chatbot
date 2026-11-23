# Production RAG Service

Upload PDF and Word documents, then ask questions in plain language. Every answer cites the numbered
passages it came from, shows the retrieved chunks with their similarity scores, and says so clearly
when the documents do not contain the answer.

**Live demo:** _add the deployed URL here (bonus H) or remove this line_  
**Demo video:** _add the link here_

| | |
|---|---|
| API | FastAPI, async, OpenAPI docs at `/docs` |
| Web UI | HTML + JavaScript + Tailwind CSS, served by the API at `/app/` |
| Embeddings and rerank | Cohere Embed v4 (1024 dimensions) and Cohere Rerank |
| Vector database | Pinecone serverless (cosine) |
| Metadata database | Neon Postgres (SQLAlchemy async) |
| Answer model | Any of Cohere, Claude, or an OpenAI-compatible API, chosen in `.env` |
| Quality | pytest suite, a 40-question golden set, retrieval and answer metrics |

## How it works

```
Upload  ->  validate (type by content, size, hash)  ->  extract text (PDF / DOCX)
        ->  chunk (token-aware, keeps page / section)  ->  embed (Cohere)  ->  Pinecone + Neon

Question -> input guard -> embed -> vector search (top 20) -> rerank (top 5) -> threshold
         -> LLM answers from the excerpts as JSON -> citations checked against the retrieved chunks
         -> output guard -> answer + citations + retrieved chunks with scores (or a refusal)
```

See `DESIGN.md` for every decision and its trade-offs, and `EVAL.md` for the measured results.

## Setup (Windows PowerShell)

Requirements: Python 3.11 or newer, a Cohere API key, a Pinecone account, a Neon Postgres database,
and a key for the answer model.

```powershell
git clone https://github.com/maazsy54-prog/Portfolio-1chatbot.git
cd Portfolio-1chatbot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt     # tests and evaluation tooling
Copy-Item .env.example .env             # then edit .env (see below)
```

**Create the Pinecone index once** (name `rag-portfolio`, dimension **1024**, metric **cosine**,
serverless) in the Pinecone console, or with:

```powershell
python -c "from app.config import get_settings; from pinecone import Pinecone, ServerlessSpec; s=get_settings(); Pinecone(api_key=s.pinecone_api_key.get_secret_value()).create_index(name=s.pinecone_index, dimension=s.embedding_dim, metric='cosine', spec=ServerlessSpec(cloud='aws', region='us-east-1'))"
```

### Configuration (`.env`)

Secrets are read only from environment variables. `.env` is git-ignored; `.env.example` lists every setting.

| Variable | Purpose |
|---|---|
| `COHERE_API_KEY` | Embeddings and reranking (and answers, if `LLM_PROVIDER=cohere`) |
| `PINECONE_API_KEY`, `PINECONE_INDEX` | Vector database |
| `DATABASE_URL` | Neon connection string. Any `postgresql://...?sslmode=require` form is converted for asyncpg automatically; use the pooled host |
| `LLM_PROVIDER` | `cohere` (default), `anthropic`, or `openai_compatible` |
| `LLM_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL` | Answer model settings for the non-Cohere providers |
| `JUDGE_MODEL` | Optional separate model for the evaluation judge |
| `LLM_CALLS_PER_MINUTE`, `RERANK_CALLS_PER_MINUTE` | Client-side pacing for rate-limited keys (0 = unlimited) |
| `RERANK_ENABLED`, `RERANK_THRESHOLD`, `SIMILARITY_THRESHOLD` | Retrieval behaviour |

Check that everything is reachable (one tiny call per service; shows each provider's own error text):

```powershell
python -m scripts.check_services
```

## Run

```powershell
uvicorn app.main:app --reload
```

- Web UI: http://127.0.0.1:8000 (redirects to `/app/`)
- API docs: http://127.0.0.1:8000/docs

Tables are created in Neon on first start.

### API summary

| Method and path | Purpose | Success |
|---|---|---|
| `POST /documents` | Upload and ingest a PDF or DOCX | 201 |
| `GET /documents` / `GET /documents/{id}` | List / fetch documents | 200 |
| `DELETE /documents/{id}` | Delete a document and all its vectors | 204 |
| `POST /ask` | Ask a question; returns answer, citations, retrieved chunks, scores | 200 |
| `GET /health` | Liveness | 200 |

Errors share one shape: `{"error": {"code", "message", "request_id", "details"}}` with 400, 404, 409
(duplicate), 413, 415, 422, 429 (with `Retry-After`) and 503 (an upstream service is down).

## Tests

```powershell
pytest -q                               # unit + API tests (no network needed)
cd frontend; npm install; npm test      # browser-side smoke test of the UI (jsdom)
```

The Python tests use in-memory fakes for Cohere, Pinecone and the LLM, and a real SQLite database, so
they run offline. `tests/integration/` is reserved for live-service checks.

## Evaluation

```powershell
python -m evals.validate_golden         # labels match the corpus text
python -m evals.run                     # all configurations; saves evals/results/<timestamp>/
python -m evals.judge_selftest --n 16   # is the judge lenient or harsh?
python -m evals.make_eval_md            # writes EVAL.md from the newest run
```

Use `--configs B E` to run a subset and `--llm-rpm N` to pace calls under your provider's limit.

## Project layout

```
app/
  main.py            wiring only (routers, middleware, static UI)
  routers/           HTTP layer: documents.py, ask.py, health.py
  services/          documents.py (ingest / delete), qa.py (ask)
  ingestion/         validate.py, extract.py, chunking.py, records.py
  retrieval.py       embed -> search -> rerank -> threshold
  generation.py      prompt, LLM call, citation validation, refusal
  guardrails.py      input guard + output guard
  clients/           cohere, pinecone, LLM providers, retries, pacing
  db/                SQLAlchemy models and session
  config.py          all settings (secrets from the environment only)
frontend/            index.html, app.js, styles.css, tests/
evals/               golden_set.json, runner, judge, reports, results/
notebooks/           pipeline built step by step (01 extraction ... 04 answering)
tests/               unit/ and api/
```

## Troubleshooting

- **`429` with "Trial key ... API calls / month"**: your Cohere trial allowance is used up. Switch the answer model with `LLM_PROVIDER` or upgrade the key.
- **Neon `sslmode` / `channel_binding` errors**: use the pooled connection string; the app rewrites the URL for asyncpg.
- **Pinecone `404 Namespace not found`** in logs during evaluation is expected: namespaces are cleared before each run.
- **Notebook says `No module named 'pydantic'`**: select the `.venv` kernel.

## Git workflow

`main` is the stable branch. Work was done on feature branches (`feature/ingestion`, `feature/chunking`,
`feature/retrieval`, `feature/answering`, `feature/api`, `feature/guardrails`, `feature/evals`,
`feature/frontend`) and merged through pull requests. No secrets are committed.

## Documentation

- `DESIGN.md`: architecture and the justification of every decision
- `EVAL.md`: golden set, metrics, configuration comparison, failure analysis
- `AI_USAGE.md`: how AI tools were used
