# AI usage

I used an AI assistant as a learning and development aid for this project. This file documents how,
what it produced, and how I checked and understood it.

## Tools and models

| Tool | Model | Used for |
|---|---|---|
| Claude (Anthropic), claude.ai chat | _write the model name shown in the chat interface_ | Planning, code generation, debugging, test writing, documentation drafts |
| GitHub / VS Code | none | Version control and editing (no code-completion assistant was used) _(edit if untrue)_ |

Runtime models used **inside the product** (not as development aids): Cohere Embed v4 (embeddings),
Cohere Rerank, and the answer LLM configured in `.env` (_name it_), plus an LLM judge for evaluation.

## Configuration

Default settings of the chat interface; no custom instructions or system prompt of my own _(edit if
untrue)_. The conversation was one long thread about this project.

## Prompts and chat history

- **Complete chat history:** `docs/ai_chat_history/` _(export the whole conversation here)_. In claude.ai, use
  the account/data export, or share or print the conversation, then save it as text or PDF. Check the
  current interface for the exact menu.
- **Complete list of prompts:** `docs/ai_chat_history/prompts.md` _(list every message you sent, in order)_.
  The conversation export contains them; extract your own messages into this list.

## How the assistant was used

The assistant proposed an implementation plan, then wrote most of the first draft of the code, tests,
notebooks and documents in stages (extraction, chunking, retrieval, answering, API, guardrails,
evaluation, frontend). It also diagnosed errors that I pasted back (for example Python environment
problems, Git branch mistakes, Pinecone and rate-limit errors).

## What I did myself

_Be specific and truthful. Examples to adapt:_
- Chose the stack (Neon, Pinecone, Cohere) and the corpus; wrote or selected my own PDF and DOCX files.
- Ran every command, created the cloud accounts, keys and the Pinecone index, and debugged my environment.
- Reviewed the code stage by stage, ran the notebooks on my corpus, and recorded observations in them.
- Decided on the LLM provider when the Cohere trial allowance ran out (chat endpoint rate limited).
- Labelled _N_ judge answers by hand to measure judge reliability (`EVAL.md`).
- Tested the application in the browser and recorded the demo.

## How the output was verified

- The automated suite (`pytest -q`) and the UI smoke test (`npm test` in `frontend/`) must pass.
- Live checks against the real services (`python -m scripts.check_services`) and the evaluation
  (`python -m evals.run`) measure behaviour that fakes cannot.
- Several assistant errors were found by tests or by running the code and were then fixed, for
  example: a chunk dropped when it equalled the previous one, a URL validator that corrupted SQLite
  URLs, a Pinecone 404 mis-read as an outage, and a retry window shorter than a rate-limit window.
- Synthetic material is disclosed: `nimbus_faq.pdf`, the golden set built from it, and the edge-case
  files from `scripts/make_demo_files.py` were generated with AI assistance, so their ground truth is
  known by construction.

## Understanding the code

I can explain every module. Before submission I walked through these in order and can describe the
decision behind each: `config.py`, `errors.py`, `logging_conf.py`, `ingestion/validate.py`,
`ingestion/extract.py`, `ingestion/chunking.py`, `clients/resilience.py`, `clients/throttle.py`,
`clients/pinecone_client.py`, `retrieval.py`, `generation.py`, `guardrails.py`, `services/documents.py`,
`services/qa.py`, `routers/`, `frontend/app.js`, and `evals/`.
