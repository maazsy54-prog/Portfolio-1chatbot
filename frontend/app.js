"use strict";
/*
 * Document Q&A frontend (vanilla JS, same-origin API).
 * Security: all server-provided text (answers, filenames, document passages) is inserted with
 * textContent / text nodes, never innerHTML, so document content cannot inject markup.
 */
(function () {
  const MAX_QUESTION = 1000;
  const EXAMPLES = [
    "How long does standard shipping take?",
    "What is the return window, and who pays for return shipping?",
    "Can you match a lower price I found at another shop?",
    "Do you sell hiking boots?",
  ];
  // How each refusal reason is shown. tone: warn (amber) | danger (brick)
  const REFUSALS = {
    no_documents: { label: "No documents yet", tone: "warn" },
    not_in_documents: { label: "Not found in your documents", tone: "warn" },
    no_relevant_context: { label: "Not found in your documents", tone: "warn" },
    unverified_answer: { label: "Could not verify an answer", tone: "warn" },
    input_blocked: { label: "Blocked by the safety filter", tone: "danger" },
    output_blocked: { label: "Answer withheld for safety", tone: "danger" },
  };

  const $ = (id) => document.getElementById(id);
  const state = { selected: new Set(), documents: [], busy: false, turnCounter: 0 };

  /* ---------------------------------------------------------------- helpers */

  function el(tag, opts = {}, children = []) {
    const node = document.createElement(tag);
    if (opts.class) node.className = opts.class;
    if (opts.text != null) node.textContent = opts.text;
    if (opts.attrs) for (const [k, v] of Object.entries(opts.attrs)) node.setAttribute(k, v);
    if (opts.on) for (const [k, fn] of Object.entries(opts.on)) node.addEventListener(k, fn);
    for (const child of [].concat(children)) if (child) node.append(child);
    return node;
  }

  function formatBytes(n) {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
    return `${(n / 1024 / 1024).toFixed(1)} MB`;
  }
  const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
  const clamp01 = (x) => Math.max(0, Math.min(1, Number(x) || 0));

  class ApiError extends Error {
    constructor(status, body, retryAfter) {
      super(body.message);
      this.status = status;
      this.code = body.code;
      this.requestId = body.request_id || null;
      this.details = body.details;
      this.retryAfter = retryAfter;
    }
  }

  async function api(path, options = {}) {
    let response;
    try {
      response = await fetch(path, options);
    } catch (_) {
      throw new ApiError(0, { code: "network_error", message: "Could not reach the server. Check that it is running." });
    }
    if (response.status === 204) return null;
    let data = null;
    try { data = await response.json(); } catch (_) { /* non-JSON body */ }
    if (!response.ok) {
      const body = (data && data.error) || { code: "http_error", message: `The request failed (HTTP ${response.status}).` };
      throw new ApiError(response.status, body, response.headers.get("Retry-After"));
    }
    return data;
  }

  /** A short, actionable explanation for an API error. */
  function describeError(err) {
    const status = err.status ? ` (HTTP ${err.status})` : "";
    switch (err.code) {
      case "rate_limited":
        return `Too many requests${status}. Wait ${err.retryAfter || "a few"} seconds, then try again.`;
      case "service_unavailable":
        return `A backend service is not responding${status}. Try again in a moment.`;
      case "duplicate_document":
        return `${err.message}${status}`;
      default:
        return `${err.message}${status}`;
    }
  }

  /* ----------------------------------------------------------------- toasts */

  function toast(kind, message) {
    const tone = kind === "error" ? "border-danger text-danger" : "border-accent text-accent";
    const node = el("div", {
      class: `pointer-events-auto rounded border-l-4 bg-sheet px-3 py-2 text-sm shadow ${tone}`,
      attrs: { role: kind === "error" ? "alert" : "status" },
    }, [
      el("span", { class: "text-ink", text: message }),
      el("button", {
        class: "ml-3 text-xs font-medium underline", text: "Dismiss", attrs: { type: "button" },
        on: { click: () => node.remove() },
      }),
    ]);
    $("toasts").append(node);
    setTimeout(() => node.remove(), kind === "error" ? 9000 : 4000);
  }

  /* -------------------------------------------------------------- documents */

  function renderDocuments() {
    const list = $("docList");
    list.replaceChildren();
    $("docCount").textContent = `(${state.documents.length})`;
    $("docEmpty").hidden = state.documents.length > 0;

    for (const doc of state.documents) {
      const checkId = `sel-${doc.id}`;
      const meta = [
        doc.file_type.toUpperCase(),
        doc.page_count ? plural(doc.page_count, "page") : null,
        plural(doc.chunk_count, "passage"),
        formatBytes(doc.size_bytes),
      ].filter(Boolean).join(", ");

      const checkbox = el("input", {
        class: "mt-1 h-4 w-4 accent-[#0E5A78]",
        attrs: { type: "checkbox", id: checkId, "aria-label": `Search only ${doc.filename}` },
        on: {
          change: (e) => {
            e.target.checked ? state.selected.add(doc.id) : state.selected.delete(doc.id);
            updateScopeNote();
          },
        },
      });
      checkbox.checked = state.selected.has(doc.id);

      list.append(el("li", { class: "flex items-start gap-3 py-3", attrs: { "data-doc-id": doc.id } }, [
        checkbox,
        el("div", { class: "min-w-0 flex-1" }, [
          el("p", { class: "truncate text-sm font-medium", text: doc.filename, attrs: { title: doc.filename } }),
          el("p", { class: "text-xs text-ink/60", text: meta }),
        ]),
        el("button", {
          class: "text-sm text-danger underline underline-offset-2 hover:text-ink focus-visible:outline focus-visible:outline-2 focus-visible:outline-danger",
          text: "Delete", attrs: { type: "button", "aria-label": `Delete ${doc.filename}` },
          on: { click: () => deleteDocument(doc) },
        }),
      ]));
    }
    updateScopeNote();
  }

  function updateScopeNote() {
    const n = state.selected.size;
    $("scopeNote").textContent = n === 0
      ? "Searching all documents."
      : `Searching only ${plural(n, "selected document")}.`;
  }

  async function loadDocuments() {
    try {
      const data = await api("/documents");
      state.documents = data.documents;
      const ids = new Set(state.documents.map((d) => d.id));
      for (const id of [...state.selected]) if (!ids.has(id)) state.selected.delete(id);
      renderDocuments();
    } catch (err) {
      toast("error", `Could not load your documents. ${describeError(err)}`);
    }
  }

  async function deleteDocument(doc) {
    if (!window.confirm(`Delete "${doc.filename}"? Its content will no longer appear in answers.`)) return;
    try {
      await api(`/documents/${encodeURIComponent(doc.id)}`, { method: "DELETE" });
      toast("ok", `Deleted ${doc.filename}.`);
    } catch (err) {
      toast("error", `Could not delete ${doc.filename}. ${describeError(err)}`);
    }
    await loadDocuments();
  }

  async function uploadFiles(files) {
    for (const file of files) {
      const row = el("li", { class: "flex items-start gap-2" }, [
        el("span", { class: "spinner mt-0.5 shrink-0", attrs: { "aria-hidden": "true" } }),
        el("span", { text: `Adding ${file.name}…` }),
      ]);
      $("uploadStatus").prepend(row);
      const form = new FormData();
      form.append("file", file, file.name);
      try {
        const doc = await api("/documents", { method: "POST", body: form });
        row.replaceChildren(el("span", { class: "text-accent", text: `Added ${doc.filename}: ${plural(doc.chunk_count, "passage")} ready to search.` }));
      } catch (err) {
        const message = `${file.name} was not added. ${describeError(err)}`;
        row.replaceChildren(el("span", { class: "text-danger", text: message }));
        toast("error", message);
      }
      await loadDocuments();
    }
    while ($("uploadStatus").children.length > 5) $("uploadStatus").lastElementChild.remove();
  }

  /* ------------------------------------------------------------ answer turns */

  /** Answer text with [n] marks turned into buttons that jump to passage n. */
  function answerNodes(text, turnId, chunkCount) {
    const nodes = [];
    for (const part of text.split(/(\[\d{1,2}\])/g)) {
      const m = part.match(/^\[(\d{1,2})\]$/);
      if (m && Number(m[1]) >= 1 && Number(m[1]) <= chunkCount) {
        const n = Number(m[1]);
        nodes.push(el("button", {
          class: "mx-0.5 inline-block rounded-sm bg-mark px-1.5 align-baseline font-sans text-xs font-semibold text-ink hover:bg-ink hover:text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent",
          text: `[${n}]`,
          attrs: { type: "button", "aria-label": `Show source ${n}` },
          on: { click: () => jumpToChunk(turnId, n) },
        }));
      } else if (part) {
        nodes.push(document.createTextNode(part));
      }
    }
    return nodes;
  }

  function jumpToChunk(turnId, n) {
    const target = document.getElementById(`${turnId}-chunk-${n}`);
    if (!target) return;
    const details = target.closest("details");
    if (details) details.open = true;
    target.scrollIntoView({ behavior: "smooth", block: "center" });
    target.classList.remove("flash");
    void target.offsetWidth; // restart the animation
    target.classList.add("flash");
  }

  function scoreRow(label, value) {
    return el("div", { class: "flex items-center gap-2 text-xs text-ink/70" }, [
      el("span", { class: "w-24 shrink-0", text: label }),
      el("span", { class: "h-1.5 w-24 shrink-0 overflow-hidden rounded-full bg-rule", attrs: { "aria-hidden": "true" } },
        el("span", { class: "block h-full bg-accent", attrs: { style: `width:${(clamp01(value) * 100).toFixed(0)}%` } })),
      el("span", { class: "tabular-nums font-medium text-ink", text: Number(value).toFixed(3) }),
    ]);
  }

  function chunkNode(turnId, chunk, n) {
    const where = chunk.page ? `page ${chunk.page}` : chunk.section ? `section “${chunk.section}”` : "";
    const full = chunk.text;
    const long = full.length > 280;
    const passage = el("p", { class: "passage mt-2 font-serif text-sm leading-relaxed", text: long ? `${full.slice(0, 280)}…` : full });
    const toggle = long ? el("button", {
      class: "mt-1 text-xs text-accent underline underline-offset-2",
      text: "Show full passage", attrs: { type: "button" },
      on: {
        click: (e) => {
          const expanded = e.target.dataset.expanded === "1";
          passage.textContent = expanded ? `${full.slice(0, 280)}…` : full;
          e.target.dataset.expanded = expanded ? "0" : "1";
          e.target.textContent = expanded ? "Show full passage" : "Show less";
        },
      },
    }) : null;

    return el("li", {
      class: `py-3 pl-3 ${chunk.cited ? "border-l-4 border-mark" : "border-l-4 border-rule"}`,
      attrs: { id: `${turnId}-chunk-${n}` },
    }, [
      el("div", { class: "flex flex-wrap items-baseline gap-x-3 gap-y-1" }, [
        el("span", { class: "rounded-sm bg-mark/70 px-1.5 text-xs font-semibold", text: `[${n}]` }),
        el("span", { class: "text-sm font-medium", text: chunk.filename }),
        where ? el("span", { class: "text-xs text-ink/60", text: where }) : null,
        chunk.cited ? el("span", { class: "text-xs font-medium text-accent", text: "Used in the answer" }) : null,
      ]),
      el("div", { class: "mt-2 space-y-1" }, [
        scoreRow("Similarity", chunk.score),
        chunk.rerank_score != null ? scoreRow("Rerank score", chunk.rerank_score) : null,
      ]),
      passage, toggle,
    ]);
  }

  function chunksSection(turnId, data) {
    const summary = el("summary", { class: "text-sm font-medium" }, [
      document.createTextNode(`Retrieved chunks: ${data.chunks_fetched} fetched from the vector database, ${data.chunks_returned} used`),
      data.reranked ? el("span", { class: "ml-2 rounded-sm bg-ground px-1.5 text-xs font-normal text-ink/70", text: "reranked" }) : null,
    ]);
    const body = data.retrieved_chunks.length
      ? el("ol", { class: "mt-2 divide-y divide-rule" }, data.retrieved_chunks.map((c, i) => chunkNode(turnId, c, i + 1)))
      : el("p", { class: "mt-2 text-sm text-ink/70", text: "No passages were retrieved for this question." });
    const details = el("details", { class: "chunks border-t border-rule px-4 py-3" }, [summary, body]);
    return details;
  }

  function createTurn(question) {
    const turnId = `turn${++state.turnCounter}`;
    const answerBox = el("div", { class: "px-4 py-4" }, [
      el("div", { class: "flex items-center gap-2 text-sm text-ink/70" }, [
        el("span", { class: "spinner", attrs: { "aria-hidden": "true" } }),
        el("span", { text: "Searching your documents…" }),
      ]),
    ]);
    const sheet = el("div", { class: "mt-3 overflow-hidden rounded border border-rule bg-sheet" }, [answerBox]);
    const root = el("article", { attrs: { "aria-label": "Question and answer" } }, [
      el("h3", { class: "max-w-3xl font-serif text-xl font-semibold leading-snug", text: question }),
      sheet,
    ]);

    return {
      root,
      showResult(data) {
        const refusal = data.refused ? (REFUSALS[data.refusal_reason] || { label: "No answer", tone: "warn" }) : null;
        const toneClass = refusal ? (refusal.tone === "danger" ? "text-danger" : "text-warn") : "text-accent";
        const label = refusal ? refusal.label : "Answer";
        const statusLine = el("p", { class: `text-sm font-semibold ${toneClass}`, text: label });
        const answer = el("p", { class: "answer-text mt-2 max-w-3xl font-serif text-lg leading-relaxed" },
          answerNodes(data.answer, turnId, data.retrieved_chunks.length));

        const parts = [statusLine, answer];
        if (data.guardrail) {
          parts.push(el("p", { class: "mt-2 text-sm text-ink/70", text: `Filter category: ${data.guardrail.replace(/_/g, " ")}.` }));
        }
        if (data.citations.length) {
          parts.push(el("p", { class: "mt-3 text-sm text-ink/70", text:
            "Sources: " + data.citations.map((c) => `[${c.number}] ${c.filename}${c.page ? `, page ${c.page}` : ""}`).join("; ") }));
        }
        const meta = [
          `${(data.latency_ms / 1000).toFixed(2)} s`,
          `${data.usage.input_tokens + data.usage.output_tokens} tokens`,
          `request ${data.request_id}`,
        ];
        parts.push(el("p", { class: "mt-3 flex flex-wrap gap-x-4 text-xs tabular-nums text-ink/50" },
          meta.map((m) => el("span", { text: m }))));

        answerBox.replaceChildren(...parts);
        if (data.refusal_reason !== "no_documents" && data.refusal_reason !== "input_blocked") {
          document.querySelectorAll("#thread details.chunks").forEach((d) => { d.open = false; });
          const section = chunksSection(turnId, data);
          section.open = true;
          sheet.append(section);
        }
      },
      showError(err) {
        answerBox.replaceChildren(
          el("p", { class: "text-sm font-semibold text-danger", text: "The question could not be answered" }),
          el("p", { class: "mt-1 max-w-3xl text-base", text: describeError(err) }),
          err.requestId ? el("p", { class: "mt-2 text-xs text-ink/50", text: `request ${err.requestId}` }) : null,
        );
      },
    };
  }

  /* -------------------------------------------------------------------- ask */

  function setBusy(busy) {
    state.busy = busy;
    updateAskButton();
  }

  function updateAskButton() {
    const text = $("questionInput").value.trim();
    $("askBtn").disabled = state.busy || text.length === 0;
    $("askBtn").textContent = state.busy ? "Searching…" : "Ask";
    const n = $("questionInput").value.length;
    $("charCount").textContent = `${n} / ${MAX_QUESTION}`;
    $("charCount").className = `text-sm tabular-nums ${n > MAX_QUESTION ? "font-semibold text-danger" : "text-ink/60"}`;
  }

  async function ask(question) {
    const empty = $("emptyThread");
    if (empty) empty.remove();
    const turn = createTurn(question);
    $("thread").append(turn.root);
    turn.root.scrollIntoView({ behavior: "smooth", block: "start" });
    setBusy(true);

    const body = { question };
    if (state.selected.size) body.document_ids = [...state.selected];
    const rerank = $("rerankSelect").value;
    if (rerank !== "default") body.use_rerank = rerank === "on";
    try {
      const data = await api("/ask", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
      });
      turn.showResult(data);
    } catch (err) {
      turn.showError(err);
      if (err.status === 404) loadDocuments(); // a selected document may have been deleted
    } finally {
      setBusy(false);
    }
  }

  /* ------------------------------------------------------------------- init */

  async function checkHealth() {
    try {
      await api("/health");
      $("healthDot").className = "h-2 w-2 rounded-full bg-accent";
      $("healthText").textContent = "Server connected";
    } catch (_) {
      $("healthDot").className = "h-2 w-2 rounded-full bg-danger";
      $("healthText").textContent = "Server not reachable";
    }
  }

  function init() {
    const input = $("questionInput");
    input.addEventListener("input", updateAskButton);
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("askForm").requestSubmit(); }
    });
    $("askForm").addEventListener("submit", (e) => {
      e.preventDefault();
      const question = input.value.trim();
      if (!question || state.busy) return;
      input.value = "";
      updateAskButton();
      ask(question);
    });

    $("fileInput").addEventListener("change", (e) => {
      const files = [...e.target.files];
      e.target.value = "";
      if (files.length) uploadFiles(files);
    });
    const zone = $("dropzone");
    ["dragenter", "dragover"].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("drag-over"); }));
    ["dragleave", "drop"].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove("drag-over"); }));
    zone.addEventListener("drop", (e) => {
      const files = [...(e.dataTransfer ? e.dataTransfer.files : [])];
      if (files.length) uploadFiles(files);
    });
    $("refreshBtn").addEventListener("click", loadDocuments);

    for (const q of EXAMPLES) {
      $("examples").append(el("button", {
        class: "rounded border border-rule bg-sheet px-3 py-1.5 text-left font-serif text-sm hover:border-accent focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent",
        text: q, attrs: { type: "button" },
        on: { click: () => { input.value = q; updateAskButton(); input.focus(); } },
      }));
    }
    updateAskButton();
    checkHealth();
    loadDocuments();
  }

  // Exposed for tests only.
  window.__ragApp = { answerNodes, describeError, formatBytes };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
