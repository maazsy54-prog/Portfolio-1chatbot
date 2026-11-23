const { JSDOM } = require("jsdom");
const fs = require("fs");
const ROOT = require("path").join(__dirname, "..");
let html = fs.readFileSync(`${ROOT}/index.html`, "utf8")
  .replace(/<script src="https:[^>]*><\/script>/g, "")
  .replace(/<script>\s*tailwind\.config[\s\S]*?<\/script>/, "")
  .replace(/<link[^>]*>/g, "").replace(/<script src="app.js"[^>]*><\/script>/, "");
const js = fs.readFileSync(`${ROOT}/app.js`, "utf8");

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let passed = 0, failed = 0;
function check(name, cond, extra = "") {
  if (cond) { passed++; console.log("  ok  ", name); } else { failed++; console.log("  FAIL", name, extra); }
}

(async () => {
  const dom = new JSDOM(html, { runScripts: "outside-only", pretendToBeVisual: true });
  const { window } = dom; const { document } = window;
  window.Element.prototype.scrollIntoView = function () { this.__scrolled = true; };
  window.confirm = () => true;

  // ---------- mocked API ----------
  let docs = []; const asks = []; let dupNext = false;
  const resp = (status, data, headers = {}) => ({ ok: status < 400, status, headers: { get: (k) => headers[k] ?? null }, json: async () => data });
  const chunk = (i, text, cited, extra = {}) => ({ id: `d:${i}`, document_id: "d", filename: "nimbus_faq.pdf", page: 1, section: null, position: i, text, score: 0.71 - i * 0.1, rerank_score: 0.93 - i * 0.2, cited, ...extra });
  window.fetch = async (path, opts = {}) => {
    const method = opts.method || "GET";
    if (path === "/health") return resp(200, { status: "ok" });
    if (path === "/documents" && method === "GET") return resp(200, { documents: docs, total: docs.length });
    if (path === "/documents" && method === "POST") {
      if (dupNext) return resp(409, { error: { code: "duplicate_document", message: "This document has already been uploaded.", request_id: "r1" } });
      const f = opts.body.get("file");
      const d = { id: "d", filename: f.name, file_type: "pdf", size_bytes: 48213, page_count: 4, chunk_count: 12, status: "ready", created_at: "2026-11-22T00:00:00Z" };
      docs = [d]; return resp(201, d);
    }
    if (path.startsWith("/documents/") && method === "DELETE") { docs = []; return resp(204, null); }
    if (path === "/ask") {
      const body = JSON.parse(opts.body); asks.push(body); const q = body.question;
      if (q.length > 1000) return resp(422, { error: { code: "validation_error", message: "Invalid request data.", request_id: "r2" } });
      if (q.includes("ratelimit")) return resp(429, { error: { code: "rate_limited", message: "Rate limit exceeded", request_id: "r3" } }, { "Retry-After": "12" });
      if (q.includes("down")) return resp(503, { error: { code: "service_unavailable", message: "Vector database is temporarily unavailable.", request_id: "r4" } });
      const base = { citations: [], retrieved_chunks: [], chunks_fetched: 20, chunks_returned: 2, reranked: true, usage: { input_tokens: 900, output_tokens: 40, llm_calls: 1 }, request_id: "abc123", latency_ms: 1840, guardrail: null, refused: false, refusal_reason: null };
      if (q.includes("blocked")) return resp(200, { ...base, answer: "I can't help with that request.", refused: true, refusal_reason: "input_blocked", guardrail: "prompt_injection", chunks_fetched: 0, chunks_returned: 0 });
      if (q.includes("refuse")) return resp(200, { ...base, answer: "The documents do not cover that.", refused: true, refusal_reason: "not_in_documents", retrieved_chunks: [chunk(0, "Unrelated text about shipping.", false)] });
      if (q.includes("xss")) return resp(200, { ...base, answer: "Safe <img src=x onerror=\"window.__pwned=1\"> text [1]", citations: [{ number: 1, chunk_id: "d:0", document_id: "d", filename: "<b>evil</b>.pdf", page: 1, section: null }], retrieved_chunks: [chunk(0, "<script>window.__pwned=2</script><img src=x onerror=\"window.__pwned=3\">", true, { filename: "<b>evil</b>.pdf" })] });
      return resp(200, { ...base, answer: "Standard shipping takes 5 to 7 business days [1]. Express takes 2 days [2].",
        citations: [{ number: 1, chunk_id: "d:0", document_id: "d", filename: "nimbus_faq.pdf", page: 1, section: null }, { number: 2, chunk_id: "d:1", document_id: "d", filename: "nimbus_faq.pdf", page: 1, section: null }],
        retrieved_chunks: [chunk(0, "Q1. How long does standard shipping take? " + "x".repeat(400), true), chunk(1, "Express shipping takes 2 days.", true)] });
    }
    return resp(404, { error: { code: "not_found", message: "nope", request_id: "r" } });
  };

  window.eval(js);
  await sleep(60);
  const $ = (id) => document.getElementById(id);
  const type = (text) => { $("questionInput").value = text; $("questionInput").dispatchEvent(new window.Event("input")); };
  const submit = async () => { $("askForm").dispatchEvent(new window.Event("submit", { cancelable: true })); await sleep(60); };
  const turns = () => [...document.querySelectorAll("#thread article")];
  const last = () => turns().at(-1);

  console.log("startup");
  check("health shows connected", $("healthText").textContent === "Server connected");
  check("empty-state message visible", $("docEmpty").hidden === false);
  check("example question chips rendered", $("examples").children.length === 4);
  check("Ask disabled when empty", $("askBtn").disabled === true);
  document.querySelector("#examples button").click();
  check("clicking an example fills the box", $("questionInput").value.startsWith("How long") && !$("askBtn").disabled);
  type("");

  console.log("upload");
  const file = new window.File(["%PDF-"], "nimbus_faq.pdf", { type: "application/pdf" });
  Object.defineProperty($("fileInput"), "files", { value: [file], configurable: true });
  $("fileInput").dispatchEvent(new window.Event("change")); await sleep(80);
  check("document appears in the list", $("docList").children.length === 1 && $("docList").textContent.includes("nimbus_faq.pdf"));
  check("meta line shows pages/passages/size", $("docList").textContent.includes("4 pages") && $("docList").textContent.includes("12 passages") && $("docList").textContent.includes("47 KB"));
  check("upload status says added", $("uploadStatus").textContent.includes("Added nimbus_faq.pdf"));
  check("count updated", $("docCount").textContent === "(1)" && $("docEmpty").hidden === true);
  dupNext = true;
  Object.defineProperty($("fileInput"), "files", { value: [file], configurable: true });
  $("fileInput").dispatchEvent(new window.Event("change")); await sleep(80);
  check("duplicate shows status 409 message", $("uploadStatus").textContent.includes("was not added") && $("uploadStatus").textContent.includes("HTTP 409"));
  check("error toast shown", $("toasts").textContent.includes("already been uploaded"));
  dupNext = false;

  console.log("ask / citations / retrieved chunks");
  type("How long does standard shipping take?"); await submit();
  let t = last();
  check("question shown as heading", t.querySelector("h3").textContent === "How long does standard shipping take?");
  check("status line says Answer", t.textContent.includes("Answer"));
  const chips = [...t.querySelectorAll(".answer-text button")];
  check("two citation chips in the answer", chips.length === 2 && chips[0].textContent === "[1]" && chips[1].textContent === "[2]");
  check("sources line lists files and pages", t.textContent.includes("Sources: [1] nimbus_faq.pdf, page 1; [2] nimbus_faq.pdf, page 1"));
  check("retrieved-chunks summary has fetched + used counts", t.textContent.includes("Retrieved chunks: 20 fetched from the vector database, 2 used"));
  check("reranked badge", t.textContent.includes("reranked"));
  check("each chunk shows similarity score", t.textContent.includes("Similarity") && t.textContent.includes("0.710"));
  check("each chunk shows rerank score", t.textContent.includes("Rerank score") && t.textContent.includes("0.930"));
  const items = t.querySelectorAll("details ol > li");
  check("two chunk items, cited ones marked", items.length === 2 && items[0].className.includes("border-mark") && t.textContent.includes("Used in the answer"));
  check("long passage truncated with toggle", t.textContent.includes("Show full passage"));
  [...t.querySelectorAll("button")].find((b) => b.textContent === "Show full passage").click();
  check("toggle expands the passage", items[0].textContent.includes("x".repeat(400)));
  chips[1].click();
  const target = document.getElementById("turn1-chunk-2");
  check("clicking [2] scrolls to chunk 2 and flashes it", target.__scrolled === true && target.classList.contains("flash"));
  check("meta shows latency, tokens, request id", t.textContent.includes("1.84 s") && t.textContent.includes("940 tokens") && t.textContent.includes("request abc123"));
  check("input cleared and Ask disabled after submit", $("questionInput").value === "" && $("askBtn").disabled);

  console.log("request options");
  $("docList").querySelector("input[type=checkbox]").click(); // select doc
  $("docList").querySelector("input[type=checkbox]").dispatchEvent(new window.Event("change"));
  check("scope note updates", $("scopeNote").textContent.includes("only 1 selected document"));
  $("rerankSelect").value = "off";
  type("scoped question"); await submit();
  check("document_ids and use_rerank sent", JSON.stringify(asks.at(-1).document_ids) === '["d"]' && asks.at(-1).use_rerank === false);
  $("docList").querySelector("input[type=checkbox]").click(); $("docList").querySelector("input[type=checkbox]").dispatchEvent(new window.Event("change"));
  $("rerankSelect").value = "default";
  type("plain question"); await submit();
  check("defaults send neither option", asks.at(-1).document_ids === undefined && asks.at(-1).use_rerank === undefined);
  check("only newest turn keeps chunks open", [...document.querySelectorAll("#thread details.chunks")].filter((d) => d.open).length === 1);

  console.log("refusals, guardrail, errors");
  type("please refuse this"); await submit(); t = last();
  check("refusal labelled and styled amber", t.textContent.includes("Not found in your documents") && t.querySelector("p.text-warn"));
  check("refusal still shows retrieved chunks", t.textContent.includes("Retrieved chunks: 20 fetched"));
  type("blocked attack"); await submit(); t = last();
  check("guardrail block labelled in danger style", t.textContent.includes("Blocked by the safety filter") && t.querySelector("p.text-danger"));
  check("guardrail category shown", t.textContent.includes("Filter category: prompt injection"));
  check("no chunks section for blocked input", !t.querySelector("details"));
  type("ratelimit"); await submit();
  check("429 shows retry guidance", last().textContent.includes("Wait 12 seconds"));
  type("down service"); await submit();
  check("503 explained", last().textContent.includes("backend service is not responding") && last().textContent.includes("request r4"));
  type("a".repeat(1001));
  check("counter turns red over the limit", $("charCount").className.includes("text-danger") && $("charCount").textContent === "1001 / 1000");
  await submit();
  check("overlong question reaches server and 422 is shown", last().textContent.includes("Invalid request data") && last().textContent.includes("HTTP 422"));

  console.log("security");
  type("xss attempt"); await submit(); t = last();
  check("no element injected from answer or passage", !t.querySelector("img") && !t.querySelector("script") && !t.querySelector("b"));
  check("no injected script ran", window.__pwned === undefined);
  check("hostile text shown literally", t.textContent.includes("<script>window.__pwned=2</script>") && t.textContent.includes("<b>evil</b>.pdf"));

  console.log("keyboard + delete");
  type("enter key question");
  $("questionInput").dispatchEvent(new window.KeyboardEvent("keydown", { key: "Enter", bubbles: true, cancelable: true })); await sleep(80);
  check("Enter submits", asks.at(-1).question === "enter key question");
  const before = asks.length;
  type("shift enter question");
  $("questionInput").dispatchEvent(new window.KeyboardEvent("keydown", { key: "Enter", shiftKey: true, bubbles: true, cancelable: true })); await sleep(40);
  check("Shift+Enter does not submit", asks.length === before);
  [...$("docList").querySelectorAll("button")].find((b) => b.textContent === "Delete").click(); await sleep(80);
  check("delete removes the document", $("docList").children.length === 0 && $("docEmpty").hidden === false && $("docCount").textContent === "(0)");
  check("delete toast", $("toasts").textContent.includes("Deleted nimbus_faq.pdf"));

  console.log(`\n${passed} passed, ${failed} failed`);
  process.exit(failed ? 1 : 0);
})().catch((e) => { console.error("CRASH", e); process.exit(2); });
