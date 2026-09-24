import * as pdfjs from "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.10.38/pdf.min.mjs";
pdfjs.GlobalWorkerOptions.workerSrc = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.10.38/pdf.worker.min.mjs";

const API = "/api/v1";
const $ = (id) => document.getElementById(id);
const state = { docs: [], doc: null, pdf: null, page: 1, passages: {}, marksFor: null };

const STATE_WORDS = {
  queued: "Queued", parsing: "Reading pages", enriching: "Describing figures and tables",
  embedding: "Indexing", ready: "Ready", failed: "Failed",
};

// ---------- identity: an anonymous workspace per browser, the user's own Gemini key ----------
// The workspace token is 256 random bits and *is* the library: anyone holding it can read it,
// which is why it never leaves this browser except as a request header.
const store = {
  get: (k) => { try { return localStorage.getItem(`synaptix.${k}`) || ""; } catch { return ""; } },
  set: (k, v) => { try { v ? localStorage.setItem(`synaptix.${k}`, v) : localStorage.removeItem(`synaptix.${k}`); } catch {} },
};
function workspaceToken() {
  let t = store.get("workspace");
  if (!t) {
    const bytes = crypto.getRandomValues(new Uint8Array(32));
    t = btoa(String.fromCharCode(...bytes)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
    store.set("workspace", t);
  }
  return t;
}

function headers(extra = {}) {
  const h = { ...extra, "x-workspace": workspaceToken() };
  if (store.get("gemini")) h["x-gemini-key"] = store.get("gemini");
  if (store.get("key")) h.authorization = `Bearer ${store.get("key")}`;
  return h;
}
// Every API call goes through here, so the identity headers are added in one place.
async function call(path, opts = {}) {
  const r = await fetch(API + path, { ...opts, headers: headers(opts.headers) });
  if (r.status === 401 || r.status === 400) {
    const detail = await r.clone().json().then((d) => String(d.detail || "")).catch(() => "");
    if (/Gemini API key/.test(detail)) openSettings("Add your Gemini API key to read and ask documents.");
    else if (/invalid or missing API key/.test(detail)) {
      const key = prompt("This server requires an access key.");
      if (key) { store.set("key", key.trim()); return call(path, opts); }
    }
  }
  return r;
}

// Figures are workspace-scoped, and an <img src> cannot send headers: fetch, then show a blob.
async function hydrateImages(root) {
  for (const img of root.querySelectorAll("img[data-src]")) {
    const url = img.dataset.src;
    img.removeAttribute("data-src");
    const r = await fetch(url, { headers: headers() });
    if (r.ok) img.src = URL.createObjectURL(await r.blob());
  }
}

// ---------- settings ----------
function openSettings(message = "") {
  $("settings").open = true;
  $("settings-note").textContent = message;
  if (message) $("gemini-key").focus();
}
$("gemini-key").value = store.get("gemini");
$("workspace-token").value = workspaceToken();
$("save-key").onclick = () => {
  store.set("gemini", $("gemini-key").value.trim());
  $("settings-note").textContent = store.get("gemini") ? "Key saved in this browser only." : "Key removed.";
};
$("copy-workspace").onclick = async () => {
  await navigator.clipboard.writeText(workspaceToken()).catch(() => {});
  $("settings-note").textContent = "Workspace key copied. Paste it on another device to open this library.";
};
$("use-workspace").onclick = () => {
  const t = $("workspace-token").value.trim();
  if (t.length < 32) { $("settings-note").textContent = "A workspace key is at least 32 characters."; return; }
  store.set("workspace", t);
  location.reload();
};
async function api(path, opts = {}) {
  const r = await call(path, { headers: { "content-type": "application/json" }, ...opts });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail?.toString?.() || r.statusText);
  return r.status === 204 ? null : r.json();
}

async function loadDocs() {
  state.docs = await api("/documents");
  const ul = $("documents");
  ul.replaceChildren(...state.docs.map((d) => {
    const li = document.createElement("li");
    li.className = d.id === state.doc?.id ? "selected" : "";
    const cls = d.status === "ready" ? "ready" : d.status === "failed" ? "failed" : "working";
    li.innerHTML = `<span class="name" title="${esc(d.filename)}">${esc(d.title || d.filename)}</span>
      <span class="state ${cls}">${STATE_WORDS[d.status]}${d.page_count ? ` · ${d.page_count} pages` : ""}</span>`;
    li.onclick = () => select(d);
    return li;
  }));
  fillCompare();
  if (state.doc) {
    const fresh = state.docs.find((d) => d.id === state.doc.id);
    if (fresh && fresh.status !== state.doc.status) select(fresh);
  }
  if (state.docs.some((d) => !["ready", "failed"].includes(d.status))) setTimeout(loadDocs, 4000);
}

$("file").onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const body = new FormData();
  body.append("file", file);
  try {
    const doc = await (await call("/documents", { method: "POST", body })).json();
    if (doc.detail) throw new Error(doc.detail);
    await loadDocs();
    select(state.docs.find((d) => d.id === doc.id));
  } catch (err) { alert(`Upload failed: ${err.message}`); }
  e.target.value = "";
};

// Comparison candidates: every other ready document. Refilled on selection, not just on load.
function fillCompare() {
  const others = state.docs.filter((d) => d.status === "ready" && d.id !== state.doc?.id);
  $("compare-with").replaceChildren(...others.map((d) => new Option(d.title || d.filename, d.id)));
  $("run-comparison").disabled = !others.length;
}

async function select(doc) {
  state.doc = doc;
  fillCompare();
  $("doc-title").textContent = doc.title || doc.filename;
  document.querySelectorAll(".documents li").forEach((li, i) => li.classList.toggle("selected", state.docs[i].id === doc.id));
  $("thread").innerHTML = doc.status === "ready"
    ? `<p class="empty">Ask anything about this document.</p>`
    : `<p class="empty">${STATE_WORDS[doc.status]}${doc.error ? `: ${esc(doc.error)}` : ""}</p>`;
  $("summary").innerHTML = doc.summary ? renderMarkdown(doc.summary)
    : `<p class="empty">${doc.status === "ready" ? `<button id="summarise">Summarise this document</button>` : "Not ready yet."}</p>`;
  $("summarise")?.addEventListener("click", summarise);
  $("analysis").innerHTML = `<p class="empty">Loading analyses…</p>`;
  loadAnalyses();
  $("graph").replaceChildren();
  $("graph-stats").textContent = "";
  $("marks").replaceChildren();
  $("passage").replaceChildren();
  state.pdf = await pdfjs.getDocument({ url: `${API}/documents/${doc.id}/file`, httpHeaders: headers() }).promise;
  showPage(1);
}

// ---------- ask (streamed, cited) ----------
$("ask-form").onsubmit = async (e) => {
  e.preventDefault();
  if (!state.doc || state.doc.status !== "ready") return;
  const question = $("question").value.trim();
  if (!question) return;
  $("question").value = "";
  $("ask-btn").disabled = true;
  const thread = $("thread");
  thread.querySelector(".empty")?.remove();
  thread.insertAdjacentHTML("beforeend", `<div class="turn q">${esc(question)}</div><div class="turn a"><p class="empty">Searching…</p></div>`);
  const answerEl = thread.lastElementChild;
  thread.scrollTop = thread.scrollHeight;

  let text = "";
  let passages = [];
  try {
    const r = await call("/query", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ question, document_ids: [state.doc.id], style: $("style").value, stream: true }),
    });
    if (!r.ok) throw new Error(await r.json().then((d) => d.detail).catch(() => r.statusText));
    for await (const ev of sse(r)) {
      if (ev.event === "passages") { passages = ev.data; passages.forEach((p) => (state.passages[p.n] = p)); }
      else if (ev.event === "token") { text += ev.data; renderAnswer(answerEl, text, passages); }
      else if (ev.event === "done") renderAnswer(answerEl, text, passages, `${ev.data.candidates} candidates, ${ev.data.queries.length} query variants`);
      thread.scrollTop = thread.scrollHeight;
    }
  } catch (err) { answerEl.innerHTML = `<p>Could not answer: ${esc(err.message)}</p>`; }
  $("ask-btn").disabled = false;
};

async function* sse(response) {
  const reader = response.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, i); buf = buf.slice(i + 2);
      const event = /^event: (.*)$/m.exec(block)?.[1];
      const data = /^data: (.*)$/m.exec(block)?.[1];
      if (event && data) yield { event, data: JSON.parse(data) };
    }
  }
}

function renderAnswer(el, markdown, passages, meta) {
  const cited = new Set([...markdown.matchAll(/\[(\d{1,2})\]/g)].map((m) => +m[1]));
  const figures = passages
    .filter((p) => cited.has(p.n) && p.images?.length)
    .map((p) => `<figure><img data-src="${p.images[0]}" alt="Figure cited as ${p.n}">
      <figcaption><button class="cite" data-n="${p.n}">${p.n}</button> page ${p.page_start}</figcaption></figure>`)
    .join("");
  el.innerHTML =
    withCites(renderMarkdown(markdown), passages) +
    (figures ? `<div class="figures">${figures}</div>` : "") +
    (meta ? `<div class="meta">${meta}</div>` : "");
  hydrateImages(el);
}

// [n] markers → buttons that reveal the evidence; `passages` scopes which numbers are valid.
function withCites(html, passages) {
  const valid = new Set(passages.map((p) => p.n));
  return html.replace(/\[(\d{1,2})\]/g, (m, n) =>
    valid.has(+n) ? `<button class="cite" data-n="${n}" title="Show evidence">${n}</button>` : m);
}
document.addEventListener("click", (e) => {
  const b = e.target.closest(".cite");
  if (!b) return;
  const scope = b.closest("[data-passages]");
  const passages = scope ? JSON.parse(scope.dataset.passages) : state.passages;
  const p = Array.isArray(passages) ? passages.find((x) => x.n === +b.dataset.n) : passages[+b.dataset.n];
  if (p) reveal(p, b);
});

// ---------- evidence pane ----------
async function reveal(p, btn) {
  document.querySelectorAll(".cite.active").forEach((c) => c.classList.remove("active"));
  btn?.classList.add("active");
  const path = p.heading_path?.[0] === p.title ? p.heading_path.slice(1) : p.heading_path || [];
  const where = [p.title, path.join(" › "), p.page_start === p.page_end ? `p. ${p.page_start}` : `pp. ${p.page_start}–${p.page_end}`].filter(Boolean).join(" — ");
  $("passage").innerHTML = `<div class="where">${esc(where)}</div>${renderMarkdown(p.text || "")}`;
  if (p.document_id !== state.doc?.id) {
    const d = state.docs.find((x) => x.id === p.document_id);
    if (d) await select(d);
  }
  const elements = (await Promise.all(p.chunk_ids.map((c) => api(`/documents/${p.document_id}/chunks/${c}`)))).flat();
  const boxes = elements.filter((el) => el.bbox);
  const page = boxes[0]?.page ?? p.page_start;
  state.marksFor = boxes;
  await showPage(page);
}

async function showPage(n) {
  if (!state.pdf) return;
  n = Math.max(1, Math.min(n, state.pdf.numPages));
  state.page = n;
  const page = await state.pdf.getPage(n);
  const paneWidth = $("stage").parentElement.clientWidth - 32;
  const scale = Math.min(1.6, paneWidth / page.getViewport({ scale: 1 }).width);
  const viewport = page.getViewport({ scale });
  const canvas = $("canvas");
  canvas.width = viewport.width; canvas.height = viewport.height;
  await page.render({ canvasContext: canvas.getContext("2d"), viewport }).promise;
  $("page-label").textContent = `Page ${n} of ${state.pdf.numPages}`;
  drawMarks(page, viewport);
}

// PyMuPDF bboxes are top-left-origin points relative to the crop box; pdf.js user space is
// bottom-left-origin, so convert through the page's view box and the viewport transform.
function drawMarks(page, viewport) {
  const [vx0, , , vy1] = page.view;
  const marks = $("marks");
  marks.replaceChildren();
  for (const el of state.marksFor || []) {
    if (el.page !== state.page) continue;
    const [x0, y0, x1, y1] = el.bbox;
    const [ax, ay] = viewport.convertToViewportPoint(vx0 + x0, vy1 - y0);
    const [bx, by] = viewport.convertToViewportPoint(vx0 + x1, vy1 - y1);
    const m = document.createElement("div");
    m.className = "mark";
    m.style.cssText = `left:${Math.min(ax, bx)}px;top:${Math.min(ay, by)}px;width:${Math.abs(bx - ax)}px;height:${Math.abs(by - ay)}px`;
    marks.append(m);
    requestAnimationFrame(() => m.classList.add("on"));
  }
  const first = marks.firstElementChild;
  if (first) {
    const scroller = $("stage").parentElement;  // scroll only the page pane, never the window
    const top = first.offsetTop - scroller.clientHeight / 2 + first.offsetHeight / 2;
    scroller.scrollTo({ top, behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  }
}
$("prev").onclick = () => showPage(state.page - 1);
$("next").onclick = () => showPage(state.page + 1);

// ---------- summary ----------
async function summarise() {
  $("summary").innerHTML = `<p class="empty">Summarising…</p>`;
  try {
    const { summary } = await api(`/documents/${state.doc.id}/summary`, { method: "POST" });
    state.doc.summary = summary;
    $("summary").innerHTML = renderMarkdown(summary);
  } catch (err) { $("summary").innerHTML = `<p class="empty">Could not summarise: ${esc(err.message)}</p>`; }
}

// ---------- analysis ----------
const STAGE_TITLES = { outline: "Study outline", practice: "Practice questions", comparison: "Comparison" };
const FIELD_TITLES = {
  key_concepts: "Key concepts", formulas: "Formulas", worked_examples: "Worked examples",
  pitfalls: "Watch out for", questions: "Questions", data_and_setup: "Data and setup",
  strengths_and_limitations: "Strengths and limitations",
};
// Each schema field renders to its own shape; anything unlisted falls back to `.text`.
const ITEM = {
  key_concepts: (c) => `<strong>${esc(c.name)}</strong> — ${inline(c.explanation)}`,
  formulas: (f) => `<strong>${esc(f.name)}</strong>${renderMarkdown(`$$${f.latex}$$`)}
    <span class="symbols">${inline(f.symbols)}</span>`,
  questions: (q) => `<div class="question">${inline(q.question)}
    <span class="tag">${esc(q.difficulty)}</span></div><div class="answer">${inline(q.answer)}</div>`,
};
const inline = (md) => renderMarkdown(md).replace(/^<p>/, "").replace(/<\/p>\s*$/, "");

async function loadAnalyses() {
  if (!state.doc) return;
  const runs = await api(`/analyses?document_id=${state.doc.id}`);
  const box = $("analysis");
  if (!runs.length) { box.innerHTML = `<p class="empty">No study guides yet for this document.</p>`; return; }
  box.replaceChildren(...runs.map(renderRun));
  if (runs.some((r) => ["queued", "running"].includes(r.status))) setTimeout(loadAnalyses, 5000);
}

function renderRun(run) {
  const div = document.createElement("div");
  div.className = "run";
  const when = new Date(run.created_at).toLocaleString();
  const kind = run.kind === "study_guide" ? "Study guide" : "Comparison";
  if (run.status !== "done") {
    div.innerHTML = `<div class="head">${kind}, ${when}</div><p class="empty">${
      run.status === "failed" ? `Failed: ${esc(run.error)}` : "Working through the document…"
    }</p>`;
    return div;
  }
  let html = `<div class="head">${kind}, ${when}</div>`;
  for (const [stage, { output, passages }] of Object.entries(run.result)) {
    html += `<section data-passages='${esc(JSON.stringify(passages))}'>`;
    html += `<h3>${STAGE_TITLES[stage] || stage}</h3>`;
    for (const [field, value] of Object.entries(output)) {
      if (typeof value === "string") { html += inline(value); continue; }
      if (!value.length) continue;
      const item = ITEM[field] || ((x) => inline(x.text ?? String(x)));
      html += `<h4>${FIELD_TITLES[field] || field.replaceAll("_", " ")}</h4><ul>`;
      html += value.map((x) => `<li>${item(x)}${cites(x.evidence)}</li>`).join("");
      html += "</ul>";
    }
    html += "</section>";
  }
  div.innerHTML = html;
  return div;
}

const cites = (ev) =>
  ev?.length ? " " + ev.map((n) => `<button class="cite" data-n="${n}">${n}</button>`).join("") : "";

$("run-study-guide").onclick = () => startAnalysis({ kind: "study_guide", document_ids: [state.doc.id] });
$("run-comparison").onclick = () => {
  const other = $("compare-with").value;
  if (other) startAnalysis({ kind: "comparison", document_ids: [state.doc.id, other] });
};
async function startAnalysis(body) {
  if (!state.doc || state.doc.status !== "ready") return;
  try { await api("/analyses", { method: "POST", body: JSON.stringify(body) }); loadAnalyses(); }
  catch (err) { alert(`Could not start: ${err.message}`); }
}

// ---------- graph ----------
const TYPE_COLOR = { organisation: "#1f4e79", person: "#7a3e9d", product: "#2e7d4f", technology: "#0f766e", method: "#0f766e", metric: "#b45309", location: "#6b7280", concept: "#9ca3af" };
$("build-graph").onclick = async () => {
  if (!state.doc || state.doc.status !== "ready") return;
  $("graph-stats").textContent = "Extracting entities…";
  try {
    const g = await api(`/documents/${state.doc.id}/graph`, { method: "POST" });
    $("graph-stats").textContent = `${g.nodes.length} entities, ${g.edges.length} relations`;
    const nodes = g.nodes.map((n) => ({ id: n.id, label: n.name, value: n.mentions, color: TYPE_COLOR[n.type] || TYPE_COLOR.concept, title: n.type }));
    const edges = g.edges.map((e) => ({ from: e.source, to: e.target, label: e.predicate, title: e.pages ? `pp. ${e.pages[0]}–${e.pages[1]}` : "" }));
    new vis.Network($("graph"), { nodes, edges }, {
      nodes: { shape: "dot", scaling: { min: 8, max: 26 }, font: { face: "IBM Plex Sans", size: 13, color: "#1f2328" }, borderWidth: 0 },
      edges: { color: { color: "#9aa3ad" }, font: { face: "IBM Plex Sans", size: 10, color: "#5b6470", strokeWidth: 0 }, arrows: "to", smooth: { type: "continuous" } },
      physics: { stabilization: { iterations: 200 }, barnesHut: { gravitationalConstant: -6000, springLength: 140 } },
      interaction: { hover: true },
    });
  } catch (err) { $("graph-stats").textContent = `Could not build graph: ${err.message}`; }
};

// ---------- tabs & utils ----------
$("tabs").onclick = (e) => {
  const tab = e.target.dataset.tab;
  if (!tab) return;
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("active", b === e.target));
  document.querySelectorAll(".tab").forEach((s) => s.classList.toggle("active", s.id === `tab-${tab}`));
};
// Markdown eats maths (`V_m \sin` turns into italics), so maths spans are lifted out before
// marked runs and rendered back in with KaTeX afterwards.
const MATH_RE = [
  [/\$\$([\s\S]+?)\$\$/g, true],
  [/\\\[([\s\S]+?)\\\]/g, true],
  [/\\\(([\s\S]+?)\\\)/g, false],
  [/\$(?!\s)([^\n$]+?)(?<!\s)\$/g, false],
];
function renderMarkdown(md) {
  const spans = [];
  let src = String(md ?? "");
  for (const [re, display] of MATH_RE) {
    src = src.replace(re, (_, tex) => `MATHSPAN${spans.push({ tex, display }) - 1}ENDSPAN`);
  }
  return marked.parse(src).replace(/MATHSPAN(\d+)ENDSPAN/g, (whole, i) => {
    const span = spans[i];
    if (!span) return whole;
    try {
      return katex.renderToString(span.tex, { displayMode: span.display, throwOnError: false });
    } catch {
      return esc(span.tex);
    }
  });
}

function esc(s) { return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

if (!store.get("gemini")) openSettings("This app runs on your own free Gemini API key. Add it to start.");
loadDocs();
