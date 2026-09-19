import * as pdfjs from "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.10.38/pdf.min.mjs";
pdfjs.GlobalWorkerOptions.workerSrc = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.10.38/pdf.worker.min.mjs";

const API = "/api/v1";
const $ = (id) => document.getElementById(id);
const state = { docs: [], doc: null, pdf: null, page: 1, passages: {}, marksFor: null };

const STATE_WORDS = {
  queued: "Queued", parsing: "Reading pages", enriching: "Describing figures and tables",
  embedding: "Indexing", ready: "Ready", failed: "Failed",
};

// ---------- api ----------
const call = (path, opts = {}) => fetch(API + path, opts);

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

async function select(doc) {
  state.doc = doc;
  $("doc-title").textContent = doc.title || doc.filename;
  document.querySelectorAll(".documents li").forEach((li, i) => li.classList.toggle("selected", state.docs[i].id === doc.id));
  $("thread").innerHTML = doc.status === "ready"
    ? `<p class="empty">Ask anything about this document.</p>`
    : `<p class="empty">${STATE_WORDS[doc.status]}${doc.error ? `: ${esc(doc.error)}` : ""}</p>`;
  $("marks").replaceChildren();
  $("passage").replaceChildren();
  state.pdf = await pdfjs.getDocument(`${API}/documents/${doc.id}/file`).promise;
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
      body: JSON.stringify({ question, document_ids: [state.doc.id], stream: true }),
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
  el.innerHTML = withCites(renderMarkdown(markdown), passages) + (meta ? `<div class="meta">${meta}</div>` : "");
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
  const p = state.passages[+b.dataset.n];
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

// ---------- utils ----------
function renderMarkdown(md) { return marked.parse(String(md ?? "")); }

function esc(s) { return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

loadDocs();
