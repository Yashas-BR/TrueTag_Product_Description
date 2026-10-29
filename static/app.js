"use strict";

const $ = (id) => document.getElementById(id);
const file = $("file"), drop = $("drop"), preview = $("preview"), go = $("go"), hint = $("hint");
let chosen = null;
let ready = false;

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x) => `${Math.round(x * 100)}%`;

// ---- model status --------------------------------------------------------------
async function poll() {
  try {
    const r = await fetch("/api/health");
    const h = await r.json();
    const s = $("status");
    if (h.error) {
      s.textContent = "Model error";
      s.className = "status status-bad";
      s.title = h.error;
      return;
    }
    if (h.ready) {
      ready = true;
      s.textContent = `Ready · ${h.device === "cuda" ? "GPU" : "CPU"}`;
      s.className = "status status-ok";
      s.title = `${h.extractor} + ${h.llm}`;
      refresh();
      return;
    }
  } catch (e) { /* server starting */ }
  setTimeout(poll, 2500);
}

function refresh() {
  go.disabled = !(chosen && ready);
  hint.textContent = !chosen ? "Choose a photo to begin."
    : !ready ? "Models are still loading (about a minute on first start)…"
    : "Takes about 5–20 seconds on a CPU.";
}

// ---- choosing a photo ------------------------------------------------------------
function choose(f) {
  if (!f || !f.type.startsWith("image/")) return;
  chosen = f;
  preview.src = URL.createObjectURL(f);
  preview.hidden = false;
  $("drop-text").hidden = true;
  refresh();
}
file.addEventListener("change", () => choose(file.files[0]));
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => choose(e.dataTransfer.files[0]));
document.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (item) choose(item.getAsFile());
});

// ---- rendering --------------------------------------------------------------------
function show(which) {
  for (const id of ["empty", "busy", "result", "error"]) $(id).hidden = id !== which;
}

function attrCard(name, score, value) {
  const unknown = value === "unknown";
  const t = score.threshold;
  const mark = t == null ? "" : `<span class="mark" style="left:calc(${Math.min(t, 1) * 100}% - 1px)" title="threshold ${pct(t)}"></span>`;
  const meta = unknown
    ? `Best guess “${esc(score.pred)}” at ${pct(score.confidence)}, below the ${t == null ? "" : pct(t) + " "}bar → not mentioned`
    : `${pct(score.confidence)} confident`;
  return `<div class="attr ${unknown ? "unknown" : ""}">
    <div class="name">${esc(name)}</div>
    <div class="value">${unknown ? "unknown" : esc(value)}</div>
    <div class="bar"><span class="fill" style="width:${pct(score.confidence)}"></span>${mark}</div>
    <div class="meta">${meta}</div>
  </div>`;
}

function guardHTML(res) {
  const g = res.guard;
  const out = [];
  if (res.mode === "strict") {
    out.push(`<div class="badge ok"><b>✓ Safe by construction</b><span class="body">Attribute words were inserted by the system from the detected values${g.fallback ? "; the template fallback was used" : ""}.</span></div>`);
  } else if (g.replaced) {
    const list = g.problems.map((p) => `${esc(p.attribute)}: “${esc(p.stated.join(", "))}” ${esc(p.reason)}`).join("; ");
    out.push(`<div class="badge warn"><b>Corrected</b><span class="body">The writer's first draft had an unsupported attribute (${list}). It was rebuilt in strict mode.<br><s>${esc(g.original)}</s></span></div>`);
  } else if (g.problems && g.problems.length) {
    const list = g.problems.map((p) => `${esc(p.attribute)}: “${esc(p.stated.join(", "))}” ${esc(p.reason)}`).join("; ");
    out.push(`<div class="badge bad"><b>Unsupported</b><span class="body">${list}. Safe mode would have corrected this.</span></div>`);
  } else {
    out.push(`<div class="badge ok"><b>✓ Verified</b><span class="body">Every garment type, color and fabric in the text matches what was detected in the photo.</span></div>`);
  }
  if (g.dropped_sentences && g.dropped_sentences.length) {
    const n = g.dropped_sentences.length;
    out.push(`<div class="badge warn"><b>Trimmed</b><span class="body">Removed ${n} sentence${n > 1 ? "s" : ""} about fabric, which the photo doesn't show clearly.<br><s>${g.dropped_sentences.map(esc).join(" ")}</s></span></div>`);
  }
  if (res.mode === "forced") {
    const risky = Object.entries(res.scores).filter(([a, s]) => res.evidence[a] && s.threshold != null && s.confidence < s.threshold).map(([a]) => a);
    if (risky.length) out.push(`<div class="badge warn"><b>Risky</b><span class="body">Without abstention, low-confidence guesses were used for: ${risky.map(esc).join(", ")}.</span></div>`);
  }
  return out.join("");
}

function techHTML(res) {
  const rows = Object.entries(res.scores).map(([a, s]) =>
    `<tr><td>${esc(a)}</td><td>${s.top.map((t) => `${esc(t.label)} ${pct(t.p)}`).join(", ")}</td><td>${s.threshold == null ? "always abstain" : pct(s.threshold)}</td><td>${esc(res.evidence[a])}</td></tr>`).join("");
  const m = res.guard.mentions || {};
  return `<table><thead><tr><th>Attribute</th><th>Top 3 guesses</th><th>Threshold</th><th>Sent to writer</th></tr></thead><tbody>${rows}</tbody></table>
    <p class="muted">Attributes found in the text: category ${esc((m.category || []).join(", ") || "none")}, color ${esc((m.color || []).join(", ") || "none")}, material ${esc((m.material || []).join(", ") || "none")}.
    Time: reading ${res.timings.extract_s}s, writing ${res.timings.write_s}s.</p>`;
}

function render(res) {
  const names = { category: "Garment type", color: "Color", material: "Fabric" };
  $("attrs").innerHTML = Object.keys(names).map((a) => attrCard(names[a], res.scores[a], res.evidence[a])).join("");
  $("desc").textContent = res.description;
  $("guard").innerHTML = guardHTML(res);
  $("tech").innerHTML = techHTML(res);
  show("result");
}

// ---- submit -------------------------------------------------------------------------
go.addEventListener("click", async () => {
  if (!chosen) return;
  const mode = document.querySelector("input[name=mode]:checked").value;
  const form = new FormData();
  form.append("image", chosen);
  form.append("mode", mode);
  go.disabled = true;
  show("busy");
  const steps = ["Reading the photo…", "Deciding what is certain…", "Writing the description…", "Checking every attribute…"];
  let i = 0;
  $("busy-text").textContent = steps[0];
  const ticker = setInterval(() => { i = Math.min(i + 1, steps.length - 1); $("busy-text").textContent = steps[i]; }, 2500);
  try {
    const r = await fetch("/api/describe", { method: "POST", body: form });
    const body = await r.json();
    if (!r.ok) throw new Error(body.detail || `Request failed (${r.status})`);
    render(body);
  } catch (e) {
    $("error").textContent = e.message;
    show("error");
  } finally {
    clearInterval(ticker);
    refresh();
  }
});

$("copy").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText($("desc").textContent);
    $("copy").textContent = "Copied";
    setTimeout(() => ($("copy").textContent = "Copy"), 1500);
  } catch (e) { /* clipboard blocked */ }
});

poll();
