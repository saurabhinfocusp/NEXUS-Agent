"use strict";

/* ---------------------------------------------------------------------
 * Small persisted bits of client state: which runs this browser has
 * submitted (the API has no "list runs" endpoint, so this is the only
 * way to come back to a past run without knowing its UUID by heart)
 * and the reviewer's display name (so they don't retype it per claim).
 * ------------------------------------------------------------------- */
const RECENT_RUNS_KEY = "nexus.recentRuns";
const REVIEWER_NAME_KEY = "nexus.reviewerName";
const MAX_RECENT_RUNS = 8;

function getRecentRuns() {
  try {
    return JSON.parse(localStorage.getItem(RECENT_RUNS_KEY)) || [];
  } catch (err) {
    return [];
  }
}

function saveRecentRun(runId, sampleId) {
  const runs = getRecentRuns().filter((r) => r.run_id !== runId);
  runs.unshift({ run_id: runId, sample_id: sampleId });
  localStorage.setItem(RECENT_RUNS_KEY, JSON.stringify(runs.slice(0, MAX_RECENT_RUNS)));
}

function getReviewerName() {
  return localStorage.getItem(REVIEWER_NAME_KEY) || "";
}

function setReviewerName(name) {
  if (name) localStorage.setItem(REVIEWER_NAME_KEY, name);
}

/* ---------------------------------------------------------------------
 * Tabs -- WAI-ARIA tabs pattern: one tab has tabindex=0 (roving tabindex),
 * Left/Right/Home/End move focus between tabs and activate the target,
 * matching how a screen-reader or keyboard-only user expects a tablist
 * to behave (not just a row of buttons).
 * ------------------------------------------------------------------- */
function initTabs() {
  const buttons = Array.from(document.querySelectorAll(".tab-btn"));

  const activate = (btn) => {
    buttons.forEach((b) => {
      const selected = b === btn;
      b.setAttribute("aria-selected", String(selected));
      b.tabIndex = selected ? 0 : -1;
    });
    document.querySelectorAll(".view").forEach((v) => (v.hidden = true));
    document.getElementById(btn.dataset.view).hidden = false;
    if (btn.dataset.view === "queue-view") {
      loadEscalations();
      loadCorrections();
    }
  };

  buttons.forEach((btn, i) => {
    btn.addEventListener("click", () => activate(btn));
    btn.addEventListener("keydown", (event) => {
      const moves = { ArrowRight: 1, ArrowLeft: -1, Home: -Infinity, End: Infinity };
      if (!(event.key in moves)) return;
      event.preventDefault();
      const delta = moves[event.key];
      const nextIndex = Number.isFinite(delta)
        ? (i + delta + buttons.length) % buttons.length
        : delta > 0 ? buttons.length - 1 : 0;
      buttons[nextIndex].focus();
      activate(buttons[nextIndex]);
    });
  });
}

/* ---------------------------------------------------------------------
 * Dropzones
 * ------------------------------------------------------------------- */
function initDropzone(dropId, inputId) {
  const drop = document.getElementById(dropId);
  const input = document.getElementById(inputId);
  const defaultText = drop.textContent;

  const showFile = () => {
    if (input.files.length > 0) {
      drop.textContent = input.files[0].name;
      drop.classList.add("has-file");
    } else {
      drop.textContent = defaultText;
      drop.classList.remove("has-file");
    }
  };

  input.addEventListener("change", showFile);

  ["dragover", "dragleave", "drop"].forEach((evt) => {
    drop.addEventListener(evt, (e) => e.preventDefault());
  });
  drop.addEventListener("dragover", () => drop.classList.add("drag-over"));
  drop.addEventListener("dragleave", () => drop.classList.remove("drag-over"));
  drop.addEventListener("drop", (e) => {
    drop.classList.remove("drag-over");
    if (e.dataTransfer.files.length > 0) {
      input.files = e.dataTransfer.files;
      showFile();
    }
  });
}

/* ---------------------------------------------------------------------
 * Submit & Track
 * ------------------------------------------------------------------- */
let pollHandle = null;

function initUploadForm() {
  const form = document.getElementById("upload-form");
  const errorBanner = document.getElementById("submit-error");
  const submitBtn = document.getElementById("submit-btn");

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    errorBanner.hidden = true;
    submitBtn.disabled = true;
    submitBtn.textContent = "Submitting…";

    const sampleId = document.getElementById("sample_id").value;
    const formData = new FormData();
    formData.append("sample_id", sampleId);
    formData.append("image", document.getElementById("image").files[0]);
    formData.append("expression", document.getElementById("expression").files[0]);

    try {
      const response = await fetch("/api/runs", { method: "POST", body: formData });
      if (!response.ok) {
        throw new Error(`${response.status} ${await response.text()}`);
      }
      const { run_id } = await response.json();
      saveRecentRun(run_id, sampleId);
      renderRecentRuns();
      trackRun(run_id);
    } catch (err) {
      errorBanner.textContent = `Upload failed: ${err.message}`;
      errorBanner.hidden = false;
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = "Run pipeline";
    }
  });

  document.getElementById("new-run-btn").addEventListener("click", () => {
    clearInterval(pollHandle);
    document.getElementById("upload-card").hidden = false;
    document.getElementById("new-run-btn").hidden = true;
    document.getElementById("status-card").hidden = true;
    document.getElementById("report-card").hidden = true;
    document.getElementById("claims-card").hidden = true;
    setActiveChip(null);
  });

  document.getElementById("load-run-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const input = document.getElementById("load-run-id");
    const runId = input.value.trim();
    if (runId) trackRun(runId);
    input.value = "";
  });
}

function trackRun(runId) {
  clearInterval(pollHandle);
  document.getElementById("upload-card").hidden = true;
  document.getElementById("new-run-btn").hidden = false;
  document.getElementById("report-card").hidden = true;
  document.getElementById("claims-card").hidden = true;
  setActiveChip(runId);
  setStepper("pending");
  setStatus("pending", "Submitted. Waiting for the pipeline…", true);
  document.getElementById("status-card").hidden = false;

  pollRun(runId);
  pollHandle = setInterval(() => pollRun(runId), 4000);
}

async function pollRun(runId) {
  let response;
  try {
    response = await fetch(`/api/runs/${runId}`);
  } catch (err) {
    setStatus("failed", `Network error: ${err.message}`, false);
    clearInterval(pollHandle);
    return;
  }
  if (!response.ok) {
    setStatus("failed", `Could not fetch run status: ${response.status}`, false);
    clearInterval(pollHandle);
    return;
  }
  const run = await response.json();

  if (run.status === "pending" || run.status === "running") {
    setStepper(run.status);
    setStatus(run.status, `Status: ${run.status}… (real CPU inference can take several minutes)`, true);
    return;
  }
  clearInterval(pollHandle);

  if (run.status === "failed") {
    setStepper("failed");
    setStatus("failed", `Run failed: ${run.error}`, false);
    return;
  }

  setStepper("done");
  setStatus(run.verdict || "done", `Done — verdict: ${run.verdict}`, false);
  renderReport(run);
  renderClaims(run);
}

function setStepper(status) {
  const order = ["pending", "running", "done"];
  const failed = status === "failed";
  const activeIndex = failed ? order.length : order.indexOf(status);

  document.querySelectorAll("#stepper .step").forEach((el, i) => {
    el.classList.remove("active", "complete", "failed");
    if (failed && i === order.length - 1) {
      el.classList.add("failed");
    } else if (i < activeIndex) {
      el.classList.add("complete");
    } else if (i === activeIndex) {
      el.classList.add("active");
    }
  });
}

function setStatus(badgeClass, text, showSpinner) {
  const badge = document.getElementById("status-badge");
  badge.className = `badge ${badgeClass}`;
  badge.textContent = badgeClass;
  document.getElementById("status-text").textContent = text;
  document.getElementById("status-spinner").hidden = !showSpinner;
}

function renderReport(run) {
  const card = document.getElementById("report-card");
  if (!run.report_html) {
    card.hidden = true;
    return;
  }
  card.hidden = false;
  document.getElementById("report-frame").srcdoc = run.report_html;
}

function confidenceClass(confidence) {
  if (confidence >= 0.7) return "";
  if (confidence >= 0.4) return "mid";
  return "low";
}

function renderClaims(run) {
  const card = document.getElementById("claims-card");
  const grid = document.getElementById("claims-grid");
  grid.innerHTML = "";

  if (!run.claims || run.claims.length === 0) {
    card.hidden = true;
    return;
  }
  card.hidden = false;

  for (const claim of run.claims) {
    grid.appendChild(buildClaimCard(run, claim));
  }
}

function buildClaimCard(run, claim) {
  const box = document.createElement("div");
  box.className = "claim-card";

  const pct = Math.round((claim.confidence ?? 0) * 100);
  const provisionalTag = claim.provisional ? '<span class="tag">provisional</span>' : "";

  box.innerHTML = `
    <div class="claim-head">
      <strong>${escapeHtml(claim.cell_id)}</strong>${provisionalTag}
    </div>
    <div class="claim-field"><b>cell type:</b> ${escapeHtml(claim.cell_type ?? "unclassified")}</div>
    <div class="claim-field"><b>spatial domain:</b> ${escapeHtml(claim.spatial_domain ?? "—")}</div>
    <div class="claim-field"><b>confidence:</b> ${pct}%</div>
    <div class="confidence-bar ${confidenceClass(claim.confidence ?? 0)}"><span style="width:${pct}%"></span></div>
    <button type="button" class="correction-toggle">Suggest a correction</button>
    <div class="correction-form">
      <select class="field-select">
        <option value="cell_type">cell_type</option>
        <option value="spatial_domain">spatial_domain</option>
        <option value="interpretation">interpretation</option>
      </select>
      <input type="text" class="reviewer-input" placeholder="your name" value="${escapeHtml(getReviewerName())}">
      <input type="text" class="corrected-input" placeholder="corrected value" required>
      <textarea class="reason-input" placeholder="reason (optional)" rows="2"></textarea>
      <button type="button" class="ghost small submit-correction">Submit correction</button>
      <span class="correction-result" aria-live="polite"></span>
    </div>
  `;

  box.querySelector(".correction-toggle").addEventListener("click", () => {
    box.querySelector(".correction-form").classList.toggle("open");
  });
  box.querySelector(".submit-correction").addEventListener("click", () => submitCorrection(run, claim, box));
  return box;
}

async function submitCorrection(run, claim, box) {
  const resultEl = box.querySelector(".correction-result");
  const reviewer = box.querySelector(".reviewer-input").value.trim() || "anonymous";
  const correctedValue = box.querySelector(".corrected-input").value.trim();
  const field = box.querySelector(".field-select").value;
  const reason = box.querySelector(".reason-input").value.trim();

  if (!correctedValue) {
    resultEl.className = "correction-result err";
    resultEl.textContent = "corrected value is required";
    return;
  }
  setReviewerName(reviewer);

  const originalValue = field === "cell_type" ? claim.cell_type ?? null
    : field === "spatial_domain" ? claim.spatial_domain ?? null
    : null;

  const body = {
    run_id: run.run_id,
    task_id: run.task_id,
    claim_id: claim.cell_id,
    field,
    reviewer,
    original_value: originalValue,
    corrected_value: correctedValue,
    reason: reason || null,
  };

  resultEl.className = "correction-result";
  resultEl.textContent = "saving…";
  try {
    const response = await fetch("/review/corrections", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) throw new Error(`${response.status}`);
    resultEl.className = "correction-result ok";
    resultEl.textContent = "saved";
  } catch (err) {
    resultEl.className = "correction-result err";
    resultEl.textContent = `error: ${err.message}`;
  }
}

/* ---------------------------------------------------------------------
 * Recent runs
 * ------------------------------------------------------------------- */
function setActiveChip(runId) {
  document.querySelectorAll("#recent-runs-chips .chip").forEach((chip) => {
    chip.setAttribute("aria-current", String(chip.dataset.runId === runId));
  });
}

function renderRecentRuns() {
  const container = document.getElementById("recent-runs-chips");
  const runs = getRecentRuns();
  container.innerHTML = "";

  if (runs.length === 0) {
    container.innerHTML = '<span class="empty-hint">No runs yet on this browser.</span>';
    return;
  }

  for (const run of runs) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.dataset.runId = run.run_id;
    chip.setAttribute("aria-current", "false");
    chip.innerHTML = `<span class="dot"></span> ${escapeHtml(run.sample_id)} <span class="mono">${run.run_id.slice(0, 8)}</span>`;
    chip.addEventListener("click", () => trackRun(run.run_id));
    container.appendChild(chip);
  }
}

/* ---------------------------------------------------------------------
 * Reviewer Queue
 * ------------------------------------------------------------------- */
async function loadEscalations() {
  const list = document.getElementById("escalations-list");
  list.innerHTML = '<div class="loading-row">Loading…</div>';
  try {
    const response = await fetch("/review/escalations");
    if (!response.ok) throw new Error(`${response.status}`);
    const escalations = await response.json();
    renderEscalations(escalations);
  } catch (err) {
    list.innerHTML = `<div class="loading-row">Could not load escalations: ${escapeHtml(err.message)}</div>`;
  }
}

function renderEscalations(escalations) {
  const list = document.getElementById("escalations-list");
  list.innerHTML = "";
  if (escalations.length === 0) {
    list.innerHTML = '<span class="empty-hint">No pending escalations.</span>';
    return;
  }

  for (const esc of escalations) {
    const row = document.createElement("div");
    row.className = "escalation-row";
    const pct = Math.round((esc.confidence ?? 0) * 100);
    row.innerHTML = `
      <div class="row-head">
        <span><span class="badge escalate">escalate</span> <span class="mono">run ${esc.run_id.slice(0, 8)}</span> · claim ${escapeHtml(esc.claim_id ?? "—")} · confidence ${pct}%</span>
        <button type="button" class="ghost small resolve-btn">Resolve</button>
      </div>
      <div class="payload-preview">${escapeHtml(JSON.stringify(esc.payload, null, 2))}</div>
      <div class="mono">opened ${new Date(esc.created_at).toLocaleString()}</div>
    `;
    row.querySelector(".resolve-btn").addEventListener("click", async (event) => {
      event.target.disabled = true;
      event.target.textContent = "Resolving…";
      try {
        const response = await fetch(`/review/escalations/${esc.id}/resolve`, { method: "POST" });
        if (!response.ok) throw new Error(`${response.status}`);
        row.remove();
        if (document.getElementById("escalations-list").children.length === 0) {
          document.getElementById("escalations-list").innerHTML = '<span class="empty-hint">No pending escalations.</span>';
        }
      } catch (err) {
        event.target.disabled = false;
        event.target.textContent = `Error: ${err.message}`;
      }
    });
    list.appendChild(row);
  }
}

async function loadCorrections() {
  const body = document.getElementById("corrections-body");
  body.innerHTML = '<tr><td colspan="7" class="loading-row">Loading…</td></tr>';

  const runFilter = document.getElementById("corrections-run-filter").value.trim();
  const url = runFilter ? `/review/corrections?run_id=${encodeURIComponent(runFilter)}` : "/review/corrections";

  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${response.status}`);
    const corrections = await response.json();
    renderCorrections(corrections);
  } catch (err) {
    body.innerHTML = `<tr><td colspan="7" class="loading-row">Could not load corrections: ${escapeHtml(err.message)}</td></tr>`;
  }
}

function renderCorrections(corrections) {
  const body = document.getElementById("corrections-body");
  body.innerHTML = "";
  if (corrections.length === 0) {
    body.innerHTML = '<tr><td colspan="7" class="loading-row">No corrections recorded yet.</td></tr>';
    return;
  }
  // Most recent first.
  corrections.slice().reverse().forEach((c) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${new Date(c.created_at).toLocaleString()}</td>
      <td class="mono">${c.run_id.slice(0, 8)}</td>
      <td class="mono">${escapeHtml(c.claim_id)}</td>
      <td>${escapeHtml(c.field)}</td>
      <td>${escapeHtml(formatValue(c.original_value))} → ${escapeHtml(formatValue(c.corrected_value))}</td>
      <td>${escapeHtml(c.reviewer)}</td>
      <td>${escapeHtml(c.reason ?? "—")}</td>
    `;
    body.appendChild(row);
  });
}

function formatValue(value) {
  if (value === null || value === undefined) return "—";
  return typeof value === "string" ? value : JSON.stringify(value);
}

/* ---------------------------------------------------------------------
 * Theme -- three states: "light", "dark", or unset (follow the OS/browser
 * preference, handled purely in CSS via prefers-color-scheme). The toggle
 * cycles unset -> light -> dark -> unset and persists the explicit choice;
 * "unset" is intentionally not written to localStorage so a viewer who
 * never touches the toggle keeps tracking their OS setting.
 * ------------------------------------------------------------------- */
const THEME_KEY = "nexus.theme";

function applyTheme(theme) {
  if (theme === "light" || theme === "dark") {
    document.documentElement.dataset.theme = theme;
  } else {
    delete document.documentElement.dataset.theme;
  }
  const icon = document.getElementById("theme-toggle-icon");
  const label = document.getElementById("theme-toggle-label");
  if (theme === "light") { icon.textContent = "☀️"; label.textContent = "Light"; }
  else if (theme === "dark") { icon.textContent = "🌙"; label.textContent = "Dark"; }
  else { icon.textContent = "🌓"; label.textContent = "Auto"; }
}

function initTheme() {
  applyTheme(localStorage.getItem(THEME_KEY));
  document.getElementById("theme-toggle").addEventListener("click", () => {
    const current = localStorage.getItem(THEME_KEY);
    const next = current === null ? "light" : current === "light" ? "dark" : null;
    if (next === null) {
      localStorage.removeItem(THEME_KEY);
    } else {
      localStorage.setItem(THEME_KEY, next);
    }
    applyTheme(next);
  });
}

/* ---------------------------------------------------------------------
 * Utilities
 * ------------------------------------------------------------------- */
function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[ch]));
}

/* ---------------------------------------------------------------------
 * Init
 * ------------------------------------------------------------------- */
document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initTabs();
  initDropzone("image-drop", "image");
  initDropzone("expression-drop", "expression");
  initUploadForm();
  renderRecentRuns();

  document.getElementById("refresh-escalations").addEventListener("click", loadEscalations);
  document.getElementById("refresh-corrections").addEventListener("click", loadCorrections);
  document.getElementById("corrections-filter-form").addEventListener("submit", (event) => {
    event.preventDefault();
    loadCorrections();
  });
});
