"use strict";

/* ---------------------------------------------------------------------
 * Small persisted bits of client state: which runs this browser has
 * submitted (the API has no "list runs" endpoint, so this is the only
 * way to come back to a past run without knowing its UUID by heart),
 * each with its last-known status so a reload can show something
 * immediately instead of a blank list, and the reviewer's display name
 * (so they don't retype it per claim).
 * ------------------------------------------------------------------- */
const RECENT_RUNS_KEY = "nexus.recentRuns";
const REVIEWER_NAME_KEY = "nexus.reviewerName";
const MAX_RECENT_RUNS = 30;
const ACTIVE_STATUSES = ["pending", "running"];

function getRecentRuns() {
  try {
    return JSON.parse(localStorage.getItem(RECENT_RUNS_KEY)) || [];
  } catch (err) {
    return [];
  }
}

// `status` is the exact badge class to show: the raw "pending"/"running"/
// "failed" while in flight, or the verdict ("pass"/"veto"/"escalate")
// once done -- storing it pre-resolved this way means the cached copy
// renders with `updateRunEntryStatus` exactly like a fresh poll response.
function upsertRunRecord(runId, sampleId, status) {
  const runs = getRecentRuns().filter((r) => r.run_id !== runId);
  runs.unshift({ run_id: runId, sample_id: sampleId, status });
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
    if (input.files.length === 0) {
      drop.textContent = defaultText;
      drop.classList.remove("has-file");
      return;
    }
    const names = Array.from(input.files).map((f) => f.name);
    drop.textContent = names.length === 1 ? names[0] : `${names.length} files: ${names.join(", ")}`;
    drop.classList.add("has-file");
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
 * Submit & Track -- each tracked run gets its own DOM card and its own
 * polling interval (keyed in `pollHandles`), so submitting a new job
 * never disturbs whatever else is currently running or already done.
 * The upload form itself is never hidden: nothing stops another submit.
 * ------------------------------------------------------------------- */
const pollHandles = new Map();

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
    const imageInput = document.getElementById("image");
    const expressionInput = document.getElementById("expression");
    const formData = new FormData();
    formData.append("sample_id", sampleId);
    formData.append("image", imageInput.files[0]);
    for (const file of expressionInput.files) {
      formData.append("expression", file);
    }

    try {
      const response = await fetch("/api/runs", { method: "POST", body: formData });
      if (!response.ok) {
        throw new Error(`${response.status} ${await response.text()}`);
      }
      const { run_id } = await response.json();
      upsertRunRecord(run_id, sampleId, "pending");
      trackRun(run_id, sampleId, { prepend: true, autoExpand: true });

      // Clear the file pickers so a second submit right away doesn't
      // accidentally resend the same files for a different sample.
      imageInput.value = "";
      expressionInput.value = "";
      imageInput.dispatchEvent(new Event("change"));
      expressionInput.dispatchEvent(new Event("change"));
    } catch (err) {
      errorBanner.textContent = `Upload failed: ${err.message}`;
      errorBanner.hidden = false;
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = "Run pipeline";
    }
  });

  document.getElementById("load-run-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const input = document.getElementById("load-run-id");
    const runId = input.value.trim();
    if (runId) trackRun(runId, null, { prepend: true, autoExpand: true });
    input.value = "";
  });

  document.getElementById("refresh-runs-btn").addEventListener("click", () => {
    for (const run of getRecentRuns()) pollRunOnce(run.run_id);
  });
}

/* Render every run this browser knows about from its cached last-known
 * status (instant, no network) on page load. Still-active runs start
 * polling immediately, which corrects any stale cached status right
 * away; finished ones stay as-is until refreshed or expanded, so a
 * reload with a long history doesn't fire a burst of needless requests. */
function hydrateRunsList() {
  for (const run of getRecentRuns()) {
    const entry = ensureRunEntry(run.run_id, run.sample_id, { prepend: false });
    const status = run.status || "pending";
    updateRunEntryStatus(entry, status, statusMessage(status), ACTIVE_STATUSES.includes(status));
    if (ACTIVE_STATUSES.includes(status)) startPolling(run.run_id);
  }
}

function statusMessage(status) {
  if (ACTIVE_STATUSES.includes(status)) {
    return `Status: ${status}… (real CPU inference can take several minutes)`;
  }
  if (status === "failed") return "Run failed — expand for details.";
  return `Done — verdict: ${status}`;
}

function trackRun(runId, sampleIdHint, { prepend = true, autoExpand = false } = {}) {
  const entry = ensureRunEntry(runId, sampleIdHint, { prepend });
  if (autoExpand) entry.querySelector(".run-entry-details").classList.add("open");
  startPolling(runId);
}

function startPolling(runId) {
  if (pollHandles.has(runId)) return;
  pollHandles.set(runId, null); // reserve immediately so a second call can't race in
  const tick = async () => {
    const run = await pollRunOnce(runId);
    if (!run || !ACTIVE_STATUSES.includes(run.status)) stopPolling(runId);
  };
  tick();
  pollHandles.set(runId, setInterval(tick, 4000));
}

function stopPolling(runId) {
  const handle = pollHandles.get(runId);
  if (handle) clearInterval(handle);
  pollHandles.delete(runId);
}

async function pollRunOnce(runId) {
  const entry = ensureRunEntry(runId);
  let response;
  try {
    response = await fetch(`/api/runs/${runId}`);
  } catch (err) {
    updateRunEntryStatus(entry, "failed", `Network error: ${err.message}`, false);
    return null;
  }
  if (!response.ok) {
    const message = response.status === 404 ? "Run not found" : `Could not fetch run status: ${response.status}`;
    updateRunEntryStatus(entry, "failed", message, false);
    return null;
  }
  const run = await response.json();
  entry.querySelector(".run-entry-sample").textContent = run.sample_id;

  if (ACTIVE_STATUSES.includes(run.status)) {
    upsertRunRecord(run.run_id, run.sample_id, run.status);
    updateRunEntryStatus(entry, run.status, statusMessage(run.status), true);
    return run;
  }

  const badgeClass = run.status === "failed" ? "failed" : run.verdict || "done";
  upsertRunRecord(run.run_id, run.sample_id, badgeClass);
  updateRunEntryStatus(
    entry,
    badgeClass,
    run.status === "failed" ? `Run failed: ${run.error}` : `Done — verdict: ${run.verdict}`,
    false
  );
  if (run.status !== "failed") {
    renderReport(entry, run);
    renderClaims(entry, run);
  }
  return run;
}

function ensureRunEntry(runId, sampleIdHint, { prepend = true } = {}) {
  const existing = document.querySelector(`.run-entry[data-run-id="${CSS.escape(runId)}"]`);
  if (existing) return existing;

  document.getElementById("runs-empty-hint")?.remove();

  const entry = document.createElement("div");
  entry.className = "run-entry";
  entry.dataset.runId = runId;
  entry.innerHTML = `
    <div class="run-entry-head">
      <div class="run-entry-title">
        <strong class="run-entry-sample">${escapeHtml(sampleIdHint || "(loading…)")}</strong>
        <span class="mono">${escapeHtml(runId.slice(0, 8))}</span>
      </div>
      <div class="run-entry-actions">
        <span class="badge" data-role="badge"></span>
        <span class="spinner" data-role="spinner" hidden aria-hidden="true"></span>
        <button type="button" class="ghost small toggle-details">Details</button>
      </div>
    </div>
    <div class="run-entry-details">
      <div class="stepper">
        <div class="step" data-step="pending">Submitted</div>
        <div class="step" data-step="running">Running</div>
        <div class="step" data-step="done">Complete</div>
      </div>
      <div class="status-line" aria-live="polite" aria-atomic="true">
        <span data-role="status-text"></span>
      </div>
      <div class="report-wrap" hidden><iframe class="report-frame"></iframe></div>
      <div class="claims-wrap" hidden>
        <h3>Claims — submit a correction</h3>
        <p class="subtitle">Corrections are captured as structured training pairs for the next fine-tune cycle.</p>
        <div class="claims-grid"></div>
      </div>
    </div>
  `;

  const details = entry.querySelector(".run-entry-details");
  entry.querySelector(".toggle-details").addEventListener("click", async () => {
    details.classList.toggle("open");
    if (details.classList.contains("open") && !entry.dataset.loaded) {
      entry.dataset.loaded = "true";
      await pollRunOnce(runId);
    }
  });

  const list = document.getElementById("runs-list");
  if (prepend) list.prepend(entry); else list.appendChild(entry);
  return entry;
}

function setStepperFor(entry, status) {
  const order = ["pending", "running", "done"];
  const failed = status === "failed";
  const activeIndex = failed ? order.length : order.indexOf(ACTIVE_STATUSES.includes(status) ? status : "done");

  entry.querySelectorAll(".step").forEach((el, i) => {
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

function updateRunEntryStatus(entry, badgeClass, text, showSpinner) {
  const badge = entry.querySelector('[data-role="badge"]');
  badge.className = `badge ${badgeClass}`;
  badge.textContent = badgeClass;
  entry.querySelector('[data-role="status-text"]').textContent = text;
  entry.querySelector('[data-role="spinner"]').hidden = !showSpinner;
  setStepperFor(entry, badgeClass);
}

function renderReport(entry, run) {
  const wrap = entry.querySelector(".report-wrap");
  if (!run.report_html) {
    wrap.hidden = true;
    return;
  }
  wrap.hidden = false;
  wrap.querySelector("iframe").srcdoc = run.report_html;
}

function confidenceClass(confidence) {
  if (confidence >= 0.7) return "";
  if (confidence >= 0.4) return "mid";
  return "low";
}

function renderClaims(entry, run) {
  const wrap = entry.querySelector(".claims-wrap");
  const grid = entry.querySelector(".claims-grid");
  grid.innerHTML = "";

  if (!run.claims || run.claims.length === 0) {
    wrap.hidden = true;
    return;
  }
  wrap.hidden = false;

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
  hydrateRunsList();

  document.getElementById("refresh-escalations").addEventListener("click", loadEscalations);
  document.getElementById("refresh-corrections").addEventListener("click", loadCorrections);
  document.getElementById("corrections-filter-form").addEventListener("submit", (event) => {
    event.preventDefault();
    loadCorrections();
  });
});
