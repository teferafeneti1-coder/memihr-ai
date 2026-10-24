/**
 * ui.js – DOM helpers and reusable UI components.
 * No framework, no innerHTML for user data (uses textContent / createElement).
 */

// ── Toast ─────────────────────────────────────────────────────────────────────
function showToast(message, kind = "ok", duration = 3500) {
  const container = document.getElementById("toast-container");
  const el = document.createElement("div");
  el.className = `toast toast-${kind}`;
  el.textContent = message;
  container.appendChild(el);
  requestAnimationFrame(() => {
    requestAnimationFrame(() => el.classList.add("show"));
  });
  setTimeout(() => {
    el.classList.remove("show");
    setTimeout(() => el.remove(), 250);
  }, duration);
}

// ── Status helpers ────────────────────────────────────────────────────────────
function statusDotClass(status) {
  return `dot-${status || "stopped"}`;
}

function statusLabel(status) {
  return status || "stopped";
}

// ── Build a sidebar card element ──────────────────────────────────────────────
function buildSidebarCard(proj) {
  const card = document.createElement("div");
  card.className = "sidebar-card";
  card.dataset.pid = proj.id;

  const nameRow = document.createElement("div");
  nameRow.className = "card-name";
  nameRow.textContent = proj.name;

  if (proj.needs_confirmation) {
    const badge = document.createElement("span");
    badge.className = "needs-confirm-badge";
    badge.textContent = "confirm";
    nameRow.appendChild(badge);
  }

  const statusRow = document.createElement("div");
  statusRow.className = "card-status";

  const dot = document.createElement("span");
  dot.className = `status-dot ${statusDotClass(proj.status)}`;

  const statusTxt = document.createElement("span");
  statusTxt.className = "card-status-text";
  statusTxt.textContent = statusLabel(proj.status);

  statusRow.appendChild(dot);
  statusRow.appendChild(statusTxt);

  const logLine = document.createElement("div");
  logLine.className = "card-log";
  logLine.textContent = proj.last_log || "";

  const statsLine = document.createElement("div");
  statsLine.className = "card-stats";
  if (proj.status === "running") {
    statsLine.textContent =
      `CPU ${proj.cpu_percent != null ? proj.cpu_percent.toFixed(1) : "—"}%  ` +
      `RAM ${proj.memory_mb   != null ? (proj.memory_mb / 1024).toFixed(2) : "—"} GB`;
  }

  card.appendChild(nameRow);
  card.appendChild(statusRow);
  card.appendChild(logLine);
  card.appendChild(statsLine);

  return card;
}

// ── Update an existing sidebar card in-place ──────────────────────────────────
function updateSidebarCard(card, proj) {
  const dot       = card.querySelector(".status-dot");
  const statusTxt = card.querySelector(".card-status-text");
  const logLine   = card.querySelector(".card-log");
  const statsLine = card.querySelector(".card-stats");

  if (dot)       dot.className = `status-dot ${statusDotClass(proj.status)}`;
  if (statusTxt) statusTxt.textContent = statusLabel(proj.status);
  if (logLine)   logLine.textContent   = proj.last_log || "";
  if (statsLine) {
    if (proj.status === "running") {
      statsLine.textContent =
        `CPU ${proj.cpu_percent != null ? proj.cpu_percent.toFixed(1) : "—"}%  ` +
        `RAM ${proj.memory_mb   != null ? (proj.memory_mb / 1024).toFixed(2) : "—"} GB`;
    } else {
      statsLine.textContent = "";
    }
  }
}

// ── Build a home-grid card element ────────────────────────────────────────────
function buildHomeCard(proj) {
  const card = document.createElement("div");
  card.className = "home-card-big";
  card.dataset.pid = proj.id;

  const nameEl = document.createElement("div");
  nameEl.className = "card-name";

  const dot = document.createElement("span");
  dot.className = `status-dot ${statusDotClass(proj.status)}`;
  dot.style.display = "inline-block";
  dot.style.marginRight = "8px";

  nameEl.appendChild(dot);
  const nameText = document.createElement("span");
  nameText.textContent = proj.name;
  nameEl.appendChild(nameText);

  const descEl = document.createElement("p");
  descEl.className = "card-desc";
  descEl.textContent = proj.description || "";

  const portEl = document.createElement("div");
  portEl.className = "card-port";
  portEl.textContent = `Port ${proj.port}  ·  ${statusLabel(proj.status)}`;

  // Skip reason banner
  const skipEl = document.createElement("div");
  skipEl.className = "skip-msg";
  if (proj.status === "skipped" && proj.skip_reason) {
    skipEl.textContent = proj.skip_reason;
  }

  card.appendChild(nameEl);
  card.appendChild(descEl);
  card.appendChild(portEl);
  if (proj.status === "skipped" && proj.skip_reason) card.appendChild(skipEl);

  return card;
}

// ── Update an existing home-grid card ────────────────────────────────────────
function updateHomeCard(card, proj) {
  const dot  = card.querySelector(".status-dot");
  const port = card.querySelector(".card-port");
  const skip = card.querySelector(".skip-msg");

  if (dot)  dot.className = `status-dot ${statusDotClass(proj.status)}`;
  if (port) port.textContent = `Port ${proj.port}  ·  ${statusLabel(proj.status)}`;
  if (skip && proj.status === "skipped" && proj.skip_reason) {
    skip.textContent = proj.skip_reason;
    skip.hidden = false;
  } else if (skip) {
    skip.hidden = true;
  }
}

// ── Append a log line to the logs panel ──────────────────────────────────────
function appendLogLine(container, text, autoScroll) {
  const line = document.createElement("span");
  line.className = "log-line";

  const lower = text.toLowerCase();
  if (lower.includes("error") || lower.includes("traceback") || lower.includes("exception")) {
    line.classList.add("log-error");
  } else if (lower.includes("warn")) {
    line.classList.add("log-warn");
  }

  line.textContent = text;
  container.appendChild(document.createTextNode("\n"));
  container.appendChild(line);

  if (autoScroll) {
    container.scrollTop = container.scrollHeight;
  }
}

// ── Memory bar ────────────────────────────────────────────────────────────────
function updateMemoryBar(mem) {
  const fill  = document.getElementById("memory-bar-fill");
  const label = document.getElementById("memory-text");
  if (!fill || !label) return;

  const pct = mem.percent || 0;
  fill.style.width = `${pct}%`;

  if      (pct > 90) fill.style.background = "var(--red)";
  else if (pct > 70) fill.style.background = "var(--yellow)";
  else               fill.style.background = "var(--green)";

  label.textContent = `${mem.available_gb.toFixed(1)} GB free / ${mem.total_gb.toFixed(1)} GB`;
}
