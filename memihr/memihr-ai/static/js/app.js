/**
 * app.js – main application logic for the egate dashboard.
 *
 * Architecture:
 *  - Polls /api/projects every 3 s for status updates.
 *  - Polls /api/memory every 5 s.
 *  - Polls /api/wake-progress every 2 s while wake-all is active.
 *  - Maintains one <iframe> per project (created once, shown/hidden).
 *  - Projects with unload_on_hide=true have their iframe src reset to
 *    about:blank when hidden and reloaded when selected.
 *  - All user-facing text is set via textContent (never innerHTML).
 */

(function () {
  "use strict";

  // ── State ──────────────────────────────────────────────────────────────────
  let projects      = [];          // latest list from API
  let selectedPid   = null;        // currently viewed project id (null = home)
  let activeTab     = "app";       // "app" | "logs" | "edit"
  let logEventSrc   = null;        // active EventSource for log streaming
  let autoScroll    = true;
  let iframes       = {};          // pid → <iframe> element

  // ── DOM refs ───────────────────────────────────────────────────────────────
  const sidebarProjects    = document.getElementById("sidebar-projects");
  const sidebarHome        = document.getElementById("sidebar-home");
  const homeView           = document.getElementById("home-view");
  const homeGrid           = document.getElementById("home-grid");
  const projectView        = document.getElementById("project-view");
  const projectTitle       = document.getElementById("project-title");
  const projectDesc        = document.getElementById("project-desc");
  const wakeProgressText   = document.getElementById("wake-progress-text");

  const btnWakeAll         = document.getElementById("btn-wake-all");
  const btnSleepAll        = document.getElementById("btn-sleep-all");
  const btnStart           = document.getElementById("btn-start");
  const btnStop            = document.getElementById("btn-stop");
  const btnRestart         = document.getElementById("btn-restart");
  const btnOpenTab         = document.getElementById("btn-open-tab");
  const btnOpenTab2        = document.getElementById("btn-open-tab-2");
  const btnReloadIframe    = document.getElementById("btn-reload-iframe");
  const btnStartFromApp    = document.getElementById("btn-start-from-app");
  const appNotRunning      = document.getElementById("app-not-running");
  const appNotRunningMsg   = document.getElementById("app-not-running-msg");
  const iframeContainer    = document.getElementById("iframe-container");
  const logsOutput         = document.getElementById("logs-output");
  const btnClearLogs       = document.getElementById("btn-clear-logs");
  const btnCopyLogs        = document.getElementById("btn-copy-logs");
  const editForm           = document.getElementById("edit-form");
  const editError          = document.getElementById("edit-error");

  // ── Helpers ────────────────────────────────────────────────────────────────

  function getProject(pid) {
    return projects.find(p => p.id === pid) || null;
  }

  /** Build the URL for a project's iframe using the current window hostname */
  function projectUrl(proj) {
    const host = window.location.hostname;
    const path = proj.url_path || "/";
    return `http://${host}:${proj.port}${path}`;
  }

  // ── Iframe management ──────────────────────────────────────────────────────

  function getOrCreateIframe(proj) {
    if (iframes[proj.id]) return iframes[proj.id];

    const iframe = document.createElement("iframe");
    iframe.className = "project-iframe";
    iframe.setAttribute("allow", "camera; microphone; fullscreen; clipboard-read; clipboard-write");
    iframe.title = proj.name;
    iframeContainer.appendChild(iframe);
    iframes[proj.id] = iframe;
    return iframe;
  }

  function hideAllIframes() {
    Object.entries(iframes).forEach(([pid, iframe]) => {
      const proj = getProject(pid);
      if (proj && proj.unload_on_hide) {
        // Release camera/CPU
        iframe.src = "about:blank";
        iframe.setAttribute("data-loaded-src", "about:blank");
      }
      iframe.hidden = true;
    });
  }

  function showIframe(proj) {
    hideAllIframes();
    const iframe = getOrCreateIframe(proj);

    const isLive = proj.status === "running";

    if (isLive) {
      // If it was unloaded or not yet loaded, set src
      const expectedSrc = projectUrl(proj);
      const currentSrc  = iframe.getAttribute("data-loaded-src") || "";
      if (!currentSrc || currentSrc === "about:blank") {
        iframe.src = expectedSrc;
        iframe.setAttribute("data-loaded-src", expectedSrc);
      }
      iframe.hidden = false;
      appNotRunning.hidden = true;
    } else {
      iframe.hidden = true;
      appNotRunning.hidden = false;
      const msgs = {
        stopped:  "This project is stopped. Start it to load the app.",
        starting: "Project is starting up, please wait…",
        crashed:  "Project crashed. Check the Logs tab for details.",
        skipped:  "Project was skipped (low memory). Use 'Start anyway' below.",
      };
      appNotRunningMsg.textContent = msgs[proj.status] || "Project is not running.";
    }
  }

  // When a running project's status changes to running while we're watching it,
  // update the iframe automatically.
  function refreshIframeForSelected() {
    if (!selectedPid) return;
    const proj = getProject(selectedPid);
    if (!proj) return;
    if (activeTab === "app") {
      showIframe(proj);
    }
  }

  // ── Sidebar ────────────────────────────────────────────────────────────────

  function renderSidebar() {
    // Build cards if they don't exist yet
    projects.forEach(proj => {
      let card = sidebarProjects.querySelector(`[data-pid="${CSS.escape(proj.id)}"]`);
      if (!card) {
        card = buildSidebarCard(proj);
        card.addEventListener("click", () => selectProject(proj.id));
        sidebarProjects.appendChild(card);
      } else {
        updateSidebarCard(card, proj);
      }
      card.classList.toggle("active", proj.id === selectedPid);
    });
    sidebarHome.classList.toggle("active", selectedPid === null);
  }

  // ── Home grid ──────────────────────────────────────────────────────────────

  function renderHomeGrid() {
    // Build big cards if not present
    projects.forEach(proj => {
      let card = homeGrid.querySelector(`[data-pid="${CSS.escape(proj.id)}"]`);
      if (!card) {
        card = buildHomeCard(proj);
        card.addEventListener("click", () => selectProject(proj.id));
        homeGrid.appendChild(card);
      } else {
        updateHomeCard(card, proj);
      }
    });
  }

  // ── Project view ───────────────────────────────────────────────────────────

  function selectProject(pid) {
    if (pid === null || pid === "__home__") {
      selectedPid = null;
      homeView.hidden    = false;
      projectView.hidden = true;
      sidebarHome.classList.add("active");
      sidebarProjects.querySelectorAll(".sidebar-card").forEach(c => c.classList.remove("active"));
      stopLogStream();
      hideAllIframes();
      return;
    }

    const proj = getProject(pid);
    if (!proj) return;

    selectedPid = pid;
    homeView.hidden    = true;
    projectView.hidden = false;

    sidebarHome.classList.remove("active");
    sidebarProjects.querySelectorAll(".sidebar-card").forEach(c => {
      c.classList.toggle("active", c.dataset.pid === pid);
    });

    renderProjectHeader(proj);
    switchTab(activeTab);   // re-render current tab with new project
  }

  function renderProjectHeader(proj) {
    projectTitle.textContent = proj.name;
    projectDesc.textContent  = proj.description || "";

    // Skipped banner (show in header area if skipped)
    const existing = projectView.querySelector(".skipped-banner");
    if (existing) existing.remove();

    if (proj.status === "skipped" && proj.skip_reason) {
      const banner = document.createElement("div");
      banner.className = "skipped-banner";

      const msg = document.createElement("span");
      msg.textContent = proj.skip_reason;

      const btn = document.createElement("button");
      btn.className = "btn btn-orange";
      btn.textContent = "Start anyway";
      btn.addEventListener("click", () => doStart(proj.id));

      banner.appendChild(msg);
      banner.appendChild(btn);
      // Insert after header
      projectView.insertBefore(banner, document.getElementById("tabs"));
    }
  }

  // ── Tab switching ──────────────────────────────────────────────────────────

  document.getElementById("tabs").addEventListener("click", e => {
    const btn = e.target.closest(".tab");
    if (!btn) return;
    switchTab(btn.dataset.tab);
  });

  function switchTab(tab) {
    activeTab = tab;

    document.querySelectorAll(".tab").forEach(b => b.classList.toggle("active", b.dataset.tab === tab));
    document.getElementById("tab-app").hidden  = (tab !== "app");
    document.getElementById("tab-logs").hidden = (tab !== "logs");
    document.getElementById("tab-edit").hidden = (tab !== "edit");

    if (!selectedPid) return;
    const proj = getProject(selectedPid);
    if (!proj) return;

    if (tab === "app") {
      showIframe(proj);
    } else {
      hideAllIframes();
    }

    if (tab === "logs") {
      startLogStream(selectedPid);
    } else {
      stopLogStream();
    }

    if (tab === "edit") {
      populateEditForm(proj);
    }
  }

  // ── Log streaming ──────────────────────────────────────────────────────────

  function startLogStream(pid) {
    stopLogStream();
    logsOutput.textContent = "";

    logEventSrc = API.streamLogs(pid, (line) => {
      appendLogLine(logsOutput, line, autoScroll);
    });
  }

  function stopLogStream() {
    if (logEventSrc) {
      logEventSrc.close();
      logEventSrc = null;
    }
  }

  btnClearLogs.addEventListener("click", () => {
    logsOutput.textContent = "";
  });

  btnCopyLogs.addEventListener("click", async () => {
    const text = logsOutput.textContent;
    try {
      await navigator.clipboard.writeText(text);
      showToast("Logs copied to clipboard", "ok");
    } catch (_) {
      showToast("Could not copy (clipboard permission denied)", "warn");
    }
  });

  // Auto-scroll: if user scrolls up, disable; if they scroll to bottom, re-enable
  logsOutput.addEventListener("scroll", () => {
    const atBottom = logsOutput.scrollHeight - logsOutput.scrollTop - logsOutput.clientHeight < 40;
    autoScroll = atBottom;
  });

  // ── Edit form ──────────────────────────────────────────────────────────────

  function populateEditForm(proj) {
    const f = editForm;
    f.name.value        = proj.name        || "";
    f.description.value = proj.description || "";
    f.port.value        = proj.port        || "";
    f.entry.value       = proj.entry       || "";
    f.url_path.value    = proj.url_path    || "/";
    f.ram_gb.value      = proj.ram_gb      || "";
    f.autostart.checked    = !!proj.autostart;
    f.heavy_start.checked  = !!proj.heavy_start;
    f.unload_on_hide.checked = !!proj.unload_on_hide;
    editError.hidden = true;
  }

  editForm.addEventListener("submit", async e => {
    e.preventDefault();
    if (!selectedPid) return;
    const f = editForm;
    const data = {
      name:            f.name.value.trim(),
      description:     f.description.value.trim(),
      port:            parseInt(f.port.value, 10),
      entry:           f.entry.value.trim(),
      url_path:        f.url_path.value.trim() || "/",
      ram_gb:          parseFloat(f.ram_gb.value) || 0,
      autostart:       f.autostart.checked,
      heavy_start:     f.heavy_start.checked,
      unload_on_hide:  f.unload_on_hide.checked,
    };

    editError.hidden = true;
    try {
      await API.updateProject(selectedPid, data);
      showToast("Project settings saved", "ok");
      await refreshProjects();
    } catch (err) {
      editError.textContent = err.message;
      editError.hidden = false;
      showToast("Failed to save: " + err.message, "error");
    }
  });

  // ── Project actions ────────────────────────────────────────────────────────

  async function doStart(pid) {
    const proj = getProject(pid);
    if (!proj) return;
    const name = proj.name;
    showToast(`Starting ${name}…`, "ok");
    try {
      const res = await API.startProject(pid);
      if (res.ok) {
        showToast(`${name} started`, "ok");
      } else {
        showToast(`${name}: ${res.message}`, "error");
      }
    } catch (err) {
      showToast(`Failed to start ${name}: ${err.message}`, "error");
    }
    await refreshProjects();
    if (selectedPid === pid) refreshIframeForSelected();
  }

  async function doStop(pid) {
    const proj = getProject(pid);
    const name = proj ? proj.name : pid;
    showToast(`Stopping ${name}…`, "warn");
    try {
      await API.stopProject(pid);
      showToast(`${name} stopped`, "ok");
    } catch (err) {
      showToast(`Failed to stop ${name}: ${err.message}`, "error");
    }
    await refreshProjects();
  }

  async function doRestart(pid) {
    const proj = getProject(pid);
    const name = proj ? proj.name : pid;
    showToast(`Restarting ${name}…`, "warn");
    try {
      const res = await API.restartProject(pid);
      if (res.ok) {
        showToast(`${name} restarted`, "ok");
      } else {
        showToast(`${name}: ${res.message}`, "error");
      }
    } catch (err) {
      showToast(`Failed to restart ${name}: ${err.message}`, "error");
    }
    await refreshProjects();
    if (selectedPid === pid) refreshIframeForSelected();
  }

  btnStart.addEventListener("click", () => selectedPid && doStart(selectedPid));
  btnStop.addEventListener("click",  () => selectedPid && doStop(selectedPid));
  btnRestart.addEventListener("click", () => selectedPid && doRestart(selectedPid));
  btnStartFromApp.addEventListener("click", () => selectedPid && doStart(selectedPid));

  btnWakeAll.addEventListener("click", async () => {
    try {
      await API.wakeAll();
      showToast("Waking all projects…", "ok");
      startWakeProgressPolling();
    } catch (err) {
      showToast("Wake-all failed: " + err.message, "error");
    }
  });

  btnSleepAll.addEventListener("click", async () => {
    try {
      await API.sleepAll();
      showToast("Stopping all projects…", "warn");
    } catch (err) {
      showToast("Sleep-all failed: " + err.message, "error");
    }
  });

  // ── Open in new tab / reload iframe ───────────────────────────────────────

  function openInNewTab() {
    if (!selectedPid) return;
    const proj = getProject(selectedPid);
    if (!proj) return;
    window.open(projectUrl(proj), "_blank", "noopener,noreferrer");
  }

  btnOpenTab.addEventListener("click",  openInNewTab);
  btnOpenTab2.addEventListener("click", openInNewTab);

  btnReloadIframe.addEventListener("click", () => {
    if (!selectedPid) return;
    const iframe = iframes[selectedPid];
    if (iframe && !iframe.hidden) {
      iframe.src = iframe.src;  // force reload
    }
  });

  // ── Polling ────────────────────────────────────────────────────────────────

  async function refreshProjects() {
    try {
      const updated = await API.getProjects();
      const prevStatuses = {};
      projects.forEach(p => { prevStatuses[p.id] = p.status; });

      projects = updated;

      renderSidebar();
      if (!selectedPid) {
        renderHomeGrid();
      }

      // If selected project changed status, update header and iframe
      if (selectedPid) {
        const proj = getProject(selectedPid);
        if (proj) {
          renderProjectHeader(proj);
          const prev = prevStatuses[selectedPid];
          if (prev !== proj.status && activeTab === "app") {
            showIframe(proj);
          }
        }
      }
    } catch (err) {
      // Silently ignore transient fetch errors
      console.warn("refreshProjects error:", err);
    }
  }

  async function refreshMemory() {
    try {
      const mem = await API.getMemory();
      updateMemoryBar(mem);
    } catch (_) {}
  }

  let wakeProgressInterval = null;

  function startWakeProgressPolling() {
    if (wakeProgressInterval) return;
    wakeProgressInterval = setInterval(async () => {
      try {
        const data = await API.getWakeProgress();
        wakeProgressText.textContent = data.progress || "";
        if (!data.progress || data.progress.startsWith("Done")) {
          stopWakeProgressPolling();
          wakeProgressText.textContent = data.progress || "";
        }
      } catch (_) {}
    }, 2000);
  }

  function stopWakeProgressPolling() {
    if (wakeProgressInterval) {
      clearInterval(wakeProgressInterval);
      wakeProgressInterval = null;
    }
  }

  // ── Sidebar home click ─────────────────────────────────────────────────────
  sidebarHome.addEventListener("click", () => selectProject(null));

  // ── Boot ───────────────────────────────────────────────────────────────────

  async function init() {
    await refreshProjects();
    await refreshMemory();

    // Check if wake-all is already in progress (dashboard just started)
    try {
      const data = await API.getWakeProgress();
      if (data.progress && !data.progress.startsWith("Done")) {
        wakeProgressText.textContent = data.progress;
        startWakeProgressPolling();
      }
    } catch (_) {}

    // Periodic refresh
    setInterval(refreshProjects, 3000);
    setInterval(refreshMemory,   5000);

    // Start at home view
    selectProject(null);
  }

  init();

})();
