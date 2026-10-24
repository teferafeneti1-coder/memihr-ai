/**
 * api.js – thin wrappers around the dashboard REST API.
 * All functions return parsed JSON (or throw on HTTP error).
 */

const API = (() => {
  async function request(method, path, body) {
    const opts = { method, headers: {} };
    if (body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(path, opts);
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch (_) {}
      throw new Error(`${method} ${path} → ${res.status}: ${detail}`);
    }
    // 204 No Content
    if (res.status === 204) return null;
    return res.json();
  }

  return {
    getProjects:    ()         => request("GET",  "/api/projects"),
    getProject:     (id)       => request("GET",  `/api/projects/${encodeURIComponent(id)}`),
    startProject:   (id)       => request("POST", `/api/projects/${encodeURIComponent(id)}/start`),
    stopProject:    (id)       => request("POST", `/api/projects/${encodeURIComponent(id)}/stop`),
    restartProject: (id)       => request("POST", `/api/projects/${encodeURIComponent(id)}/restart`),
    updateProject:  (id, data) => request("PUT",  `/api/projects/${encodeURIComponent(id)}`, data),
    wakeAll:        ()         => request("POST", "/api/wake-all"),
    sleepAll:       ()         => request("POST", "/api/sleep-all"),
    getWakeProgress:()         => request("GET",  "/api/wake-progress"),
    getSettings:    ()         => request("GET",  "/api/settings"),
    putSettings:    (data)     => request("PUT",  "/api/settings", data),
    getMemory:      ()         => request("GET",  "/api/memory"),
    checkPackages:  (id)       => request("GET",  `/api/packages/${encodeURIComponent(id)}`),

    /**
     * Open an SSE stream for a project's logs.
     * @param {string}   id
     * @param {function} onLine   - called with each log line string
     * @returns {EventSource}
     */
    streamLogs(id, onLine) {
      const es = new EventSource(`/api/projects/${encodeURIComponent(id)}/logs`);
      es.onmessage = (e) => {
        if (typeof onLine === "function") onLine(e.data);
      };
      es.onerror = () => {
        // SSE reconnects automatically; dashboard will show status
      };
      return es;
    },
  };
})();
