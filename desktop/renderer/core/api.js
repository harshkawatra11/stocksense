// Shared API client (Phase F5). One source of truth for API_BASE/WS_BASE
// and the fetch wrappers -- every panel file uses these instead of
// re-deriving the port from the URL query string itself, so there's
// exactly one place that knows how the renderer reaches the FastAPI
// server Electron's main.js spawned.

const API_PORT = new URLSearchParams(window.location.search).get("port");
const API_BASE = `http://127.0.0.1:${API_PORT}`;
const WS_BASE = `ws://127.0.0.1:${API_PORT}`;

async function fetchJson(path) {
  const resp = await fetch(`${API_BASE}${path}`);
  if (!resp.ok) {
    const detail = await resp.json().catch(() => ({}));
    throw new Error(detail.detail || `${path} -> ${resp.status}`);
  }
  return resp.json();
}

async function postJson(path, body) {
  const resp = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  if (!resp.ok) {
    const detail = await resp.json().catch(() => ({}));
    throw new Error(detail.detail || `${path} -> ${resp.status}`);
  }
  return resp.json();
}

async function putJson(path, body) {
  const resp = await fetch(`${API_BASE}${path}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  if (!resp.ok) {
    const detail = await resp.json().catch(() => ({}));
    throw new Error(detail.detail || `${path} -> ${resp.status}`);
  }
  return resp.json();
}

// PUT with query-string params rather than a JSON body -- matches
// endpoints like /api/claude/usage/soft-alarm that take a plain
// optional scalar, not a structured payload.
async function putQuery(path, params) {
  const qs = new URLSearchParams(
    Object.entries(params || {}).filter(([, v]) => v !== null && v !== undefined)
  ).toString();
  const resp = await fetch(`${API_BASE}${path}${qs ? `?${qs}` : ""}`, { method: "PUT" });
  if (!resp.ok) {
    const detail = await resp.json().catch(() => ({}));
    throw new Error(detail.detail || `${path} -> ${resp.status}`);
  }
  return resp.json();
}
