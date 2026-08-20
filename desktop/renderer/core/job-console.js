// Job console (Phase F5, F1's frontend half). A persistent drawer, not
// a separate tab -- every panel's "trigger" button routes through
// triggerJob() here, which opens the drawer and streams the job's real
// stdout live via the WebSocket F1's server exposes. This is the
// closest the control center gets to "still feels like a terminal"
// while every action that fills it is a button, not a typed command.

let _activeJobId = null;
let _activeSocket = null;

function _consoleEl(id) {
  return document.getElementById(id);
}

function _appendLine(text, cls) {
  const body = _consoleEl("job-console-body");
  const line = document.createElement("div");
  line.className = `job-line${cls ? ` ${cls}` : ""}`;
  line.textContent = text;
  body.appendChild(line);
  body.scrollTop = body.scrollHeight;
}

function _setStatus(status) {
  const el = _consoleEl("job-console-status");
  el.textContent = status;
  el.className = `status-${status}`;
}

function openJobConsole() {
  _consoleEl("job-console").classList.add("expanded");
}

function toggleJobConsole() {
  _consoleEl("job-console").classList.toggle("expanded");
}

/**
 * Triggers a job via POST /api/jobs/{command}, opens the console, and
 * streams its output live. Returns the job_id, or throws (e.g. 409 if
 * another job is already running -- DuckDB's single-writer constraint,
 * see server/jobs.py) so callers can show that as a normal error rather
 * than crash the panel.
 */
async function triggerJob(command, params) {
  const { job_id } = await postJson(`/api/jobs/${command}`, params || {});
  _activeJobId = job_id;

  openJobConsole();
  _consoleEl("job-console-body").innerHTML = "";
  _appendLine(`$ ${command} ${JSON.stringify(params || {})}`, "job-cmd");
  _setStatus("running");
  _streamJob(job_id);
  return job_id;
}

function _streamJob(jobId) {
  if (_activeSocket) {
    try { _activeSocket.close(); } catch (e) { /* already closed */ }
  }
  const ws = new WebSocket(`${WS_BASE}/ws/jobs/${jobId}`);
  _activeSocket = ws;

  ws.onmessage = (evt) => {
    const msg = JSON.parse(evt.data);
    if (msg.line !== undefined) {
      _appendLine(msg.line);
    }
    if (msg.done) {
      _setStatus(msg.status || "completed");
      _appendLine(`[job ${msg.status}]`, `job-${msg.status}`);
      document.dispatchEvent(new CustomEvent("job-finished", { detail: { jobId, status: msg.status } }));
    }
  };
  ws.onerror = () => _appendLine("[console connection error]", "job-failed");
  ws.onclose = () => { if (_activeSocket === ws) _activeSocket = null; };
}

async function stopActiveJob() {
  if (!_activeJobId) return;
  await postJson(`/api/jobs/${_activeJobId}/stop`);
  _appendLine("[stop requested]", "job-stopping");
}

function initJobConsole() {
  _consoleEl("job-console-header").addEventListener("click", toggleJobConsole);
  _consoleEl("job-console-stop").addEventListener("click", (e) => {
    e.stopPropagation();
    stopActiveJob().catch((err) => _appendLine(`[stop failed: ${err.message}]`, "job-failed"));
  });
}

document.addEventListener("DOMContentLoaded", initJobConsole);
