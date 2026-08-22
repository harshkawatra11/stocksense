# Desktop Control Room

**Status: BUILT.** Per `docs/STATUS.md`'s discipline, this document describes what actually exists, not an aspiration.

## Why this exists

The original architecture (`docs/01-architecture.md`, `docs/07-control-room.md`) specified an Electron + Python two-process desktop application. That was 0% built for most of this project's history. This document records the concrete implementation, built in one session after reviewing github.com/omkute101/PolyAgent's live terminal dashboard and deciding: take the aesthetic, reject the weaker research/validation discipline underneath it, and deliver it as a genuine double-clickable Electron desktop app rather than a bare terminal window — closing the gap between "offline desktop app" (the original brief) and "the terminal look" (what was actually liked).

## Architecture

```
┌──────────────────────────────┐        ┌───────────────────────────┐
│  Electron shell (desktop/)   │  HTTP  │  Python API (FastAPI)     │
│  · main.js spawns the API    │ ─────▶ │  reads DuckDB, returns    │
│  · BrowserWindow             │ ◀───── │  JSON. No HTML rendering. │
│  · renderer = terminal CSS   │  JSON  │  Runs standalone too.     │
└──────────────────────────────┘        └───────────────────────────┘
```

Two processes communicating over HTTP, not stdio IPC — deliberately, so the API is independently testable (`tests/unit/test_server_app.py` uses FastAPI's `TestClient`, no Electron involved) and degrades to "open a browser tab" if the Electron shell is ever broken or dropped. Electron is a *view*; the business logic never depends on it.

## Components

| Path | Role |
|---|---|
| `src/stocksense/server/app.py` | Local JSON API: `/api/health`, `/api/summary`, `/api/doshas`, `/api/counterfactuals`, `/api/positions`, `/api/harness`, `/api/registry`, `/api/agent-runs`, `/api/research/docs`, `/api/research/doc/{name}`, `/api/ask`, `/api/jobs/*` (Phase F1/G — a closed command allowlist, `server/jobs.py`'s `COMMANDS`, is the only write path, never arbitrary execution), and (Phase G5) `/api/brief` — the daily top-N recommendation, weighted from the live model's latest predictions, with no capital query parameter and no capital-shaped field in its response (`optimizer/sizing.py`'s `min_capital_for_full_positions` is a whole-share-divisibility floor, not an account-size input or output). |
| `src/stocksense/server/run.py` | The actual `uvicorn.run` call site — bound to `127.0.0.1` **only**, never `0.0.0.0`. This is a security requirement (the API serves private P&L, positions, tax exposure), enforced at this one call site rather than left as a convention, and covered by a test that greps the module source for the literal string that would expose it. |
| `desktop/main.js` | Electron main process. Finds a free port, spawns `python -m stocksense.cli.main serve --port N` as a child process, polls `/api/health` until ready, opens a frameless-adjacent `BrowserWindow`, and — the specific failure mode this file exists to prevent — kills the Python child on every exit path (`window-all-closed`, `before-quit`, `SIGINT`), using `taskkill /T` on Windows since a plain `.kill()` does not reliably terminate a `python.exe` child there. |
| `desktop/preload.js` | Context-isolated bridge. `nodeIntegration: false`, `contextIsolation: true`, `sandbox: true`. Currently exposes nothing — the renderer only needs `fetch()` against the local API, no Node/filesystem access. |
| `desktop/renderer/` | `index.html` + `terminal.css` + `dashboard.js`. Monospace, ANSI-derived severity palette, box-drawing-style panel borders, 2Hz-equivalent (2s) polling — deliberately imitating PolyAgent's `Live`+`Layout` dashboard, rebuilt in CSS rather than Rich since this is a real GUI window, not a terminal. |

## The security-relevant details

- **API binds to `127.0.0.1` only.** Tested directly (`test_server_run.py`): `uvicorn.run` is asserted to receive `host="127.0.0.1"`, and the module source is checked for the absence of any code path that could construct a wildcard bind address.
- **Renderer has zero Node/filesystem access.** `contextIsolation` + `sandbox` + no `nodeIntegration`. The API port is passed via a URL query parameter at `loadFile` time, not injected via `executeJavaScript` after load (which would race the page's own scripts).
- **The footer permanently reads `RESEARCH · dry-run · no orders placed`.** StockSense is recommend-only end to end (`optimizer/rebalance.py` never places an order); the UI states this rather than implying it.
- **No orphaned processes.** Every Electron exit path explicitly kills the spawned Python child; this is the one failure mode the plan named as the reason to avoid stdio IPC in favor of a killable child process.

## Running it

```
cd desktop
npm install     # first time only
npm start       # spawns the API, opens the window
```

Standalone (no Electron): `stocksense serve [--port N]`, then open a browser at `http://127.0.0.1:<port>/api/health` or point any HTTP client at the endpoints above.

Packaging (`npm run build`, via `electron-builder`) produces a Windows installer. Python is currently a separate local install, not bundled into the package — a real limitation for true one-click distribution to someone else's machine, tracked as a follow-up, not a blocker for this build's own use.

## What this does not do

Place trades, write to the database, or run any computation. It is a window onto state that CLI commands and the reconcile/harness loops already produced. If nothing has been ingested or trained yet, every panel says so plainly (`No positions ingested yet`, `No models registered yet`) rather than showing stale or fabricated data.

## Verification boundary — be aware of this before trusting "it works"

Everything **except the actual visible window** was verified directly in the environment that built this:

- `node --check` on all three JS files (syntax valid).
- `npm install` with the version bumped to `electron@43`/`electron-builder@26.15.3` (the originally-pinned versions had 10 known CVEs, 9 high + 1 critical, per `npm audit`; the current pins resolve to **0 vulnerabilities**).
- The Electron binary itself downloads and installs correctly (307MB+ present under `node_modules/electron/dist`).
- The API works correctly end-to-end against a real seeded database, via both `curl` and FastAPI's `TestClient` (`test_server_app.py`, 9 tests).
- Every individual piece of `main.js`'s logic that *can* be unit-tested in isolation is (`test_server_run.py`: port-finding, the 127.0.0.1-only binding, health-check waiting).

**What could not be verified here:** the actual rendered window. The sandboxed environment this was built in has `ELECTRON_RUN_AS_NODE=1` set, which is a documented Electron feature that forces the Electron binary to run as a headless Node process instead of launching a GUI — almost certainly a deliberate sandbox safety restriction against spawning windows from an agent session, not a bug. Attempting to launch the app under this setting fails with `TypeError: Cannot read properties of undefined (reading 'whenReady')`, because `require('electron')` returns a plain-Node shim with no `app`/`BrowserWindow` when run this way. **This is expected and does not indicate a problem with the code** — it's the same failure any Electron app would show under that env var.

**Run `cd desktop && npm start` in a normal interactive session (not through this agent) to see the actual window.** If the terminal aesthetic needs adjusting, that's real UI feedback only a human looking at the rendered window can give — this was flagged from the start as one of the two backlog items (the other being the historical data backfill) that genuinely need your eyes rather than another automated pass.
