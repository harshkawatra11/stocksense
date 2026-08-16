// StockSense desktop shell (docs/18-desktop-app.md).
//
// Two processes: this Electron shell, and a Python (FastAPI) API it
// spawns as a child process. The API is the source of truth; this file
// is only presentation plumbing -- if Electron is ever dropped, the API
// still runs standalone (`stocksense serve`) and nothing about the
// business logic changes.
//
// The failure mode this file exists to prevent: an orphaned Python
// process left running after the window closes. Every exit path
// (window-all-closed, before-quit, an uncaught exception) kills the
// spawned child explicitly rather than assuming OS process-tree cleanup
// will handle it, since on Windows a detached child does not
// automatically die with its parent.

const { app, BrowserWindow } = require("electron");
const { spawn } = require("child_process");
const net = require("net");
const path = require("path");

let pythonProcess = null;
let mainWindow = null;
let apiPort = null;

function findFreePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.listen(0, "127.0.0.1", () => {
      const port = server.address().port;
      server.close(() => resolve(port));
    });
    server.on("error", reject);
  });
}

function waitForHealth(port, timeoutMs = 15000) {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolve, reject) => {
    const attempt = () => {
      const http = require("http");
      const req = http.get(`http://127.0.0.1:${port}/api/health`, (res) => {
        if (res.statusCode === 200) {
          resolve();
        } else if (Date.now() > deadline) {
          reject(new Error(`API health check failed with status ${res.statusCode}`));
        } else {
          setTimeout(attempt, 300);
        }
      });
      req.on("error", () => {
        if (Date.now() > deadline) {
          reject(new Error("API did not become healthy within timeout"));
        } else {
          setTimeout(attempt, 300);
        }
      });
    };
    attempt();
  });
}

function killPythonProcess() {
  if (pythonProcess && !pythonProcess.killed) {
    // On Windows, a plain .kill() often fails to terminate a python.exe
    // child reliably -- taskkill /T kills the whole process tree.
    if (process.platform === "win32") {
      spawn("taskkill", ["/pid", pythonProcess.pid, "/f", "/t"]);
    } else {
      pythonProcess.kill("SIGTERM");
    }
    pythonProcess = null;
  }
}

async function startPythonApi() {
  apiPort = await findFreePort();
  // "python" must be the same interpreter StockSense is installed into;
  // this assumes it's on PATH, matching every other CLI invocation in
  // this project's own tooling (no bundled interpreter yet -- a
  // packaging follow-up, not a blocker for a first working build).
  pythonProcess = spawn("python", ["-m", "stocksense.cli.main", "serve", "--port", String(apiPort)], {
    cwd: path.join(__dirname, ".."),
    stdio: ["ignore", "pipe", "pipe"],
  });

  pythonProcess.stdout.on("data", (d) => process.stdout.write(`[api] ${d}`));
  pythonProcess.stderr.on("data", (d) => process.stderr.write(`[api] ${d}`));
  pythonProcess.on("exit", (code) => {
    console.log(`API process exited with code ${code}`);
  });

  await waitForHealth(apiPort);
  return apiPort;
}

async function createWindow() {
  const port = await startPythonApi();

  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    backgroundColor: "#0b0e0f",
    frame: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
    },
  });

  // Pass the port via URL query param rather than injecting it with
  // executeJavaScript after loadFile -- that injection runs AFTER the
  // page's own scripts have already had a chance to run on load, which
  // is a race dashboard.js could lose. A query param is available to
  // the page's scripts from the very first tick, no race possible.
  await mainWindow.loadFile(path.join(__dirname, "renderer", "index.html"), { query: { port: String(port) } });
}

app.whenReady().then(createWindow);

app.on("window-all-closed", () => {
  killPythonProcess();
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", killPythonProcess);
process.on("exit", killPythonProcess);
process.on("SIGINT", () => {
  killPythonProcess();
  process.exit(0);
});
