// Context-isolated preload. The renderer gets NO filesystem access, no
// Node APIs, and no IPC beyond what's explicitly exposed here -- which
// today is nothing, since the renderer only needs to fetch() the local
// API (passed via URL query param, not this bridge) and render JSON.
// This file exists as the enforcement point for that boundary, not
// because anything is exposed through it yet.

const { contextBridge } = require("electron");

contextBridge.exposeInMainWorld("stocksense", {
  // Intentionally empty. If a future feature needs main-process
  // access (e.g. opening a native file dialog for statement upload),
  // add a narrowly-scoped ipcRenderer.invoke call here -- never
  // re-enable nodeIntegration as a shortcut.
});
