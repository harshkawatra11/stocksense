// Tab router (Phase F5). No framework, no history/URL routing needed --
// this is a single-window desktop app with a fixed set of tabs, so a
// plain show/hide over data-tab attributes is all the "routing" this
// needs. Each panel module can register an onActivate callback so it
// only starts polling/fetching once its tab is actually visible, not
// the moment the app launches.

const _tabActivateHandlers = {};

function onTabActivate(tabName, handler) {
  (_tabActivateHandlers[tabName] = _tabActivateHandlers[tabName] || []).push(handler);
}

function activateTab(tabName) {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === tabName);
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `tab-${tabName}`);
  });
  (_tabActivateHandlers[tabName] || []).forEach((fn) => {
    try {
      fn();
    } catch (e) {
      console.error(`tab activate handler failed for ${tabName}:`, e);
    }
  });
}

function initRouter() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => activateTab(btn.dataset.tab));
  });
  const initial = document.querySelector(".tab-btn.active")?.dataset.tab || "dashboard";
  activateTab(initial);
}

document.addEventListener("DOMContentLoaded", initRouter);
