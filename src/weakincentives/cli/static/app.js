// ============================================================================
// App - Thin coordinator that initializes the store, mounts views, and handles
// cross-cutting concerns (theme, sidebar, bundle management, view switching).
//
// Each view is a self-contained module in views/ that owns its DOM elements,
// event listeners, and rendering. State lives in a centralized store accessed
// via store.getState().
// ============================================================================

import { createStore } from "./store.js";
import { initEnvironmentView } from "./views/environment-view.js";
import { initFilesystemView } from "./views/filesystem-view.js";
import { initKeyboardShortcuts } from "./views/keyboard-shortcuts.js";
import { initLogsView } from "./views/logs-view.js";
import { initSessionsView } from "./views/sessions-view.js";
import { initTranscriptView } from "./views/transcript-view.js";
import { initZoomModal } from "./views/zoom-modal.js";

// ============================================================================
// STORE
// ============================================================================

const store = createStore();
const state = store.getState();

// ============================================================================
// API
// ============================================================================

async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return response.json();
}

// ============================================================================
// LOADING & TOASTS
// ============================================================================

const loadingOverlay = document.getElementById("loading-overlay");
const toastContainer = document.getElementById("toast-container");

function setLoading(loading) {
  loadingOverlay.classList.toggle("hidden", !loading);
}

function showToast(message, type = "default") {
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.classList.add("hiding");
    setTimeout(() => toast.remove(), 200);
  }, 2500);
}

// ============================================================================
// THEME
// ============================================================================

const themeToggle = document.getElementById("theme-toggle");

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  state.theme = theme;
  localStorage.setItem("wink-theme", theme);
}

function toggleTheme() {
  applyTheme(state.theme === "dark" ? "light" : "dark");
}

applyTheme(state.theme);
themeToggle.addEventListener("click", toggleTheme);

// ============================================================================
// SIDEBAR
// ============================================================================

function applySidebarState() {
  document.querySelectorAll(".view-container").forEach((view) => {
    view.classList.toggle("sidebar-collapsed", state.sidebarCollapsed);
  });
}

function toggleSidebar() {
  state.sidebarCollapsed = !state.sidebarCollapsed;
  localStorage.setItem("wink-sidebar-collapsed", state.sidebarCollapsed);
  applySidebarState();
  showToast(state.sidebarCollapsed ? "Sidebar hidden" : "Sidebar shown");
}

applySidebarState();

// ============================================================================
// VIEW SWITCHING
// ============================================================================

const VIEW_ELEMENTS = {
  sessions: document.getElementById("sessions-view"),
  transcript: document.getElementById("transcript-view"),
  logs: document.getElementById("logs-view"),
  filesystem: document.getElementById("filesystem-view"),
  environment: document.getElementById("environment-view"),
};

function updateViewVisibility(viewName) {
  document.querySelectorAll(".main-tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.view === viewName);
  });
  for (const [name, element] of Object.entries(VIEW_ELEMENTS)) {
    element.classList.toggle("hidden", viewName !== name);
  }
}

// Initialize views (will be set up after DOM content loads)
let sessionsView;
let transcriptView;
let logsView;
let filesystemView;
let environmentView;
let zoomModal;

function initScrollerIfNeeded(viewName) {
  if (viewName === "transcript" && !state.transcriptScroller) {
    transcriptView.initVirtualScroller();
  }
  if (viewName === "logs" && !state.logsScroller) {
    logsView.initVirtualScroller();
  }
}

function loadViewDataIfNeeded(viewName) {
  if (viewName === "transcript" && state.transcriptEntries.length === 0) {
    transcriptView.loadTranscriptFacets();
    transcriptView.loadTranscript();
  }
  if (viewName === "logs" && state.filteredLogs.length === 0) {
    logsView.loadLogFacets();
    logsView.loadLogs();
  }
  if (viewName === "filesystem" && state.allFiles.length === 0) {
    filesystemView.loadFilesystem();
  }
  if (viewName === "environment" && state.environmentData === null) {
    environmentView.loadEnvironment();
  }
}

function switchView(viewName) {
  state.activeView = viewName;
  updateViewVisibility(viewName);
  initScrollerIfNeeded(viewName);
  loadViewDataIfNeeded(viewName);
}

document.querySelectorAll(".main-tab").forEach((tab) => {
  tab.addEventListener("click", () => switchView(tab.dataset.view));
});

// ============================================================================
// BUNDLE MANAGEMENT
// ============================================================================

const bundleStatus = document.getElementById("bundle-status");
const bundleId = document.getElementById("bundle-id");
const requestId = document.getElementById("request-id");
const bundleSelect = document.getElementById("bundle-select");
const reloadButton = document.getElementById("reload-button");

function renderBundleInfo(meta) {
  bundleStatus.textContent = meta.status;
  bundleStatus.className = `pill status-${meta.status}`;
  bundleId.textContent = meta.bundle_id.slice(0, 8);
  bundleId.title = meta.bundle_id;
  requestId.textContent = meta.request_id.slice(0, 8);
  requestId.title = meta.request_id;
}

async function refreshBundles() {
  const bundles = await fetchJSON("/api/bundles");
  state.bundles = bundles;
  bundleSelect.innerHTML = "";
  bundles.forEach((entry) => {
    const option = document.createElement("option");
    option.value = entry.path;
    option.textContent = entry.name;
    if (entry.selected) {
      option.selected = true;
    }
    bundleSelect.appendChild(option);
  });
}

function resetViewState() {
  // Cancel pending async operations and clear timeouts
  transcriptView.reset();
  logsView.reset();

  state.transcriptRawEntries = [];
  state.transcriptEntries = [];
  state.transcriptTotalCount = 0;
  state.transcriptHasMore = false;
  state.allLogs = [];
  state.filteredLogs = [];
  state.logsTotalCount = 0;
  state.logsHasMore = false;
  state.allFiles = [];
  state.filesystemFiles = [];
  state.selectedFile = null;
  state.fileContent = null;
  state.hasFilesystemSnapshot = false;

  if (state.transcriptScroller) {
    state.transcriptScroller.destroy();
    state.transcriptScroller = null;
  }
  if (state.logsScroller) {
    state.logsScroller.destroy();
    state.logsScroller = null;
  }
}

async function switchBundle(path) {
  try {
    await fetchJSON("/api/switch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    resetViewState();
    await sessionsView.refreshMeta();
    renderBundleInfo(state.meta);
    await refreshBundles();
    showToast("Switched bundle", "success");
  } catch (error) {
    showToast(`Switch failed: ${error.message}`, "error");
  }
}

bundleSelect.addEventListener("change", (e) => {
  if (e.target.value) {
    switchBundle(e.target.value);
  }
});

async function reloadBundle() {
  reloadButton.classList.add("spinning");
  try {
    await fetchJSON("/api/reload", { method: "POST" });
    resetViewState();
    await sessionsView.refreshMeta();
    renderBundleInfo(state.meta);
    await refreshBundles();
    showToast("Bundle reloaded", "success");
  } catch (error) {
    showToast(`Reload failed: ${error.message}`, "error");
  } finally {
    reloadButton.classList.remove("spinning");
  }
}

reloadButton.addEventListener("click", reloadBundle);

function navigateBundle(delta) {
  const idx = state.bundles.findIndex((b) => b.path === bundleSelect.value);
  const newIdx = Math.max(0, Math.min(idx + delta, state.bundles.length - 1));
  if (newIdx !== idx) {
    bundleSelect.value = state.bundles[newIdx].path;
    switchBundle(state.bundles[newIdx].path);
  }
}

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener("DOMContentLoaded", async () => {
  setLoading(true);
  try {
    // Initialize all view modules with shared dependencies
    const deps = { store, fetchJSON, showToast };

    sessionsView = initSessionsView(deps);
    transcriptView = initTranscriptView(deps);
    logsView = initLogsView(deps);
    filesystemView = initFilesystemView(deps);
    environmentView = initEnvironmentView(deps);
    zoomModal = initZoomModal({ store, transcriptView, showToast });

    // Initialize keyboard shortcuts (cross-cutting concern)
    initKeyboardShortcuts({
      store,
      sessionsView,
      transcriptView,
      logsView,
      zoomModal,
      switchView,
      reloadBundle,
      toggleTheme,
      toggleSidebar,
      navigateBundle,
    });

    // Initialize virtual scrollers
    logsView.initVirtualScroller();
    transcriptView.initVirtualScroller();

    // Load initial data (sessionsView.refreshMeta fetches /api/meta and
    // sets state.meta, so we render bundle info from that same response
    // rather than making a duplicate request)
    await sessionsView.refreshMeta();
    renderBundleInfo(state.meta);
    await refreshBundles();
  } catch (error) {
    showToast(`Load failed: ${error.message}`, "error");
  } finally {
    setLoading(false);
  }
});
