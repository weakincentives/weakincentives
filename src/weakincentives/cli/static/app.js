// ============================================================================
// App - Thin coordinator that initializes state, mounts views, and handles
// cross-cutting concerns (theme, sidebar, bundle management, view switching).
//
// Each view is a self-contained module in views/ that owns its DOM elements,
// event listeners, and rendering. All modules share a single state object
// created by createInitialState() and passed directly at initialization.
// ============================================================================

import { createInitialState } from "./store.js";
import { initEnvironmentView } from "./views/environment-view.js";
import { initFilesystemView } from "./views/filesystem-view.js";
import { initKeyboardShortcuts } from "./views/keyboard-shortcuts.js";
import { initLogsView } from "./views/logs-view.js";
import { initSessionsView } from "./views/sessions-view.js";
import { initTranscriptView } from "./views/transcript-view.js";

// ============================================================================
// STATE
// ============================================================================

const state = createInitialState();

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
// SIDEBAR (collapsible + resizable)
// ============================================================================

function applySidebarState() {
  document.querySelectorAll(".view-container").forEach((view) => {
    view.classList.toggle("sidebar-collapsed", state.sidebarCollapsed);
    if (!state.sidebarCollapsed) {
      view.style.gridTemplateColumns = `${state.sidebarWidth}px 1fr`;
    }
  });
}

function toggleSidebar() {
  state.sidebarCollapsed = !state.sidebarCollapsed;
  localStorage.setItem("wink-sidebar-collapsed", state.sidebarCollapsed);
  applySidebarState();
  showToast(state.sidebarCollapsed ? "Sidebar hidden" : "Sidebar shown");
}

applySidebarState();

// Resizable sidebar drag
function initSidebarResize() {
  let resizing = false;
  let startX = 0;
  let startWidth = 0;

  document.addEventListener("mousedown", (e) => {
    // Check if clicking on the sidebar resize handle (::after pseudo-element area)
    const sidebar = e.target.closest(".sidebar");
    if (!sidebar) {
      return;
    }
    const rect = sidebar.getBoundingClientRect();
    if (e.clientX < rect.right - 6 || e.clientX > rect.right + 6) {
      return;
    }

    resizing = true;
    startX = e.clientX;
    startWidth = state.sidebarWidth;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    e.preventDefault();
  });

  document.addEventListener("mousemove", (e) => {
    if (!resizing) {
      return;
    }
    const delta = e.clientX - startX;
    const newWidth = Math.max(180, Math.min(500, startWidth + delta));
    state.sidebarWidth = newWidth;
    document.querySelectorAll(".view-container:not(.sidebar-collapsed)").forEach((view) => {
      view.style.gridTemplateColumns = `${newWidth}px 1fr`;
    });
  });

  document.addEventListener("mouseup", () => {
    if (!resizing) {
      return;
    }
    resizing = false;
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
    localStorage.setItem("wink-sidebar-width", state.sidebarWidth);
  });
}

initSidebarResize();

// ============================================================================
// COMMAND PALETTE
// ============================================================================

const commandPaletteOverlay = document.getElementById("command-palette-overlay");
const commandPaletteInput = document.getElementById("command-palette-input");
const commandPaletteResults = document.getElementById("command-palette-results");

function openCommandPalette() {
  state.commandPaletteOpen = true;
  commandPaletteOverlay.classList.remove("hidden");
  commandPaletteInput.value = "";
  commandPaletteInput.focus();
  renderCommandResults("");
}

function closeCommandPalette() {
  state.commandPaletteOpen = false;
  commandPaletteOverlay.classList.add("hidden");
}

function getCommandItems(query) {
  const items = [];
  const q = query.toLowerCase();

  // Static commands
  const commands = [
    {
      title: "Sessions",
      subtitle: "Switch to sessions view",
      action: () => switchView("sessions"),
      shortcut: "1",
    },
    {
      title: "Transcript",
      subtitle: "Switch to transcript view",
      action: () => switchView("transcript"),
      shortcut: "2",
    },
    {
      title: "Logs",
      subtitle: "Switch to logs view",
      action: () => switchView("logs"),
      shortcut: "3",
    },
    {
      title: "Filesystem",
      subtitle: "Switch to filesystem view",
      action: () => switchView("filesystem"),
      shortcut: "4",
    },
    {
      title: "Environment",
      subtitle: "Switch to environment view",
      action: () => switchView("environment"),
      shortcut: "5",
    },
    {
      title: "Reload Bundle",
      subtitle: "Reload current debug bundle",
      action: reloadBundle,
      shortcut: "R",
    },
    {
      title: "Toggle Theme",
      subtitle: "Switch between dark and light",
      action: toggleTheme,
      shortcut: "D",
    },
    {
      title: "Toggle Sidebar",
      subtitle: "Show or hide the sidebar",
      action: toggleSidebar,
      shortcut: "[",
    },
  ];

  commands.forEach((cmd) => {
    if (!q || cmd.title.toLowerCase().includes(q) || cmd.subtitle.toLowerCase().includes(q)) {
      items.push({ type: "command", ...cmd });
    }
  });

  // Search slices
  if (state.meta?.slices && q) {
    state.meta.slices.forEach((slice) => {
      const name = slice.display_name || slice.slice_type;
      if (name.toLowerCase().includes(q)) {
        items.push({
          type: "slice",
          title: name,
          subtitle: `Slice (${slice.count || 0} items)`,
          action: () => {
            switchView("sessions"); /* slice will be selected via existing flow */
          },
        });
      }
    });
  }

  // Search files
  if (state.filesystemFiles.length > 0 && q) {
    state.filesystemFiles.forEach((file) => {
      if (file.toLowerCase().includes(q)) {
        items.push({
          type: "file",
          title: file,
          subtitle: "File",
          action: () => switchView("filesystem"),
        });
      }
    });
  }

  return items.slice(0, 12);
}

let selectedCommandIndex = 0;

function renderCommandResults(query) {
  const items = getCommandItems(query);
  selectedCommandIndex = 0;

  if (items.length === 0) {
    commandPaletteResults.innerHTML = '<div class="command-palette-empty">No results found</div>';
    return;
  }

  commandPaletteResults.innerHTML = "";
  items.forEach((item, index) => {
    const el = document.createElement("div");
    el.className = `command-result${index === 0 ? " selected" : ""}`;
    el.innerHTML = `
      <div class="command-result-text">
        <div class="command-result-title">${escapeCommandHtml(item.title)}</div>
        <div class="command-result-subtitle">${escapeCommandHtml(item.subtitle)}</div>
      </div>
      ${item.shortcut ? `<div class="command-result-shortcut"><kbd>${item.shortcut}</kbd></div>` : ""}
    `;
    el.addEventListener("click", () => {
      closeCommandPalette();
      item.action();
    });
    el.addEventListener("mouseenter", () => {
      commandPaletteResults.querySelectorAll(".command-result").forEach((r, i) => {
        r.classList.toggle("selected", i === index);
      });
      selectedCommandIndex = index;
    });
    commandPaletteResults.appendChild(el);
  });
}

function escapeCommandHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

commandPaletteInput.addEventListener("input", () => {
  renderCommandResults(commandPaletteInput.value);
});

commandPaletteInput.addEventListener("keydown", (e) => {
  const results = commandPaletteResults.querySelectorAll(".command-result");
  if (e.key === "ArrowDown") {
    e.preventDefault();
    selectedCommandIndex = Math.min(selectedCommandIndex + 1, results.length - 1);
    results.forEach((r, i) => r.classList.toggle("selected", i === selectedCommandIndex));
    results[selectedCommandIndex]?.scrollIntoView({ block: "nearest" });
  } else if (e.key === "ArrowUp") {
    e.preventDefault();
    selectedCommandIndex = Math.max(selectedCommandIndex - 1, 0);
    results.forEach((r, i) => r.classList.toggle("selected", i === selectedCommandIndex));
    results[selectedCommandIndex]?.scrollIntoView({ block: "nearest" });
  } else if (e.key === "Enter") {
    e.preventDefault();
    results[selectedCommandIndex]?.click();
  } else if (e.key === "Escape") {
    e.preventDefault();
    closeCommandPalette();
  }
});

commandPaletteOverlay
  .querySelector(".command-palette-backdrop")
  .addEventListener("click", closeCommandPalette);

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

// View modules (initialized in DOMContentLoaded)
let sessionsView;
let transcriptView;
let logsView;
let filesystemView;
let environmentView;
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

const transcriptTab = document.querySelector('.main-tab[data-view="transcript"]');

function renumberTabShortcuts() {
  let shortcut = 1;
  for (const tab of document.querySelectorAll(".main-tab")) {
    const kbd = tab.querySelector("kbd");
    if (!kbd) {
      continue;
    }
    if (tab.classList.contains("hidden")) {
      kbd.textContent = "";
    } else {
      kbd.textContent = String(shortcut);
      shortcut++;
    }
  }
}

function updateTranscriptTabVisibility(hasTranscript) {
  state.hasTranscript = hasTranscript;
  transcriptTab.classList.toggle("hidden", !hasTranscript);
  // If transcript is the active view but is now hidden, switch to sessions.
  if (!hasTranscript && state.activeView === "transcript") {
    switchView("sessions");
  }
  renumberTabShortcuts();
}

function renderBundleInfo(meta) {
  bundleStatus.textContent = meta.status;
  bundleStatus.className = `pill status-${meta.status}`;
  bundleId.textContent = meta.bundle_id.slice(0, 8);
  bundleId.title = meta.bundle_id;
  requestId.textContent = meta.request_id.slice(0, 8);
  requestId.title = meta.request_id;
  updateTranscriptTabVisibility(meta.has_transcript);
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
  // Cancel pending async operations and clear timeouts.
  // Guard: views are initialized in DOMContentLoaded; resetViewState is only
  // reachable from user-triggered switchBundle/reloadBundle after that, but
  // guard defensively against the ordering constraint.
  if (transcriptView) {
    transcriptView.reset();
  }
  if (logsView) {
    logsView.reset();
  }

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
  state.environmentData = null;
  state.hasEnvironmentData = false;

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
    const deps = { state, fetchJSON, showToast };

    sessionsView = initSessionsView(deps);
    transcriptView = initTranscriptView(deps);
    logsView = initLogsView(deps);
    filesystemView = initFilesystemView(deps);
    environmentView = initEnvironmentView(deps);
    initKeyboardShortcuts({
      state,
      sessionsView,
      transcriptView,
      logsView,
      switchView,
      reloadBundle,
      toggleTheme,
      toggleSidebar,
      navigateBundle,
      openCommandPalette,
      closeCommandPalette,
    });

    logsView.initVirtualScroller();
    transcriptView.initVirtualScroller();

    await sessionsView.refreshMeta();
    renderBundleInfo(state.meta);
    await refreshBundles();
  } catch (error) {
    showToast(`Load failed: ${error.message}`, "error");
  } finally {
    setLoading(false);
  }
});
