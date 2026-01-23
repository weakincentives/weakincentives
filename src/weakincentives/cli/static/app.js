// ============================================================================
// STATE
// ============================================================================

const state = {
  meta: null,
  bundles: [],
  activeView: "sessions",
  theme: localStorage.getItem("wink-theme") || "light",
  sidebarCollapsed: localStorage.getItem("wink-sidebar-collapsed") === "true",
  // Sessions (slices) state
  selectedSlice: null,
  sliceBuckets: { state: [], event: [] },
  currentItems: [],
  openPaths: new Set(),
  closedPaths: new Set(),
  markdownViews: new Map(),
  expandDepth: 2,
  searchQuery: "",
  focusedItemIndex: -1,
  // Logs state
  allLogs: [],
  filteredLogs: [],
  logsLevels: new Set(["DEBUG", "INFO", "WARNING", "ERROR"]),
  logsSearch: "",
  // Logs pagination
  logsLimit: 200,
  logsTotalCount: 0,
  logsHasMore: false,
  logsLoading: false,
  // Task state
  taskView: "input",
  taskInput: null,
  taskOutput: null,
  taskExpandDepth: 2,
  // Filesystem state
  allFiles: [],
  filesystemFiles: [],
  selectedFile: null,
  fileContent: null,
  filesystemFilter: "",
  hasFilesystemSnapshot: false,
  // Shortcuts overlay
  shortcutsOpen: false,
};

const MARKDOWN_KEY = "__markdown__";

// ============================================================================
// ELEMENTS
// ============================================================================

const elements = {
  // Header
  bundleStatus: document.getElementById("bundle-status"),
  bundleId: document.getElementById("bundle-id"),
  requestId: document.getElementById("request-id"),
  bundleSelect: document.getElementById("bundle-select"),
  themeToggle: document.getElementById("theme-toggle"),
  reloadButton: document.getElementById("reload-button"),
  helpButton: document.getElementById("help-button"),
  // Views
  sessionsView: document.getElementById("sessions-view"),
  logsView: document.getElementById("logs-view"),
  taskView: document.getElementById("task-view"),
  filesystemView: document.getElementById("filesystem-view"),
  // Sessions (slices)
  sliceFilter: document.getElementById("slice-filter"),
  stateSliceList: document.getElementById("state-slice-list"),
  eventSliceList: document.getElementById("event-slice-list"),
  stateSliceCount: document.getElementById("state-slice-count"),
  eventSliceCount: document.getElementById("event-slice-count"),
  sliceEmptyState: document.getElementById("slice-empty-state"),
  sliceContent: document.getElementById("slice-content"),
  sliceTitle: document.getElementById("slice-title"),
  itemCount: document.getElementById("item-count"),
  typeRow: document.getElementById("type-row"),
  jsonViewer: document.getElementById("json-viewer"),
  itemSearch: document.getElementById("item-search"),
  depthInput: document.getElementById("depth-input"),
  expandAll: document.getElementById("expand-all"),
  collapseAll: document.getElementById("collapse-all"),
  copyButton: document.getElementById("copy-button"),
  // Logs
  logsSearch: document.getElementById("logs-search"),
  logsClearFilters: document.getElementById("logs-clear-filters"),
  logsShowing: document.getElementById("logs-showing"),
  logsCopy: document.getElementById("logs-copy"),
  logsScrollBottom: document.getElementById("logs-scroll-bottom"),
  logsList: document.getElementById("logs-list"),
  // Task
  taskPanelTitle: document.getElementById("task-panel-title"),
  taskContent: document.getElementById("task-content"),
  taskDepthInput: document.getElementById("task-depth-input"),
  taskExpandAll: document.getElementById("task-expand-all"),
  taskCollapseAll: document.getElementById("task-collapse-all"),
  taskCopy: document.getElementById("task-copy"),
  // Filesystem
  filesystemFilter: document.getElementById("filesystem-filter"),
  filesystemList: document.getElementById("filesystem-list"),
  filesystemEmptyState: document.getElementById("filesystem-empty-state"),
  filesystemNoSnapshot: document.getElementById("filesystem-no-snapshot"),
  filesystemContent: document.getElementById("filesystem-content"),
  filesystemCurrentPath: document.getElementById("filesystem-current-path"),
  filesystemViewer: document.getElementById("filesystem-viewer"),
  filesystemCopy: document.getElementById("filesystem-copy"),
  // Overlays
  loadingOverlay: document.getElementById("loading-overlay"),
  toastContainer: document.getElementById("toast-container"),
  shortcutsOverlay: document.getElementById("shortcuts-overlay"),
  shortcutsClose: document.getElementById("shortcuts-close"),
};

// ============================================================================
// THEME
// ============================================================================

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  state.theme = theme;
  localStorage.setItem("wink-theme", theme);
}

function toggleTheme() {
  applyTheme(state.theme === "dark" ? "light" : "dark");
}

applyTheme(state.theme);
elements.themeToggle.addEventListener("click", toggleTheme);

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
// LOADING & TOASTS
// ============================================================================

function setLoading(loading) {
  elements.loadingOverlay.classList.toggle("hidden", !loading);
}

function showToast(message, type = "default") {
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  elements.toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.classList.add("hiding");
    setTimeout(() => toast.remove(), 200);
  }, 2500);
}

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
// VIEW SWITCHING
// ============================================================================

function switchView(viewName) {
  state.activeView = viewName;

  // Update tabs
  document.querySelectorAll(".main-tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.view === viewName);
  });

  // Update views
  elements.sessionsView.classList.toggle("hidden", viewName !== "sessions");
  elements.logsView.classList.toggle("hidden", viewName !== "logs");
  elements.taskView.classList.toggle("hidden", viewName !== "task");
  elements.filesystemView.classList.toggle("hidden", viewName !== "filesystem");

  // Load data if needed
  if (viewName === "logs" && state.allLogs.length === 0) {
    loadLogs();
  } else if (viewName === "task" && state.taskInput === null) {
    loadTaskData();
  } else if (viewName === "filesystem" && state.allFiles.length === 0) {
    loadFilesystem();
  }
}

document.querySelectorAll(".main-tab").forEach((tab) => {
  tab.addEventListener("click", () => switchView(tab.dataset.view));
});

// ============================================================================
// BUNDLE MANAGEMENT
// ============================================================================

function renderBundleInfo(meta) {
  elements.bundleStatus.textContent = meta.status;
  elements.bundleStatus.className = `pill status-${meta.status}`;
  elements.bundleId.textContent = meta.bundle_id.slice(0, 8);
  elements.bundleId.title = meta.bundle_id;
  elements.requestId.textContent = meta.request_id.slice(0, 8);
  elements.requestId.title = meta.request_id;
}

async function refreshBundles() {
  const bundles = await fetchJSON("/api/bundles");
  state.bundles = bundles;
  elements.bundleSelect.innerHTML = "";
  bundles.forEach((entry) => {
    const option = document.createElement("option");
    option.value = entry.path;
    option.textContent = entry.name;
    if (entry.selected) option.selected = true;
    elements.bundleSelect.appendChild(option);
  });
}

async function switchBundle(path) {
  try {
    await fetchJSON("/api/switch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    resetViewState();
    await refreshMeta();
    await refreshBundles();
    showToast("Switched bundle", "success");
  } catch (error) {
    showToast(`Switch failed: ${error.message}`, "error");
  }
}

elements.bundleSelect.addEventListener("change", (e) => {
  if (e.target.value) switchBundle(e.target.value);
});

function resetViewState() {
  state.allLogs = [];
  state.filteredLogs = [];
  state.logsTotalCount = 0;
  state.logsHasMore = false;
  state.taskInput = null;
  state.taskOutput = null;
  state.allFiles = [];
  state.filesystemFiles = [];
  state.selectedFile = null;
  state.fileContent = null;
  state.hasFilesystemSnapshot = false;
}

async function reloadBundle() {
  elements.reloadButton.classList.add("spinning");
  try {
    await fetchJSON("/api/reload", { method: "POST" });
    resetViewState();
    await refreshMeta();
    await refreshBundles();
    showToast("Bundle reloaded", "success");
  } catch (error) {
    showToast(`Reload failed: ${error.message}`, "error");
  } finally {
    elements.reloadButton.classList.remove("spinning");
  }
}

elements.reloadButton.addEventListener("click", reloadBundle);

// ============================================================================
// SLICES
// ============================================================================

async function isEventSlice(entry) {
  if (!entry.count) return false;
  try {
    const encoded = encodeURIComponent(entry.slice_type);
    const detail = await fetchJSON(`/api/slices/${encoded}?limit=1`);
    const sample = detail.items[0];
    if (sample && typeof sample === "object") {
      return (
        Object.prototype.hasOwnProperty.call(sample, "event_id") &&
        Object.prototype.hasOwnProperty.call(sample, "created_at")
      );
    }
  } catch (e) {
    console.warn("Failed to classify slice", entry.slice_type, e);
  }
  return false;
}

async function bucketSlices(entries) {
  const buckets = { state: [], event: [] };
  // Classify all entries concurrently, preserving order via Promise.all
  const classifications = await Promise.all(entries.map((entry) => isEventSlice(entry)));
  // Bucket entries in original order based on classification results
  entries.forEach((entry, i) => {
    buckets[classifications[i] ? "event" : "state"].push(entry);
  });
  return buckets;
}

function renderSliceList() {
  const filter = elements.sliceFilter.value.toLowerCase().trim();

  const renderBucket = (target, entries, emptyLabel) => {
    target.innerHTML = "";
    const filtered = entries.filter(
      (e) =>
        (e.display_name || e.slice_type).toLowerCase().includes(filter) ||
        (e.item_display_name || e.item_type).toLowerCase().includes(filter)
    );
    if (!filtered.length) {
      const li = document.createElement("li");
      li.className = "slice-item muted";
      li.textContent = emptyLabel;
      target.appendChild(li);
      return;
    }
    filtered.forEach((entry) => {
      const li = document.createElement("li");
      li.className = "slice-item" + (entry.slice_type === state.selectedSlice ? " active" : "");
      li.innerHTML = `
        <div class="slice-title">${entry.display_name || entry.slice_type}</div>
        <div class="slice-subtitle">${entry.item_display_name || entry.item_type} · ${entry.count} items</div>
      `;
      li.addEventListener("click", () => selectSlice(entry.slice_type));
      target.appendChild(li);
    });
  };

  renderBucket(elements.stateSliceList, state.sliceBuckets.state, "No state slices");
  renderBucket(elements.eventSliceList, state.sliceBuckets.event, "No event slices");
  elements.stateSliceCount.textContent = `${state.sliceBuckets.state.length}`;
  elements.eventSliceCount.textContent = `${state.sliceBuckets.event.length}`;
}

elements.sliceFilter.addEventListener("input", renderSliceList);

async function selectSlice(sliceType) {
  state.selectedSlice = sliceType;
  renderSliceList();

  try {
    const encoded = encodeURIComponent(sliceType);
    const detail = await fetchJSON(`/api/slices/${encoded}`);
    renderSliceDetail(detail);
  } catch (error) {
    elements.jsonViewer.textContent = error.message;
  }
}

function renderSliceDetail(slice) {
  elements.sliceEmptyState.classList.add("hidden");
  elements.sliceContent.classList.remove("hidden");

  elements.sliceTitle.textContent = slice.display_name || slice.slice_type;
  elements.itemCount.textContent = `${slice.items.length} items`;

  elements.typeRow.innerHTML = `
    <span class="pill">slice: ${slice.display_name || slice.slice_type}</span>
    <span class="pill">item: ${slice.item_display_name || slice.item_type}</span>
  `;

  state.currentItems = slice.items;
  state.markdownViews = new Map();
  state.searchQuery = "";
  elements.itemSearch.value = "";

  applyDepth(state.currentItems, state.expandDepth);
  renderItems(state.currentItems);
}

// Slice toolbar events
elements.itemSearch.addEventListener("input", () => {
  state.searchQuery = elements.itemSearch.value || "";
  renderItems(state.currentItems);
});

elements.depthInput.addEventListener("change", () => {
  const value = Number(elements.depthInput.value);
  const depth = Number.isFinite(value) ? Math.max(1, Math.min(10, value)) : 1;
  state.expandDepth = depth;
  elements.depthInput.value = String(depth);
  applyDepth(state.currentItems, depth);
  renderItems(state.currentItems);
});

elements.expandAll.addEventListener("click", () => {
  setOpenForAll(state.currentItems, true);
  renderItems(state.currentItems);
});

elements.collapseAll.addEventListener("click", () => {
  setOpenForAll(state.currentItems, false);
  renderItems(state.currentItems);
});

elements.copyButton.addEventListener("click", async () => {
  const text = JSON.stringify(getFilteredItems().map((e) => e.item), null, 2);
  try {
    await navigator.clipboard.writeText(text);
    showToast("Copied to clipboard", "success");
  } catch {
    showToast("Failed to copy", "error");
  }
});

// ============================================================================
// LOGS
// ============================================================================

async function loadLogs(append = false) {
  if (state.logsLoading) return;

  try {
    state.logsLoading = true;
    const offset = append ? state.allLogs.length : 0;
    const result = await fetchJSON(`/api/logs?offset=${offset}&limit=${state.logsLimit}`);
    const entries = result.entries || [];

    state.allLogs = append ? state.allLogs.concat(entries) : entries;
    state.logsTotalCount = result.total || state.allLogs.length;
    state.logsHasMore = state.allLogs.length < state.logsTotalCount;
    applyLogsFilters();
  } catch (error) {
    elements.logsList.innerHTML = `<p class="muted">Failed to load logs: ${error.message}</p>`;
  } finally {
    state.logsLoading = false;
  }
}

async function loadMoreLogs() {
  await loadLogs(true);
}

function applyLogsFilters() {
  const search = state.logsSearch.toLowerCase();

  state.filteredLogs = state.allLogs.filter((log) => {
    // Level filter
    const level = (log.level || "INFO").toUpperCase();
    if (!state.logsLevels.has(level)) return false;

    // Search filter
    if (search) {
      const message = (log.message || "").toLowerCase();
      const event = (log.event || "").toLowerCase();
      const context = JSON.stringify(log.context || {}).toLowerCase();
      if (!message.includes(search) && !event.includes(search) && !context.includes(search)) {
        return false;
      }
    }

    return true;
  });

  renderLogs();
  updateLogsStats();
}

function updateLogsStats() {
  let status = `Showing ${state.filteredLogs.length} of ${state.allLogs.length}`;
  if (state.logsHasMore) status += ` (${state.logsTotalCount} total)`;
  elements.logsShowing.textContent = status;
}

function renderLogs() {
  elements.logsList.innerHTML = "";

  if (state.filteredLogs.length === 0) {
    elements.logsList.innerHTML = '<div class="logs-empty">No log entries match filters</div>';
    return;
  }

  state.filteredLogs.forEach((log, index) => {
    const level = (log.level || "INFO").toLowerCase();
    const entry = document.createElement("div");
    entry.className = `log-entry log-${level}`;
    entry.dataset.index = index;

    let html = `<div class="log-header">`;
    html += `<span class="log-level">${(log.level || "INFO").toUpperCase()}</span>`;
    if (log.timestamp) {
      html += `<span class="log-timestamp">${log.timestamp}</span>`;
    }
    if (log.logger) {
      html += `<span class="log-logger">${escapeHtml(log.logger)}</span>`;
    }
    if (log.event) {
      html += `<span class="log-event-name">${escapeHtml(log.event)}</span>`;
    }
    html += `</div>`;

    if (log.message) {
      html += `<div class="log-message">${escapeHtml(log.message)}</div>`;
    }

    if (log.context && Object.keys(log.context).length > 0) {
      html += `<pre class="log-context">${escapeHtml(JSON.stringify(log.context, null, 2))}</pre>`;
    }

    if (log.exc_info) {
      html += `<pre class="log-exception">${escapeHtml(log.exc_info)}</pre>`;
    }

    entry.innerHTML = html;
    elements.logsList.appendChild(entry);
  });

  // Add "Load more" button if there are more logs
  if (state.logsHasMore) {
    const loadMoreContainer = document.createElement("div");
    loadMoreContainer.className = "logs-load-more";
    const remaining = state.logsTotalCount - state.allLogs.length;
    loadMoreContainer.innerHTML = `
      <button class="ghost logs-load-more-btn" type="button">
        Load more (${remaining} remaining)
      </button>
    `;
    loadMoreContainer.querySelector("button").addEventListener("click", loadMoreLogs);
    elements.logsList.appendChild(loadMoreContainer);
  }
}

// Logs filter events
elements.logsSearch.addEventListener("input", () => {
  state.logsSearch = elements.logsSearch.value;
  applyLogsFilters();
});

// Level checkboxes
document.querySelectorAll(".level-checkbox input").forEach((checkbox) => {
  checkbox.addEventListener("change", () => {
    if (checkbox.checked) {
      state.logsLevels.add(checkbox.value);
    } else {
      state.logsLevels.delete(checkbox.value);
    }
    applyLogsFilters();
  });
});

elements.logsClearFilters.addEventListener("click", () => {
  state.logsSearch = "";
  state.logsLevels = new Set(["DEBUG", "INFO", "WARNING", "ERROR"]);

  elements.logsSearch.value = "";
  document.querySelectorAll(".level-checkbox input").forEach((cb) => (cb.checked = true));

  applyLogsFilters();
});

elements.logsCopy.addEventListener("click", async () => {
  const text = JSON.stringify(state.filteredLogs, null, 2);
  try {
    await navigator.clipboard.writeText(text);
    showToast("Copied filtered logs", "success");
  } catch {
    showToast("Failed to copy", "error");
  }
});

elements.logsScrollBottom.addEventListener("click", () => {
  elements.logsList.scrollTop = elements.logsList.scrollHeight;
});

// ============================================================================
// TASK
// ============================================================================

async function loadTaskData() {
  try {
    const [input, output] = await Promise.all([
      fetchJSON("/api/request/input"),
      fetchJSON("/api/request/output"),
    ]);
    state.taskInput = input;
    state.taskOutput = output;
    renderTaskData();
  } catch (error) {
    elements.taskContent.innerHTML = `<p class="muted">Failed to load task data: ${error.message}</p>`;
  }
}

function renderTaskData() {
  const data = state.taskView === "input" ? state.taskInput : state.taskOutput;
  elements.taskPanelTitle.textContent = state.taskView === "input" ? "Input" : "Output";
  elements.taskContent.innerHTML = "";

  if (data === null || data === undefined) {
    elements.taskContent.innerHTML = `<p class="muted">No data available</p>`;
    return;
  }

  const container = document.createElement("div");
  container.className = "tree-root";
  container.appendChild(renderTree(data, ["task"], 0, "root"));
  elements.taskContent.appendChild(container);
}

document.querySelectorAll(".request-nav-item").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".request-nav-item").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    state.taskView = btn.dataset.requestPanel;
    renderTaskData();
  });
});

elements.taskCopy.addEventListener("click", async () => {
  const data = state.taskView === "input" ? state.taskInput : state.taskOutput;
  try {
    await navigator.clipboard.writeText(JSON.stringify(data, null, 2));
    showToast("Copied to clipboard", "success");
  } catch {
    showToast("Failed to copy", "error");
  }
});

elements.taskDepthInput.addEventListener("change", () => {
  const value = Number(elements.taskDepthInput.value);
  const depth = Number.isFinite(value) ? Math.max(1, Math.min(10, value)) : 2;
  state.taskExpandDepth = depth;
  elements.taskDepthInput.value = String(depth);
  state.openPaths = new Set();
  state.closedPaths = new Set();
  renderTaskData();
});

elements.taskExpandAll.addEventListener("click", () => {
  const data = state.taskView === "input" ? state.taskInput : state.taskOutput;
  if (data) {
    setOpenForAll([data], true);
    renderTaskData();
  }
});

elements.taskCollapseAll.addEventListener("click", () => {
  const data = state.taskView === "input" ? state.taskInput : state.taskOutput;
  if (data) {
    setOpenForAll([data], false);
    renderTaskData();
  }
});

// ============================================================================
// FILESYSTEM
// ============================================================================

const FILESYSTEM_PREFIX = "filesystem/";

async function loadFilesystem() {
  try {
    const files = await fetchJSON("/api/files");
    state.allFiles = files;

    // Filter to only include filesystem snapshot files
    state.filesystemFiles = files
      .filter((f) => f.startsWith(FILESYSTEM_PREFIX))
      .map((f) => f.slice(FILESYSTEM_PREFIX.length)); // Remove prefix for display

    state.hasFilesystemSnapshot = state.filesystemFiles.length > 0;
    renderFilesystemList();
  } catch (error) {
    elements.filesystemList.innerHTML = `<p class="muted">Failed to load files: ${error.message}</p>`;
  }
}

function renderFilesystemList() {
  // Handle no snapshot state
  if (!state.hasFilesystemSnapshot) {
    elements.filesystemList.innerHTML = "";
    elements.filesystemEmptyState.classList.add("hidden");
    elements.filesystemNoSnapshot.classList.remove("hidden");
    elements.filesystemContent.classList.add("hidden");
    return;
  }

  elements.filesystemNoSnapshot.classList.add("hidden");

  const filter = state.filesystemFilter.toLowerCase();
  const filtered = state.filesystemFiles.filter((f) => f.toLowerCase().includes(filter));

  elements.filesystemList.innerHTML = "";

  if (filtered.length === 0) {
    elements.filesystemList.innerHTML = '<p class="muted">No files match filter</p>';
    return;
  }

  filtered.forEach((displayPath) => {
    const item = document.createElement("div");
    const fullPath = FILESYSTEM_PREFIX + displayPath;
    item.className = "file-item" + (fullPath === state.selectedFile ? " active" : "");
    item.textContent = displayPath;
    item.addEventListener("click", () => selectFilesystemFile(fullPath, displayPath));
    elements.filesystemList.appendChild(item);
  });
}

elements.filesystemFilter.addEventListener("input", () => {
  state.filesystemFilter = elements.filesystemFilter.value;
  renderFilesystemList();
});

async function selectFilesystemFile(fullPath, displayPath) {
  state.selectedFile = fullPath;
  renderFilesystemList();

  try {
    const result = await fetchJSON(`/api/files/${encodeURIComponent(fullPath)}`);

    elements.filesystemEmptyState.classList.add("hidden");
    elements.filesystemNoSnapshot.classList.add("hidden");
    elements.filesystemContent.classList.remove("hidden");
    elements.filesystemCurrentPath.textContent = displayPath;

    if (result.type === "binary") {
      elements.filesystemViewer.innerHTML = '<p class="muted">Binary file cannot be displayed</p>';
      state.fileContent = null;
    } else {
      const content = result.type === "json" ? JSON.stringify(result.content, null, 2) : result.content;
      state.fileContent = content;
      elements.filesystemViewer.innerHTML = `<pre class="file-content">${escapeHtml(content)}</pre>`;
    }
  } catch (error) {
    elements.filesystemViewer.innerHTML = `<p class="muted">Failed to load file: ${error.message}</p>`;
  }
}

elements.filesystemCopy.addEventListener("click", async () => {
  if (!state.fileContent) {
    showToast("No content to copy", "error");
    return;
  }
  try {
    await navigator.clipboard.writeText(state.fileContent);
    showToast("Copied to clipboard", "success");
  } catch {
    showToast("Failed to copy", "error");
  }
});

// ============================================================================
// SHORTCUTS OVERLAY
// ============================================================================

function openShortcuts() {
  state.shortcutsOpen = true;
  elements.shortcutsOverlay.classList.remove("hidden");
}

function closeShortcuts() {
  state.shortcutsOpen = false;
  elements.shortcutsOverlay.classList.add("hidden");
}

elements.shortcutsClose.addEventListener("click", closeShortcuts);
elements.shortcutsOverlay.querySelector(".shortcuts-backdrop").addEventListener("click", closeShortcuts);
elements.helpButton.addEventListener("click", openShortcuts);

// ============================================================================
// KEYBOARD SHORTCUTS
// ============================================================================

document.addEventListener("keydown", (e) => {
  // Escape - close dialogs
  if (e.key === "Escape") {
    if (state.shortcutsOpen) {
      e.preventDefault();
      closeShortcuts();
    }
    return;
  }

  // Don't process if dialog open or typing in input
  if (state.shortcutsOpen) return;
  const tag = e.target.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

  // Number keys for tabs
  if (e.key >= "1" && e.key <= "4") {
    e.preventDefault();
    const views = ["sessions", "logs", "task", "filesystem"];
    switchView(views[parseInt(e.key, 10) - 1]);
    return;
  }

  // Global shortcuts
  if (e.key === "r" || e.key === "R") { e.preventDefault(); reloadBundle(); return; }
  if (e.key === "d" || e.key === "D") { e.preventDefault(); toggleTheme(); return; }
  if (e.key === "[") { e.preventDefault(); toggleSidebar(); return; }
  if (e.key === "?" || (e.shiftKey && e.key === "/")) { e.preventDefault(); openShortcuts(); return; }
  if (e.key === "/") { e.preventDefault(); focusCurrentSearch(); return; }

  // J/K navigation
  if (e.key === "j" || e.key === "J") { e.preventDefault(); navigateNext(); return; }
  if (e.key === "k" || e.key === "K") { e.preventDefault(); navigatePrev(); return; }

  // Arrow keys for bundles
  if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
    e.preventDefault();
    navigateBundle(e.key === "ArrowRight" ? 1 : -1);
  }
});

function focusCurrentSearch() {
  if (state.activeView === "sessions") elements.itemSearch.focus();
  else if (state.activeView === "logs") elements.logsSearch.focus();
  else if (state.activeView === "filesystem") elements.filesystemFilter.focus();
}

function navigateNext() {
  if (state.activeView === "sessions") focusItem(state.focusedItemIndex + 1);
  else if (state.activeView === "logs") scrollLogsBy(1);
}

function navigatePrev() {
  if (state.activeView === "sessions") focusItem(state.focusedItemIndex - 1);
  else if (state.activeView === "logs") scrollLogsBy(-1);
}

function scrollLogsBy(delta) {
  const entries = elements.logsList.querySelectorAll(".log-entry");
  if (entries.length === 0) return;
  const scrollHeight = entries[0].offsetHeight;
  elements.logsList.scrollBy({ top: scrollHeight * delta, behavior: "smooth" });
}

function focusItem(index) {
  const items = elements.jsonViewer.querySelectorAll(".item-card");
  if (items.length === 0) return;
  const newIndex = Math.max(0, Math.min(index, items.length - 1));
  state.focusedItemIndex = newIndex;
  items.forEach((item, i) => item.classList.toggle("focused", i === newIndex));
  items[newIndex].scrollIntoView({ behavior: "smooth", block: "center" });
}

function selectNextSlice() {
  const all = [...state.sliceBuckets.state, ...state.sliceBuckets.event];
  const idx = all.findIndex((s) => s.slice_type === state.selectedSlice);
  if (idx < all.length - 1) selectSlice(all[idx + 1].slice_type);
}

function selectPrevSlice() {
  const all = [...state.sliceBuckets.state, ...state.sliceBuckets.event];
  const idx = all.findIndex((s) => s.slice_type === state.selectedSlice);
  if (idx > 0) selectSlice(all[idx - 1].slice_type);
}

function navigateBundle(delta) {
  const idx = state.bundles.findIndex((b) => b.path === elements.bundleSelect.value);
  const newIdx = Math.max(0, Math.min(idx + delta, state.bundles.length - 1));
  if (newIdx !== idx) {
    elements.bundleSelect.value = state.bundles[newIdx].path;
    switchBundle(state.bundles[newIdx].path);
  }
}

// ============================================================================
// JSON TREE RENDERING
// ============================================================================

const isObject = (v) => typeof v === "object" && v !== null;
const isPrimitive = (v) => v === null || (typeof v !== "object" && !Array.isArray(v));
const isSimpleArray = (v) => Array.isArray(v) && v.every(isPrimitive);

function getMarkdownPayload(value) {
  if (!value || typeof value !== "object" || Array.isArray(value) || !Object.prototype.hasOwnProperty.call(value, MARKDOWN_KEY)) return null;
  const payload = value[MARKDOWN_KEY];
  return payload && typeof payload.text === "string" && typeof payload.html === "string" ? payload : null;
}

function valueType(value) {
  if (getMarkdownPayload(value)) return "markdown";
  if (Array.isArray(value)) return "array";
  if (value === null) return "null";
  return typeof value;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function pathKey(path) { return path.join("."); }
function getMarkdownView(path) { return state.markdownViews.get(pathKey(path)) || "html"; }
function setMarkdownView(path, view) { state.markdownViews.set(pathKey(path), view); }

function shouldOpen(path, depth) {
  const key = pathKey(path);
  if (state.closedPaths.has(key)) return false;
  if (state.openPaths.has(key)) return true;
  return depth < state.expandDepth;
}

function setOpen(path, open) {
  const key = pathKey(path);
  if (open) { state.openPaths.add(key); state.closedPaths.delete(key); }
  else { state.openPaths.delete(key); state.closedPaths.add(key); }
}

function applyDepth(items, depth) {
  state.openPaths = new Set();
  state.closedPaths = new Set();
  const walk = (value, path, d) => {
    if (!(Array.isArray(value) || isObject(value))) return;
    if (d < depth) setOpen(path, true);
    if (Array.isArray(value)) value.forEach((c, i) => walk(c, path.concat(String(i)), d + 1));
    else Object.entries(value).forEach(([k, v]) => walk(v, path.concat(k), d + 1));
  };
  items.forEach((item, i) => walk(item, [`item-${i}`], 0));
}

function setOpenForAll(items, open) {
  state.openPaths = new Set();
  state.closedPaths = new Set();
  const update = (value, path) => {
    if (!(Array.isArray(value) || isObject(value))) return;
    setOpen(path, open);
    if (Array.isArray(value)) value.forEach((c, i) => update(c, path.concat(String(i))));
    else Object.entries(value).forEach(([k, v]) => update(v, path.concat(k)));
  };
  items.forEach((item, i) => update(item, [`item-${i}`]));
}

function getFilteredItems() {
  const query = state.searchQuery.toLowerCase().trim();
  if (!query) return state.currentItems.map((item, index) => ({ item, index }));
  return state.currentItems
    .map((item, index) => ({ item, index, text: JSON.stringify(item).toLowerCase() }))
    .filter((e) => e.text.includes(query))
    .map(({ item, index }) => ({ item, index }));
}

function renderTree(value, path, depth, label) {
  const node = document.createElement("div");
  node.className = "tree-node";

  const header = document.createElement("div");
  header.className = "tree-header";

  const markdown = getMarkdownPayload(value);
  const type = valueType(value);

  const name = document.createElement("span");
  name.className = "tree-label";
  name.textContent = label;
  header.appendChild(name);

  const badge = document.createElement("span");
  badge.className = "pill pill-quiet";
  badge.textContent = type === "array" ? `array (${value.length})` : type === "object" && value !== null ? `object (${Object.keys(value).length})` : type;
  header.appendChild(badge);

  const body = document.createElement("div");
  body.className = "tree-body";

  const expandable = !markdown && (Array.isArray(value) || isObject(value));

  if (!expandable) {
    const wrapper = document.createElement("div");
    wrapper.className = "leaf-wrapper";

    if (markdown) {
      const view = getMarkdownView(path);
      wrapper.classList.add("markdown-leaf");
      wrapper.innerHTML = `
        <div class="markdown-toggle">
          <button type="button" class="${view === "html" ? "active" : ""}">Rendered</button>
          <button type="button" class="${view === "raw" ? "active" : ""}">Raw</button>
        </div>
        <div class="markdown-section" style="display:${view === "html" ? "flex" : "none"}">
          <div class="markdown-rendered">${markdown.html}</div>
        </div>
        <div class="markdown-section" style="display:${view === "raw" ? "flex" : "none"}">
          <pre class="markdown-raw">${escapeHtml(markdown.text)}</pre>
        </div>
      `;
      const buttons = wrapper.querySelectorAll(".markdown-toggle button");
      const sections = wrapper.querySelectorAll(".markdown-section");
      buttons[0].addEventListener("click", () => { setMarkdownView(path, "html"); buttons[0].classList.add("active"); buttons[1].classList.remove("active"); sections[0].style.display = "flex"; sections[1].style.display = "none"; });
      buttons[1].addEventListener("click", () => { setMarkdownView(path, "raw"); buttons[1].classList.add("active"); buttons[0].classList.remove("active"); sections[1].style.display = "flex"; sections[0].style.display = "none"; });
    } else {
      const leaf = document.createElement("div");
      leaf.className = "tree-leaf";
      leaf.textContent = String(value);
      wrapper.appendChild(leaf);
    }
    body.appendChild(wrapper);
    node.appendChild(header);
    node.appendChild(body);
    return node;
  }

  const childCount = Array.isArray(value) ? value.length : Object.keys(value).length;
  const hasChildren = childCount > 0;

  if (hasChildren) {
    const controls = document.createElement("div");
    controls.className = "tree-controls";

    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "tree-toggle";
    toggle.textContent = shouldOpen(path, depth) ? "Collapse" : "Expand";
    controls.appendChild(toggle);

    toggle.addEventListener("click", () => {
      setOpen(path, !shouldOpen(path, depth));
      // Re-render the appropriate view based on active view
      if (state.activeView === "task") {
        renderTaskData();
      } else {
        renderItems(state.currentItems);
      }
    });

    header.appendChild(controls);
  }

  const childrenContainer = document.createElement("div");
  childrenContainer.className = "tree-children";

  if (!hasChildren) {
    childrenContainer.innerHTML = '<span class="muted">(empty)</span>';
  } else if (!shouldOpen(path, depth)) {
    childrenContainer.style.display = "none";
  } else {
    if (Array.isArray(value)) {
      if (isSimpleArray(value)) {
        childrenContainer.classList.add("compact-array");
        value.forEach((child) => {
          const chip = document.createElement("span");
          chip.className = "array-chip";
          chip.textContent = String(child);
          childrenContainer.appendChild(chip);
        });
      } else {
        value.forEach((child, i) => {
          childrenContainer.appendChild(renderTree(child, path.concat(String(i)), depth + 1, `[${i}]`));
        });
      }
    } else {
      Object.entries(value).forEach(([key, child]) => {
        childrenContainer.appendChild(renderTree(child, path.concat(key), depth + 1, key));
      });
    }
  }

  body.appendChild(childrenContainer);
  node.appendChild(header);
  node.appendChild(body);
  return node;
}

function renderItems(items) {
  elements.jsonViewer.innerHTML = "";
  const filtered = getFilteredItems();

  if (state.searchQuery.trim()) {
    elements.itemCount.textContent = `${filtered.length} of ${items.length} items`;
  } else {
    elements.itemCount.textContent = `${items.length} items`;
  }

  filtered.forEach(({ item, index }) => {
    const card = document.createElement("div");
    card.className = "item-card";
    card.innerHTML = `<div class="item-header"><h3>Item ${index + 1}</h3></div>`;

    const body = document.createElement("div");
    body.className = "item-body tree-root";
    body.appendChild(renderTree(item, [`item-${index}`], 0, "root"));
    card.appendChild(body);
    elements.jsonViewer.appendChild(card);
  });
}

// ============================================================================
// INITIALIZATION
// ============================================================================

async function refreshMeta() {
  const meta = await fetchJSON("/api/meta");
  state.meta = meta;
  state.sliceBuckets = await bucketSlices(meta.slices);
  renderBundleInfo(meta);
  renderSliceList();

  if (!state.selectedSlice && meta.slices.length > 0) {
    await selectSlice(meta.slices[0].slice_type);
  } else if (state.selectedSlice) {
    const exists = meta.slices.some((e) => e.slice_type === state.selectedSlice);
    if (exists) await selectSlice(state.selectedSlice);
    else if (meta.slices.length > 0) await selectSlice(meta.slices[0].slice_type);
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  setLoading(true);
  try {
    await refreshMeta();
    await refreshBundles();
  } catch (error) {
    showToast(`Load failed: ${error.message}`, "error");
  } finally {
    setLoading(false);
  }
});
