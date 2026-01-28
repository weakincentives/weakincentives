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
  // Transcript state
  transcriptEntries: [],
  transcriptFacets: { sources: [], entry_types: [] },
  transcriptSearch: "",
  transcriptSourceChipFilter: "",
  transcriptTypeChipFilter: "",
  transcriptIncludeSources: new Set(),
  transcriptExcludeSources: new Set(),
  transcriptIncludeTypes: new Set(),
  transcriptExcludeTypes: new Set(),
  transcriptLimit: 200,
  transcriptTotalCount: 0,
  transcriptHasMore: false,
  transcriptLoading: false,
  // Logs state
  allLogs: [],
  filteredLogs: [],
  logsLevels: new Set(["DEBUG", "INFO", "WARNING", "ERROR"]),
  logsSearch: "",
  logsFacets: { loggers: [], events: [], levels: [] },
  logsLoggerChipFilter: "",
  logsEventChipFilter: "",
  logsIncludeLoggers: new Set(),
  logsExcludeLoggers: new Set(),
  logsIncludeEvents: new Set(),
  logsExcludeEvents: new Set(),
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
  transcriptView: document.getElementById("transcript-view"),
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
  // Transcript
  transcriptSearch: document.getElementById("transcript-search"),
  transcriptClearFilters: document.getElementById("transcript-clear-filters"),
  transcriptShowing: document.getElementById("transcript-showing"),
  transcriptCopy: document.getElementById("transcript-copy"),
  transcriptScrollBottom: document.getElementById("transcript-scroll-bottom"),
  transcriptList: document.getElementById("transcript-list"),
  transcriptSourceFilter: document.getElementById("transcript-source-filter"),
  transcriptSourceChips: document.getElementById("transcript-source-chips"),
  transcriptTypeFilter: document.getElementById("transcript-type-filter"),
  transcriptTypeChips: document.getElementById("transcript-type-chips"),
  transcriptActiveFilters: document.getElementById("transcript-active-filters"),
  transcriptActiveFiltersGroup: document.getElementById("transcript-active-filters-group"),
  // Logs
  logsSearch: document.getElementById("logs-search"),
  logsClearFilters: document.getElementById("logs-clear-filters"),
  logsShowing: document.getElementById("logs-showing"),
  logsCopy: document.getElementById("logs-copy"),
  logsScrollBottom: document.getElementById("logs-scroll-bottom"),
  logsList: document.getElementById("logs-list"),
  logsLoggerFilter: document.getElementById("logs-logger-filter"),
  logsLoggerChips: document.getElementById("logs-logger-chips"),
  logsEventFilter: document.getElementById("logs-event-filter"),
  logsEventChips: document.getElementById("logs-event-chips"),
  logsActiveFilters: document.getElementById("logs-active-filters"),
  logsActiveFiltersGroup: document.getElementById("logs-active-filters-group"),
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
  elements.transcriptView.classList.toggle("hidden", viewName !== "transcript");
  elements.logsView.classList.toggle("hidden", viewName !== "logs");
  elements.taskView.classList.toggle("hidden", viewName !== "task");
  elements.filesystemView.classList.toggle("hidden", viewName !== "filesystem");

  // Load data if needed
  if (viewName === "transcript" && state.transcriptEntries.length === 0) {
    loadTranscriptFacets();
    loadTranscript();
  } else if (viewName === "logs" && state.filteredLogs.length === 0) {
    loadLogFacets();
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
  state.transcriptEntries = [];
  state.transcriptTotalCount = 0;
  state.transcriptHasMore = false;
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
// TRANSCRIPT
// ============================================================================

function buildTranscriptQueryParams(offset = 0) {
  const params = new URLSearchParams();
  params.set("offset", offset);
  params.set("limit", state.transcriptLimit);

  if (state.transcriptSearch.trim()) {
    params.set("search", state.transcriptSearch.trim());
  }

  if (state.transcriptIncludeSources.size > 0) {
    params.set("source", Array.from(state.transcriptIncludeSources).join(","));
  }
  if (state.transcriptExcludeSources.size > 0) {
    params.set("exclude_source", Array.from(state.transcriptExcludeSources).join(","));
  }

  if (state.transcriptIncludeTypes.size > 0) {
    params.set("entry_type", Array.from(state.transcriptIncludeTypes).join(","));
  }
  if (state.transcriptExcludeTypes.size > 0) {
    params.set("exclude_entry_type", Array.from(state.transcriptExcludeTypes).join(","));
  }

  return params.toString();
}

async function loadTranscriptFacets() {
  try {
    state.transcriptFacets = await fetchJSON("/api/transcript/facets");
    renderTranscriptFilterChips();
  } catch (error) {
    console.warn("Failed to load transcript facets:", error);
  }
}

async function loadTranscript(append = false) {
  if (state.transcriptLoading) return;

  try {
    state.transcriptLoading = true;
    const offset = append ? state.transcriptEntries.length : 0;
    const query = buildTranscriptQueryParams(offset);
    const result = await fetchJSON(`/api/transcript?${query}`);
    const entries = result.entries || [];

    state.transcriptEntries = append ? state.transcriptEntries.concat(entries) : entries;
    state.transcriptTotalCount = result.total || state.transcriptEntries.length;
    state.transcriptHasMore = state.transcriptEntries.length < state.transcriptTotalCount;

    renderTranscript();
    updateTranscriptStats();
  } catch (error) {
    elements.transcriptList.innerHTML = `<p class="muted">Failed to load transcript: ${error.message}</p>`;
  } finally {
    state.transcriptLoading = false;
  }
}

async function loadMoreTranscript() {
  await loadTranscript(true);
}

let transcriptSearchTimeout = null;
function debouncedTranscriptSearch() {
  clearTimeout(transcriptSearchTimeout);
  transcriptSearchTimeout = setTimeout(() => loadTranscript(false), 300);
}

function updateTranscriptStats() {
  let status = `Showing ${state.transcriptEntries.length}`;
  if (state.transcriptHasMore) status += ` of ${state.transcriptTotalCount}`;
  elements.transcriptShowing.textContent = status;
}

function renderTranscriptFilterChips() {
  const sourceFilter = state.transcriptSourceChipFilter.toLowerCase();
  const typeFilter = state.transcriptTypeChipFilter.toLowerCase();

  elements.transcriptSourceChips.innerHTML = "";
  (state.transcriptFacets.sources || [])
    .filter((item) => !sourceFilter || item.name.toLowerCase().includes(sourceFilter))
    .forEach((item) => {
      const chip = createFilterChip(
        item.name,
        item.count,
        state.transcriptIncludeSources.has(item.name),
        state.transcriptExcludeSources.has(item.name),
        (name, include, exclude) => toggleTranscriptSourceFilter(name, include, exclude)
      );
      elements.transcriptSourceChips.appendChild(chip);
    });

  elements.transcriptTypeChips.innerHTML = "";
  (state.transcriptFacets.entry_types || [])
    .filter((item) => !typeFilter || item.name.toLowerCase().includes(typeFilter))
    .forEach((item) => {
      const chip = createFilterChip(
        item.name,
        item.count,
        state.transcriptIncludeTypes.has(item.name),
        state.transcriptExcludeTypes.has(item.name),
        (name, include, exclude) => toggleTranscriptTypeFilter(name, include, exclude)
      );
      elements.transcriptTypeChips.appendChild(chip);
    });

  renderTranscriptActiveFilters();
}

function toggleTranscriptSourceFilter(name, include, exclude) {
  if (include) {
    state.transcriptIncludeSources.add(name);
    state.transcriptExcludeSources.delete(name);
  } else if (exclude) {
    state.transcriptExcludeSources.add(name);
    state.transcriptIncludeSources.delete(name);
  } else {
    state.transcriptIncludeSources.delete(name);
    state.transcriptExcludeSources.delete(name);
  }
  renderTranscriptFilterChips();
  loadTranscript(false);
}

function toggleTranscriptTypeFilter(name, include, exclude) {
  if (include) {
    state.transcriptIncludeTypes.add(name);
    state.transcriptExcludeTypes.delete(name);
  } else if (exclude) {
    state.transcriptExcludeTypes.add(name);
    state.transcriptIncludeTypes.delete(name);
  } else {
    state.transcriptIncludeTypes.delete(name);
    state.transcriptExcludeTypes.delete(name);
  }
  renderTranscriptFilterChips();
  loadTranscript(false);
}

function renderTranscriptActiveFilters() {
  const hasFilters =
    state.transcriptIncludeSources.size > 0 ||
    state.transcriptExcludeSources.size > 0 ||
    state.transcriptIncludeTypes.size > 0 ||
    state.transcriptExcludeTypes.size > 0;

  elements.transcriptActiveFiltersGroup.style.display = hasFilters ? "flex" : "none";
  if (!hasFilters) return;

  elements.transcriptActiveFilters.innerHTML = "";

  state.transcriptIncludeSources.forEach((name) => {
    elements.transcriptActiveFilters.appendChild(
      createActiveFilter("source", name, false, () => toggleTranscriptSourceFilter(name, false, false))
    );
  });

  state.transcriptExcludeSources.forEach((name) => {
    elements.transcriptActiveFilters.appendChild(
      createActiveFilter("source", name, true, () => toggleTranscriptSourceFilter(name, false, false))
    );
  });

  state.transcriptIncludeTypes.forEach((name) => {
    elements.transcriptActiveFilters.appendChild(
      createActiveFilter("entry_type", name, false, () => toggleTranscriptTypeFilter(name, false, false))
    );
  });

  state.transcriptExcludeTypes.forEach((name) => {
    elements.transcriptActiveFilters.appendChild(
      createActiveFilter("entry_type", name, true, () => toggleTranscriptTypeFilter(name, false, false))
    );
  });
}

function formatTranscriptContent(entry) {
  if (entry.content !== null && entry.content !== undefined) {
    if (typeof entry.content === "string") return { kind: "text", value: entry.content };
    return { kind: "json", value: JSON.stringify(entry.content, null, 2) };
  }
  if (entry.parsed !== null && entry.parsed !== undefined) {
    return { kind: "json", value: JSON.stringify(entry.parsed, null, 2) };
  }
  if (entry.raw_json !== null && entry.raw_json !== undefined) {
    return { kind: "json", value: JSON.stringify(entry.raw_json, null, 2) };
  }
  return { kind: "text", value: "" };
}

function renderTranscript() {
  elements.transcriptList.innerHTML = "";

  if (state.transcriptEntries.length === 0) {
    elements.transcriptList.innerHTML = '<div class="logs-empty">No transcript entries match filters</div>';
    return;
  }

  state.transcriptEntries.forEach((entry) => {
    const entryType = entry.entry_type || "unknown";
    const role = entry.role || "";
    const cssClass = role ? `role-${role}` : `type-${entryType}`;

    const container = document.createElement("div");
    container.className = `transcript-entry ${cssClass}`;

    const source = entry.transcript_source || "";
    const seq = entry.sequence_number !== null && entry.sequence_number !== undefined ? String(entry.sequence_number) : "";
    const promptName = entry.prompt_name || "";

    const content = formatTranscriptContent(entry);

    let html = `<div class="transcript-header">`;
    html += `<span class="transcript-type clickable" data-type="${escapeHtml(entryType)}">${escapeHtml(entryType)}</span>`;
    if (entry.timestamp) html += `<span class="transcript-timestamp">${escapeHtml(entry.timestamp)}</span>`;
    if (source) html += `<span class="transcript-source clickable" data-source="${escapeHtml(source)}">${escapeHtml(source)}${seq ? `#${escapeHtml(seq)}` : ""}</span>`;
    if (promptName) html += `<span class="transcript-prompt mono" title="${escapeHtml(promptName)}">${escapeHtml(promptName)}</span>`;
    html += `</div>`;

    if (content.value) {
      if (content.kind === "json") {
        html += `<pre class="transcript-json">${escapeHtml(content.value)}</pre>`;
      } else {
        html += `<div class="transcript-message">${escapeHtml(content.value)}</div>`;
      }
    } else {
      html += `<div class="transcript-message muted">(no content)</div>`;
    }

    const detailsPayload = entry.parsed || entry.raw_json;
    if (detailsPayload) {
      html += `<details class="transcript-details"><summary>Details</summary>`;
      html += `<pre>${escapeHtml(JSON.stringify(detailsPayload, null, 2))}</pre>`;
      html += `</details>`;
    }

    container.innerHTML = html;

    // Inline filtering
    container.querySelectorAll(".transcript-source.clickable").forEach((el) => {
      el.addEventListener("click", (e) => {
        const src = el.dataset.source;
        if (!src) return;
        if (e.shiftKey) toggleTranscriptSourceFilter(src, false, true);
        else toggleTranscriptSourceFilter(src, true, false);
      });
    });

    container.querySelectorAll(".transcript-type.clickable").forEach((el) => {
      el.addEventListener("click", (e) => {
        const typ = el.dataset.type;
        if (!typ) return;
        if (e.shiftKey) toggleTranscriptTypeFilter(typ, false, true);
        else toggleTranscriptTypeFilter(typ, true, false);
      });
    });

    elements.transcriptList.appendChild(container);
  });

  if (state.transcriptHasMore) {
    const loadMoreContainer = document.createElement("div");
    loadMoreContainer.className = "logs-load-more";
    const remaining = state.transcriptTotalCount - state.transcriptEntries.length;
    loadMoreContainer.innerHTML = `
      <button class="ghost logs-load-more-btn" type="button">
        Load more (${remaining} remaining)
      </button>
    `;
    loadMoreContainer.querySelector("button").addEventListener("click", loadMoreTranscript);
    elements.transcriptList.appendChild(loadMoreContainer);
  }
}

// Transcript filter events
elements.transcriptSearch.addEventListener("input", () => {
  state.transcriptSearch = elements.transcriptSearch.value;
  debouncedTranscriptSearch();
});

elements.transcriptSourceFilter.addEventListener("input", () => {
  state.transcriptSourceChipFilter = elements.transcriptSourceFilter.value;
  renderTranscriptFilterChips();
});

elements.transcriptTypeFilter.addEventListener("input", () => {
  state.transcriptTypeChipFilter = elements.transcriptTypeFilter.value;
  renderTranscriptFilterChips();
});

elements.transcriptClearFilters.addEventListener("click", () => {
  state.transcriptSearch = "";
  state.transcriptIncludeSources.clear();
  state.transcriptExcludeSources.clear();
  state.transcriptIncludeTypes.clear();
  state.transcriptExcludeTypes.clear();
  state.transcriptSourceChipFilter = "";
  state.transcriptTypeChipFilter = "";

  elements.transcriptSearch.value = "";
  elements.transcriptSourceFilter.value = "";
  elements.transcriptTypeFilter.value = "";

  renderTranscriptFilterChips();
  loadTranscript(false);
});

elements.transcriptCopy.addEventListener("click", async () => {
  const text = JSON.stringify(state.transcriptEntries, null, 2);
  try {
    await navigator.clipboard.writeText(text);
    showToast("Copied transcript entries", "success");
  } catch {
    showToast("Failed to copy", "error");
  }
});

elements.transcriptScrollBottom.addEventListener("click", () => {
  elements.transcriptList.scrollTop = elements.transcriptList.scrollHeight;
});

// ============================================================================
// TRANSCRIPT
// ============================================================================

function buildTranscriptQueryParams(offset = 0) {
  const params = new URLSearchParams();
  params.set("offset", offset);
  params.set("limit", state.transcriptLimit);

  if (state.transcriptSearch.trim()) {
    params.set("search", state.transcriptSearch.trim());
  }

  if (state.transcriptIncludeSources.size > 0) {
    params.set("source", Array.from(state.transcriptIncludeSources).join(","));
  }
  if (state.transcriptExcludeSources.size > 0) {
    params.set("exclude_source", Array.from(state.transcriptExcludeSources).join(","));
  }
  if (state.transcriptIncludeTypes.size > 0) {
    params.set("entry_type", Array.from(state.transcriptIncludeTypes).join(","));
  }
  if (state.transcriptExcludeTypes.size > 0) {
    params.set("exclude_entry_type", Array.from(state.transcriptExcludeTypes).join(","));
  }

  return params.toString();
}

async function loadTranscriptFacets() {
  try {
    state.transcriptFacets = await fetchJSON("/api/transcript/facets");
    renderTranscriptFilterChips();
  } catch (error) {
    console.warn("Failed to load transcript facets:", error);
  }
}

async function loadTranscript(append = false) {
  if (state.transcriptLoading) return;

  try {
    state.transcriptLoading = true;
    const offset = append ? state.transcriptEntries.length : 0;
    const query = buildTranscriptQueryParams(offset);
    const result = await fetchJSON(`/api/transcript?${query}`);
    const entries = result.entries || [];

    state.transcriptEntries = append ? state.transcriptEntries.concat(entries) : entries;
    state.transcriptTotalCount = result.total || state.transcriptEntries.length;
    state.transcriptHasMore = state.transcriptEntries.length < state.transcriptTotalCount;

    renderTranscript();
    updateTranscriptStats();
  } catch (error) {
    elements.transcriptList.innerHTML = `<p class="muted">Failed to load transcript: ${error.message}</p>`;
  } finally {
    state.transcriptLoading = false;
  }
}

async function loadMoreTranscript() {
  await loadTranscript(true);
}

let transcriptSearchTimeout = null;
function debouncedTranscriptSearch() {
  clearTimeout(transcriptSearchTimeout);
  transcriptSearchTimeout = setTimeout(() => loadTranscript(false), 300);
}

function updateTranscriptStats() {
  let status = `Showing ${state.transcriptEntries.length}`;
  if (state.transcriptHasMore) status += ` of ${state.transcriptTotalCount}`;
  elements.transcriptShowing.textContent = status;
}

function renderTranscriptFilterChips() {
  const sourceFilter = state.transcriptSourceChipFilter.toLowerCase();
  const typeFilter = state.transcriptTypeChipFilter.toLowerCase();

  elements.transcriptSourceChips.innerHTML = "";
  state.transcriptFacets.sources
    .filter((item) => !sourceFilter || item.name.toLowerCase().includes(sourceFilter))
    .forEach((item) => {
      const chip = createFilterChip(
        item.name,
        item.count,
        state.transcriptIncludeSources.has(item.name),
        state.transcriptExcludeSources.has(item.name),
        (name, include, exclude) => {
          toggleTranscriptSourceFilter(name, include, exclude);
        }
      );
      elements.transcriptSourceChips.appendChild(chip);
    });

  elements.transcriptTypeChips.innerHTML = "";
  state.transcriptFacets.entry_types
    .filter((item) => !typeFilter || item.name.toLowerCase().includes(typeFilter))
    .forEach((item) => {
      const chip = createFilterChip(
        item.name,
        item.count,
        state.transcriptIncludeTypes.has(item.name),
        state.transcriptExcludeTypes.has(item.name),
        (name, include, exclude) => {
          toggleTranscriptTypeFilter(name, include, exclude);
        }
      );
      elements.transcriptTypeChips.appendChild(chip);
    });

  renderTranscriptActiveFilters();
}

function toggleTranscriptSourceFilter(name, include, exclude) {
  if (include) {
    state.transcriptIncludeSources.add(name);
    state.transcriptExcludeSources.delete(name);
  } else if (exclude) {
    state.transcriptExcludeSources.add(name);
    state.transcriptIncludeSources.delete(name);
  } else {
    state.transcriptIncludeSources.delete(name);
    state.transcriptExcludeSources.delete(name);
  }
  renderTranscriptFilterChips();
  loadTranscript(false);
}

function toggleTranscriptTypeFilter(name, include, exclude) {
  if (include) {
    state.transcriptIncludeTypes.add(name);
    state.transcriptExcludeTypes.delete(name);
  } else if (exclude) {
    state.transcriptExcludeTypes.add(name);
    state.transcriptIncludeTypes.delete(name);
  } else {
    state.transcriptIncludeTypes.delete(name);
    state.transcriptExcludeTypes.delete(name);
  }
  renderTranscriptFilterChips();
  loadTranscript(false);
}

function renderTranscriptActiveFilters() {
  const hasFilters =
    state.transcriptIncludeSources.size > 0 ||
    state.transcriptExcludeSources.size > 0 ||
    state.transcriptIncludeTypes.size > 0 ||
    state.transcriptExcludeTypes.size > 0;

  elements.transcriptActiveFiltersGroup.style.display = hasFilters ? "flex" : "none";

  if (!hasFilters) return;

  elements.transcriptActiveFilters.innerHTML = "";

  state.transcriptIncludeSources.forEach((name) => {
    elements.transcriptActiveFilters.appendChild(
      createActiveFilter("source", name, false, () => toggleTranscriptSourceFilter(name, false, false))
    );
  });

  state.transcriptExcludeSources.forEach((name) => {
    elements.transcriptActiveFilters.appendChild(
      createActiveFilter("source", name, true, () => toggleTranscriptSourceFilter(name, false, false))
    );
  });

  state.transcriptIncludeTypes.forEach((name) => {
    elements.transcriptActiveFilters.appendChild(
      createActiveFilter("entry_type", name, false, () => toggleTranscriptTypeFilter(name, false, false))
    );
  });

  state.transcriptExcludeTypes.forEach((name) => {
    elements.transcriptActiveFilters.appendChild(
      createActiveFilter("entry_type", name, true, () => toggleTranscriptTypeFilter(name, false, false))
    );
  });
}

function transcriptKind(entry) {
  const entryType = (entry.entry_type || "").toLowerCase();
  if (entryType === "tool_result") return "tool";
  if (entryType) return entryType;
  const role = (entry.role || "").toLowerCase();
  return role || "entry";
}

function renderTranscript() {
  elements.transcriptList.innerHTML = "";

  if (state.transcriptEntries.length === 0) {
    elements.transcriptList.innerHTML = '<div class="transcript-empty">No transcript entries match filters</div>';
    return;
  }

  state.transcriptEntries.forEach((entry, index) => {
    const kind = transcriptKind(entry);
    const card = document.createElement("div");
    card.className = `transcript-entry transcript-${kind}`;
    card.dataset.index = index;

    const roleLabel = entry.role || entry.entry_type || "entry";
    const entryType = entry.entry_type && entry.entry_type !== roleLabel ? entry.entry_type : "";
    const timestamp = entry.timestamp ? escapeHtml(entry.timestamp) : "";
    const promptName = entry.prompt_name ? escapeHtml(entry.prompt_name) : "";
    const source = entry.transcript_source ? escapeHtml(entry.transcript_source) : "";
    const sequence = entry.sequence_number !== null && entry.sequence_number !== undefined ? `#${entry.sequence_number}` : "";
    const toolName = entry.tool_name ? escapeHtml(entry.tool_name) : "";
    const toolUseId = entry.tool_use_id ? escapeHtml(entry.tool_use_id) : "";

    let contentText = "";
    if (typeof entry.content === "string") contentText = entry.content;
    else if (entry.content !== null && entry.content !== undefined) contentText = JSON.stringify(entry.content, null, 2);

    let detailsPayload = null;
    if (entry.parsed) detailsPayload = entry.parsed;
    else if (entry.raw_json) detailsPayload = entry.raw_json;

    let detailsHtml = "";
    if (detailsPayload) {
      const detailsText = typeof detailsPayload === "string" ? detailsPayload : JSON.stringify(detailsPayload, null, 2);
      detailsHtml = `
        <details class="transcript-details">
          <summary>Raw entry</summary>
          <pre>${escapeHtml(detailsText)}</pre>
        </details>
      `;
    }

    card.innerHTML = `
      <div class="transcript-meta">
        <span class="pill pill-quiet transcript-role">${escapeHtml(roleLabel)}</span>
        ${entryType ? `<span class="pill pill-quiet transcript-type">${escapeHtml(entryType)}</span>` : ""}
        ${toolName ? `<span class="pill pill-quiet transcript-tool">tool: ${toolName}</span>` : ""}
        ${toolUseId ? `<span class="pill pill-quiet mono">${toolUseId}</span>` : ""}
        ${promptName ? `<span class="pill pill-quiet">prompt: ${promptName}</span>` : ""}
        ${source ? `<span class="pill pill-quiet">source: ${source}</span>` : ""}
        ${sequence ? `<span class="pill pill-quiet">${sequence}</span>` : ""}
        ${timestamp ? `<span class="transcript-timestamp">${timestamp}</span>` : ""}
      </div>
      <div class="transcript-body">
        <div class="transcript-text">${escapeHtml(contentText || "(no content)")}</div>
        ${detailsHtml}
      </div>
    `;

    elements.transcriptList.appendChild(card);
  });

  if (state.transcriptHasMore) {
    const loadMoreContainer = document.createElement("div");
    loadMoreContainer.className = "transcript-load-more";
    const remaining = state.transcriptTotalCount - state.transcriptEntries.length;
    loadMoreContainer.innerHTML = `
      <button class="ghost transcript-load-more-btn" type="button">
        Load more (${remaining} remaining)
      </button>
    `;
    loadMoreContainer.querySelector("button").addEventListener("click", loadMoreTranscript);
    elements.transcriptList.appendChild(loadMoreContainer);
  }
}

elements.transcriptSearch.addEventListener("input", () => {
  state.transcriptSearch = elements.transcriptSearch.value;
  debouncedTranscriptSearch();
});

elements.transcriptSourceFilter.addEventListener("input", () => {
  state.transcriptSourceChipFilter = elements.transcriptSourceFilter.value;
  renderTranscriptFilterChips();
});

elements.transcriptTypeFilter.addEventListener("input", () => {
  state.transcriptTypeChipFilter = elements.transcriptTypeFilter.value;
  renderTranscriptFilterChips();
});

elements.transcriptClearFilters.addEventListener("click", () => {
  state.transcriptSearch = "";
  state.transcriptIncludeSources.clear();
  state.transcriptExcludeSources.clear();
  state.transcriptIncludeTypes.clear();
  state.transcriptExcludeTypes.clear();
  state.transcriptSourceChipFilter = "";
  state.transcriptTypeChipFilter = "";

  elements.transcriptSearch.value = "";
  elements.transcriptSourceFilter.value = "";
  elements.transcriptTypeFilter.value = "";

  renderTranscriptFilterChips();
  loadTranscript(false);
});

elements.transcriptCopy.addEventListener("click", async () => {
  const text = JSON.stringify(state.transcriptEntries, null, 2);
  try {
    await navigator.clipboard.writeText(text);
    showToast("Copied transcript entries", "success");
  } catch {
    showToast("Failed to copy", "error");
  }
});

elements.transcriptScrollBottom.addEventListener("click", () => {
  elements.transcriptList.scrollTop = elements.transcriptList.scrollHeight;
});

// ============================================================================
// LOGS
// ============================================================================

function buildLogsQueryParams(offset = 0) {
  const params = new URLSearchParams();
  params.set("offset", offset);
  params.set("limit", state.logsLimit);

  // Level filter
  if (state.logsLevels.size > 0 && state.logsLevels.size < 4) {
    params.set("level", Array.from(state.logsLevels).join(","));
  }

  // Search filter (server-side)
  if (state.logsSearch.trim()) {
    params.set("search", state.logsSearch.trim());
  }

  // Logger filters
  if (state.logsIncludeLoggers.size > 0) {
    params.set("logger", Array.from(state.logsIncludeLoggers).join(","));
  }
  if (state.logsExcludeLoggers.size > 0) {
    params.set("exclude_logger", Array.from(state.logsExcludeLoggers).join(","));
  }

  // Event filters
  if (state.logsIncludeEvents.size > 0) {
    params.set("event", Array.from(state.logsIncludeEvents).join(","));
  }
  if (state.logsExcludeEvents.size > 0) {
    params.set("exclude_event", Array.from(state.logsExcludeEvents).join(","));
  }

  return params.toString();
}

async function loadLogFacets() {
  try {
    state.logsFacets = await fetchJSON("/api/logs/facets");
    renderLogFilterChips();
  } catch (error) {
    console.warn("Failed to load log facets:", error);
  }
}

async function loadLogs(append = false) {
  if (state.logsLoading) return;

  try {
    state.logsLoading = true;
    const offset = append ? state.filteredLogs.length : 0;
    const query = buildLogsQueryParams(offset);
    const result = await fetchJSON(`/api/logs?${query}`);
    const entries = result.entries || [];

    state.filteredLogs = append ? state.filteredLogs.concat(entries) : entries;
    state.logsTotalCount = result.total || state.filteredLogs.length;
    state.logsHasMore = state.filteredLogs.length < state.logsTotalCount;

    renderLogs();
    updateLogsStats();
  } catch (error) {
    elements.logsList.innerHTML = `<p class="muted">Failed to load logs: ${error.message}</p>`;
  } finally {
    state.logsLoading = false;
  }
}

async function loadMoreLogs() {
  await loadLogs(true);
}

// Debounced search to avoid too many API calls
let logsSearchTimeout = null;
function debouncedLogsSearch() {
  clearTimeout(logsSearchTimeout);
  logsSearchTimeout = setTimeout(() => loadLogs(false), 300);
}

function updateLogsStats() {
  let status = `Showing ${state.filteredLogs.length}`;
  if (state.logsHasMore) status += ` of ${state.logsTotalCount}`;
  elements.logsShowing.textContent = status;
}

function renderLogFilterChips() {
  const loggerFilter = state.logsLoggerChipFilter.toLowerCase();
  const eventFilter = state.logsEventChipFilter.toLowerCase();

  // Render logger chips
  elements.logsLoggerChips.innerHTML = "";
  state.logsFacets.loggers
    .filter((item) => !loggerFilter || item.name.toLowerCase().includes(loggerFilter))
    .forEach((item) => {
      const chip = createFilterChip(
        item.name,
        item.count,
        state.logsIncludeLoggers.has(item.name),
        state.logsExcludeLoggers.has(item.name),
        (name, include, exclude) => {
          toggleLoggerFilter(name, include, exclude);
        }
      );
      elements.logsLoggerChips.appendChild(chip);
    });

  // Render event chips
  elements.logsEventChips.innerHTML = "";
  state.logsFacets.events
    .filter((item) => !eventFilter || item.name.toLowerCase().includes(eventFilter))
    .forEach((item) => {
      const chip = createFilterChip(
        item.name,
        item.count,
        state.logsIncludeEvents.has(item.name),
        state.logsExcludeEvents.has(item.name),
        (name, include, exclude) => {
          toggleEventFilter(name, include, exclude);
        }
      );
      elements.logsEventChips.appendChild(chip);
    });

  renderActiveFilters();
}

function createFilterChip(name, count, isIncluded, isExcluded, onToggle) {
  const chip = document.createElement("span");
  chip.className = "filter-chip";
  if (isIncluded) chip.classList.add("included");
  if (isExcluded) chip.classList.add("excluded");

  const displayName = name.split(".").pop() || name;
  let prefix = "";
  if (isIncluded) prefix = "+ ";
  if (isExcluded) prefix = "− ";
  chip.innerHTML = `${prefix}${escapeHtml(displayName)} <span class="chip-count">${count}</span>`;
  chip.title = `${name}\nClick: show only | Shift+click: hide`;

  chip.addEventListener("click", (e) => {
    e.preventDefault();
    if (e.shiftKey) {
      // Shift+click to exclude
      onToggle(name, false, !isExcluded);
    } else {
      // Regular click to include
      onToggle(name, !isIncluded, false);
    }
  });

  return chip;
}

function toggleLoggerFilter(name, include, exclude) {
  if (include) {
    state.logsIncludeLoggers.add(name);
    state.logsExcludeLoggers.delete(name);
  } else if (exclude) {
    state.logsExcludeLoggers.add(name);
    state.logsIncludeLoggers.delete(name);
  } else {
    state.logsIncludeLoggers.delete(name);
    state.logsExcludeLoggers.delete(name);
  }
  renderLogFilterChips();
  loadLogs(false);
}

function toggleEventFilter(name, include, exclude) {
  if (include) {
    state.logsIncludeEvents.add(name);
    state.logsExcludeEvents.delete(name);
  } else if (exclude) {
    state.logsExcludeEvents.add(name);
    state.logsIncludeEvents.delete(name);
  } else {
    state.logsIncludeEvents.delete(name);
    state.logsExcludeEvents.delete(name);
  }
  renderLogFilterChips();
  loadLogs(false);
}

function renderActiveFilters() {
  const hasFilters =
    state.logsIncludeLoggers.size > 0 ||
    state.logsExcludeLoggers.size > 0 ||
    state.logsIncludeEvents.size > 0 ||
    state.logsExcludeEvents.size > 0;

  elements.logsActiveFiltersGroup.style.display = hasFilters ? "flex" : "none";

  if (!hasFilters) return;

  elements.logsActiveFilters.innerHTML = "";

  // Include loggers
  state.logsIncludeLoggers.forEach((name) => {
    elements.logsActiveFilters.appendChild(
      createActiveFilter("logger", name, false, () => toggleLoggerFilter(name, false, false))
    );
  });

  // Exclude loggers
  state.logsExcludeLoggers.forEach((name) => {
    elements.logsActiveFilters.appendChild(
      createActiveFilter("logger", name, true, () => toggleLoggerFilter(name, false, false))
    );
  });

  // Include events
  state.logsIncludeEvents.forEach((name) => {
    elements.logsActiveFilters.appendChild(
      createActiveFilter("event", name, false, () => toggleEventFilter(name, false, false))
    );
  });

  // Exclude events
  state.logsExcludeEvents.forEach((name) => {
    elements.logsActiveFilters.appendChild(
      createActiveFilter("event", name, true, () => toggleEventFilter(name, false, false))
    );
  });
}

function createActiveFilter(type, name, isExclude, onRemove) {
  const filter = document.createElement("span");
  filter.className = `active-filter${isExclude ? " exclude" : ""}`;

  const displayName = name.split(".").pop() || name;
  const prefix = isExclude ? "−" : "+";
  filter.innerHTML = `${prefix}${escapeHtml(displayName)} <span class="remove-filter">×</span>`;
  filter.title = `${type}: ${name}`;

  filter.querySelector(".remove-filter").addEventListener("click", onRemove);
  return filter;
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
      const loggerShort = log.logger.split(".").slice(-2).join(".");
      html += `<span class="log-logger clickable" data-logger="${escapeHtml(log.logger)}" title="${escapeHtml(log.logger)}">${escapeHtml(loggerShort)}</span>`;
    }
    if (log.event) {
      html += `<span class="log-event-name clickable" data-event="${escapeHtml(log.event)}">${escapeHtml(log.event)}</span>`;
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

    // Add click handlers for inline filtering
    entry.querySelectorAll(".log-logger.clickable").forEach((el) => {
      el.addEventListener("click", (e) => {
        const logger = el.dataset.logger;
        if (e.shiftKey) {
          toggleLoggerFilter(logger, false, true);
        } else {
          toggleLoggerFilter(logger, true, false);
        }
      });
    });

    entry.querySelectorAll(".log-event-name.clickable").forEach((el) => {
      el.addEventListener("click", (e) => {
        const event = el.dataset.event;
        if (e.shiftKey) {
          toggleEventFilter(event, false, true);
        } else {
          toggleEventFilter(event, true, false);
        }
      });
    });

    elements.logsList.appendChild(entry);
  });

  // Add "Load more" button if there are more logs
  if (state.logsHasMore) {
    const loadMoreContainer = document.createElement("div");
    loadMoreContainer.className = "logs-load-more";
    const remaining = state.logsTotalCount - state.filteredLogs.length;
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
  debouncedLogsSearch();
});

elements.logsLoggerFilter.addEventListener("input", () => {
  state.logsLoggerChipFilter = elements.logsLoggerFilter.value;
  renderLogFilterChips();
});

elements.logsEventFilter.addEventListener("input", () => {
  state.logsEventChipFilter = elements.logsEventChipFilter.value;
  renderLogFilterChips();
});

// Level checkboxes
document.querySelectorAll(".level-checkbox input").forEach((checkbox) => {
  checkbox.addEventListener("change", () => {
    if (checkbox.checked) {
      state.logsLevels.add(checkbox.value);
    } else {
      state.logsLevels.delete(checkbox.value);
    }
    loadLogs(false);
  });
});

elements.logsClearFilters.addEventListener("click", () => {
  state.logsSearch = "";
  state.logsLevels = new Set(["DEBUG", "INFO", "WARNING", "ERROR"]);
  state.logsIncludeLoggers.clear();
  state.logsExcludeLoggers.clear();
  state.logsIncludeEvents.clear();
  state.logsExcludeEvents.clear();
  state.logsLoggerChipFilter = "";
  state.logsEventChipFilter = "";

  elements.logsSearch.value = "";
  elements.logsLoggerFilter.value = "";
  elements.logsEventFilter.value = "";
  document.querySelectorAll(".level-checkbox input").forEach((cb) => (cb.checked = true));

  renderLogFilterChips();
  loadLogs(false);
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
  if (e.key >= "1" && e.key <= "5") {
    e.preventDefault();
    const views = ["sessions", "transcript", "logs", "task", "filesystem"];
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
  else if (state.activeView === "transcript") elements.transcriptSearch.focus();
  else if (state.activeView === "logs") elements.logsSearch.focus();
  else if (state.activeView === "filesystem") elements.filesystemFilter.focus();
}

function navigateNext() {
  if (state.activeView === "sessions") focusItem(state.focusedItemIndex + 1);
  else if (state.activeView === "transcript") scrollTranscriptBy(1);
  else if (state.activeView === "logs") scrollLogsBy(1);
}

function navigatePrev() {
  if (state.activeView === "sessions") focusItem(state.focusedItemIndex - 1);
  else if (state.activeView === "transcript") scrollTranscriptBy(-1);
  else if (state.activeView === "logs") scrollLogsBy(-1);
}

function scrollTranscriptBy(delta) {
  const entries = elements.transcriptList.querySelectorAll(".transcript-entry");
  if (entries.length === 0) return;
  const scrollHeight = entries[0].offsetHeight;
  elements.transcriptList.scrollBy({ top: scrollHeight * delta, behavior: "smooth" });
}

function scrollLogsBy(delta) {
  const entries = elements.logsList.querySelectorAll(".log-entry");
  if (entries.length === 0) return;
  const scrollHeight = entries[0].offsetHeight;
  elements.logsList.scrollBy({ top: scrollHeight * delta, behavior: "smooth" });
}

function scrollTranscriptBy(delta) {
  const entries = elements.transcriptList.querySelectorAll(".transcript-entry");
  if (entries.length === 0) return;
  const scrollHeight = entries[0].offsetHeight;
  elements.transcriptList.scrollBy({ top: scrollHeight * delta, behavior: "smooth" });
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
