// ============================================================================
// IMPORTS
// ============================================================================

import {
  escapeHtml,
  formatBytes,
  getMarkdownPayload,
  isObject,
  isSimpleArray,
  pathKey,
  valueType,
} from "./lib.js";

// ============================================================================
// VIRTUAL SCROLLER
// ============================================================================

/**
 * VirtualScroller - Implements windowed rendering with garbage collection
 * Only renders items visible in viewport plus a buffer zone.
 * Items scrolling out of the buffer are removed from DOM to save memory.
 */
class VirtualScroller {
  constructor(options) {
    this.container = options.container;
    this.estimatedItemHeight = options.estimatedItemHeight || 100;
    this.bufferSize = options.bufferSize || 10; // Items to keep above/below viewport
    this.renderItem = options.renderItem;
    this.onLoadMore = options.onLoadMore || null;
    this.onLoadError = options.onLoadError || null; // Callback for load errors
    this.loadMoreThreshold = options.loadMoreThreshold || 3; // Items from bottom to trigger load

    this.items = [];
    this.totalCount = 0;
    this.hasMore = false;
    this.isLoading = false;

    // Track rendered items
    this.renderedItems = new Map(); // index -> DOM element
    this.itemHeights = new Map(); // index -> measured height

    // Spacer elements
    this.topSpacer = document.createElement("div");
    this.topSpacer.className = "virtual-spacer virtual-spacer-top";
    this.bottomSpacer = document.createElement("div");
    this.bottomSpacer.className = "virtual-spacer virtual-spacer-bottom";
    this.loadMoreSentinel = document.createElement("div");
    this.loadMoreSentinel.className = "virtual-load-sentinel";

    // Intersection observer for infinite scroll
    this.loadMoreObserver = null;
    this.setupLoadMoreObserver();

    // Scroll handler with debounce
    this.scrollHandler = this.debounce(() => this.updateVisibleRange(), 16);
    this.container.addEventListener("scroll", this.scrollHandler);

    // Resize observer for container size changes
    this.resizeObserver = new ResizeObserver(() => this.updateVisibleRange());
    this.resizeObserver.observe(this.container);

    // Initial state
    this.visibleStart = 0;
    this.visibleEnd = 0;
    this.scrollTimeout = null; // Track debounce timeout for cleanup
  }

  debounce(fn, delay) {
    return (...args) => {
      clearTimeout(this.scrollTimeout);
      this.scrollTimeout = setTimeout(() => fn(...args), delay);
    };
  }

  setupLoadMoreObserver() {
    this.loadMoreObserver = new IntersectionObserver(
      (entries) => {
        // Check if any entry is intersecting (avoid race condition with multiple entries)
        const anyIntersecting = entries.some((e) => e.isIntersecting);
        if (anyIntersecting && this.hasMore && !this.isLoading && this.onLoadMore) {
          this.isLoading = true;
          this.onLoadMore()
            .catch((error) => {
              if (this.onLoadError) {
                this.onLoadError(error);
              } else {
                console.error("Failed to load more items:", error);
              }
            })
            .finally(() => {
              this.isLoading = false;
            });
        }
      },
      { root: this.container, rootMargin: "200px" }
    );
  }

  setData(items, totalCount, hasMore) {
    this.items = items;
    this.totalCount = totalCount;
    this.hasMore = hasMore;
    // Clear stale height measurements when data changes
    this.itemHeights.clear();
    this.render();
  }

  appendData(newItems, totalCount, hasMore) {
    const wasObserving = this.hasMore;
    this.items = this.items.concat(newItems);
    this.totalCount = totalCount;
    this.hasMore = hasMore;
    this.updateVisibleRange();
    this.updateSpacers();

    // If hasMore changed from false to true, re-observe the sentinel
    if (!wasObserving && this.hasMore) {
      this.loadMoreObserver.observe(this.loadMoreSentinel);
    }
  }

  reset() {
    // Unobserve sentinel before removing it from DOM to prevent memory leak
    this.loadMoreObserver.unobserve(this.loadMoreSentinel);

    this.items = [];
    this.totalCount = 0;
    this.hasMore = false;
    this.renderedItems.clear();
    this.itemHeights.clear();
    this.container.innerHTML = "";
    this.visibleStart = 0;
    this.visibleEnd = 0;
  }

  getItemHeight(index) {
    return this.itemHeights.get(index) || this.estimatedItemHeight;
  }

  getTotalHeight() {
    let total = 0;
    for (let i = 0; i < this.items.length; i++) {
      total += this.getItemHeight(i);
    }
    return total;
  }

  getOffsetForIndex(index) {
    let offset = 0;
    for (let i = 0; i < index && i < this.items.length; i++) {
      offset += this.getItemHeight(i);
    }
    return offset;
  }

  getIndexAtOffset(offset) {
    let current = 0;
    for (let i = 0; i < this.items.length; i++) {
      const height = this.getItemHeight(i);
      if (current + height > offset) {
        return i;
      }
      current += height;
    }
    return Math.max(0, this.items.length - 1);
  }

  calculateVisibleRange() {
    const scrollTop = this.container.scrollTop;
    const viewportHeight = this.container.clientHeight;

    const startIndex = Math.max(0, this.getIndexAtOffset(scrollTop) - this.bufferSize);
    const endIndex = Math.min(
      this.items.length,
      this.getIndexAtOffset(scrollTop + viewportHeight) + this.bufferSize + 1
    );

    return { startIndex, endIndex };
  }

  updateSpacers() {
    const topHeight = this.getOffsetForIndex(this.visibleStart);
    const bottomHeight = this.getTotalHeight() - this.getOffsetForIndex(this.visibleEnd);

    this.topSpacer.style.height = `${topHeight}px`;
    this.bottomSpacer.style.height = `${Math.max(0, bottomHeight)}px`;
  }

  measureRenderedItems() {
    this.renderedItems.forEach((element, index) => {
      const rect = element.getBoundingClientRect();
      if (rect.height > 0) {
        this.itemHeights.set(index, rect.height);
      }
    });
  }

  updateVisibleRange() {
    if (this.items.length === 0) {
      return;
    }

    // Measure current items before updating
    this.measureRenderedItems();

    const { startIndex, endIndex } = this.calculateVisibleRange();

    // Only update if range changed
    if (startIndex === this.visibleStart && endIndex === this.visibleEnd) {
      return;
    }

    // Garbage collection: remove items outside new range
    this.renderedItems.forEach((element, index) => {
      if (index < startIndex || index >= endIndex) {
        element.remove();
        this.renderedItems.delete(index);
      }
    });

    // Add new items in range
    for (let i = startIndex; i < endIndex; i++) {
      if (!this.renderedItems.has(i) && i < this.items.length) {
        const element = this.renderItem(this.items[i], i);
        element.dataset.virtualIndex = i;
        this.renderedItems.set(i, element);
      }
    }

    // Update visible range
    this.visibleStart = startIndex;
    this.visibleEnd = endIndex;

    // Reorder elements in DOM
    this.reorderElements();

    // Update spacers
    this.updateSpacers();
  }

  reorderElements() {
    // Get sorted indices
    const indices = Array.from(this.renderedItems.keys()).sort((a, b) => a - b);

    // Remove sentinel observer temporarily
    this.loadMoreObserver.unobserve(this.loadMoreSentinel);

    try {
      // Clear and rebuild container content
      const fragment = document.createDocumentFragment();
      fragment.appendChild(this.topSpacer);

      indices.forEach((index) => {
        fragment.appendChild(this.renderedItems.get(index));
      });

      fragment.appendChild(this.bottomSpacer);
      fragment.appendChild(this.loadMoreSentinel);

      // Use replaceChildren for atomic replacement (avoids memory leak)
      this.container.replaceChildren(fragment);
    } finally {
      // Re-observe sentinel (always reconnect, even on error)
      if (this.hasMore) {
        this.loadMoreObserver.observe(this.loadMoreSentinel);
      }
    }
  }

  render() {
    // Unobserve sentinel before clearing (it may have been observed from previous render)
    this.loadMoreObserver.unobserve(this.loadMoreSentinel);

    this.container.innerHTML = "";
    this.renderedItems.clear();

    if (this.items.length === 0) {
      return;
    }

    // Calculate initial visible range
    const { startIndex, endIndex } = this.calculateVisibleRange();
    this.visibleStart = startIndex;
    this.visibleEnd = endIndex;

    // Build DOM
    const fragment = document.createDocumentFragment();
    fragment.appendChild(this.topSpacer);

    for (let i = startIndex; i < endIndex && i < this.items.length; i++) {
      const element = this.renderItem(this.items[i], i);
      element.dataset.virtualIndex = i;
      this.renderedItems.set(i, element);
      fragment.appendChild(element);
    }

    fragment.appendChild(this.bottomSpacer);
    fragment.appendChild(this.loadMoreSentinel);

    this.container.appendChild(fragment);

    // Measure items synchronously to avoid scroll jumps
    // (accessing offsetHeight forces a synchronous layout)
    this.measureRenderedItems();

    // Update spacers with accurate measurements
    this.updateSpacers();

    // Observe sentinel for infinite scroll
    if (this.hasMore) {
      this.loadMoreObserver.observe(this.loadMoreSentinel);
    }
  }

  scrollToBottom() {
    this.container.scrollTop = this.container.scrollHeight;
  }

  scrollToIndex(index) {
    const offset = this.getOffsetForIndex(index);
    this.container.scrollTop = offset;
  }

  destroy() {
    // Clear pending debounced scroll handler to prevent post-destroy execution
    clearTimeout(this.scrollTimeout);

    this.container.removeEventListener("scroll", this.scrollHandler);
    this.resizeObserver.disconnect();
    this.loadMoreObserver.disconnect();

    // Explicitly remove rendered elements from DOM to prevent memory leaks
    this.renderedItems.forEach((element) => element.remove());
    this.renderedItems.clear();
    this.itemHeights.clear();

    // Clear container content
    this.container.innerHTML = "";
  }
}

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
  transcriptRawEntries: [], // Raw entries from API
  transcriptEntries: [], // Preprocessed entries (tool calls combined with results)
  transcriptFacets: { sources: [], entry_types: [] },
  transcriptSearch: "",
  transcriptSourceChipFilter: "",
  transcriptTypeChipFilter: "",
  transcriptIncludeSources: new Set(),
  transcriptExcludeSources: new Set(),
  transcriptIncludeTypes: new Set(),
  transcriptExcludeTypes: new Set(),
  transcriptLimit: 50,
  transcriptTotalCount: 0,
  transcriptHasMore: false,
  transcriptLoading: false,
  transcriptRequestId: 0, // Tracks current request to ignore stale responses
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
  logsLimit: 50,
  logsTotalCount: 0,
  logsHasMore: false,
  logsLoading: false,
  logsRequestId: 0, // Tracks current request to ignore stale responses
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
  // Environment state
  environmentData: null,
  hasEnvironmentData: false,
  // Shortcuts overlay
  shortcutsOpen: false,
  // Zoom modal state
  zoomOpen: false,
  zoomType: null, // 'transcript' or 'log'
  zoomIndex: -1,
  zoomEntry: null,
  // Virtual scrollers (initialized after DOM ready)
  logsScroller: null,
  transcriptScroller: null,
};

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
  // Filesystem
  filesystemFilter: document.getElementById("filesystem-filter"),
  filesystemList: document.getElementById("filesystem-list"),
  filesystemEmptyState: document.getElementById("filesystem-empty-state"),
  filesystemNoSnapshot: document.getElementById("filesystem-no-snapshot"),
  filesystemContent: document.getElementById("filesystem-content"),
  filesystemCurrentPath: document.getElementById("filesystem-current-path"),
  filesystemViewer: document.getElementById("filesystem-viewer"),
  filesystemCopy: document.getElementById("filesystem-copy"),
  // Environment
  environmentView: document.getElementById("environment-view"),
  environmentEmptyState: document.getElementById("environment-empty-state"),
  environmentContent: document.getElementById("environment-content"),
  environmentData: document.getElementById("environment-data"),
  environmentCopy: document.getElementById("environment-copy"),
  // Overlays
  loadingOverlay: document.getElementById("loading-overlay"),
  toastContainer: document.getElementById("toast-container"),
  shortcutsOverlay: document.getElementById("shortcuts-overlay"),
  shortcutsClose: document.getElementById("shortcuts-close"),
  // Zoom modal
  zoomModal: document.getElementById("zoom-modal"),
  zoomModalType: document.getElementById("zoom-modal-type"),
  zoomModalTimestamp: document.getElementById("zoom-modal-timestamp"),
  zoomModalSource: document.getElementById("zoom-modal-source"),
  zoomContent: document.getElementById("zoom-content"),
  zoomDetails: document.getElementById("zoom-details"),
  zoomClose: document.getElementById("zoom-close"),
  zoomCopy: document.getElementById("zoom-copy"),
  zoomPrev: document.getElementById("zoom-prev"),
  zoomNext: document.getElementById("zoom-next"),
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

const VIEW_ELEMENTS = {
  sessions: "sessionsView",
  transcript: "transcriptView",
  logs: "logsView",
  filesystem: "filesystemView",
  environment: "environmentView",
};

function updateViewVisibility(viewName) {
  document.querySelectorAll(".main-tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.view === viewName);
  });
  for (const [name, elementKey] of Object.entries(VIEW_ELEMENTS)) {
    elements[elementKey].classList.toggle("hidden", viewName !== name);
  }
}

function initScrollerIfNeeded(viewName) {
  if (viewName === "transcript" && !state.transcriptScroller) {
    initTranscriptVirtualScroller();
  }
  if (viewName === "logs" && !state.logsScroller) {
    initLogsVirtualScroller();
  }
}

function loadViewDataIfNeeded(viewName) {
  if (viewName === "transcript" && state.transcriptEntries.length === 0) {
    loadTranscriptFacets();
    loadTranscript();
  }
  if (viewName === "logs" && state.filteredLogs.length === 0) {
    loadLogFacets();
    loadLogs();
  }
  if (viewName === "filesystem" && state.allFiles.length === 0) {
    loadFilesystem();
  }
  if (viewName === "environment" && state.environmentData === null) {
    loadEnvironment();
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
    if (entry.selected) {
      option.selected = true;
    }
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
  if (e.target.value) {
    switchBundle(e.target.value);
  }
});

function resetViewState() {
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

  // Destroy virtual scrollers to release resources (they'll be re-initialized on view switch)
  if (state.transcriptScroller) {
    state.transcriptScroller.destroy();
    state.transcriptScroller = null;
  }
  if (state.logsScroller) {
    state.logsScroller.destroy();
    state.logsScroller = null;
  }
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
  if (!entry.count) {
    return false;
  }
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
      li.className = `slice-item${entry.slice_type === state.selectedSlice ? " active" : ""}`;
      li.innerHTML = `
        <div class="slice-title">${escapeHtml(entry.display_name || entry.slice_type)}</div>
        <div class="slice-subtitle">${escapeHtml(entry.item_display_name || entry.item_type)} · ${entry.count} items</div>
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
  const text = JSON.stringify(
    getFilteredItems().map((e) => e.item),
    null,
    2
  );
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

function processTranscriptEntries(entries, append) {
  if (append) {
    const rawEntries = [...state.transcriptRawEntries, ...entries];
    state.transcriptRawEntries = rawEntries;
    return preprocessTranscriptEntries(rawEntries);
  }
  state.transcriptRawEntries = entries;
  return preprocessTranscriptEntries(entries);
}

function updateTranscriptDisplay() {
  if (state.transcriptScroller) {
    state.transcriptScroller.setData(
      state.transcriptEntries,
      state.transcriptTotalCount,
      state.transcriptHasMore
    );
    renderTranscriptEmptyState();
  } else {
    renderTranscript();
  }
  updateTranscriptStats();
}

function showTranscriptError(message) {
  elements.transcriptList.innerHTML = `<p class="muted">Failed to load transcript: ${message}</p>`;
}

function applyTranscriptResult(result, append) {
  state.transcriptEntries = processTranscriptEntries(result.entries || [], append);
  state.transcriptTotalCount = result.total || state.transcriptRawEntries.length;
  state.transcriptHasMore = state.transcriptRawEntries.length < state.transcriptTotalCount;
  updateTranscriptDisplay();
}

async function loadTranscript(append = false) {
  const requestId = ++state.transcriptRequestId;
  const isCurrentRequest = () => requestId === state.transcriptRequestId;
  try {
    state.transcriptLoading = true;
    const offset = append ? state.transcriptEntries.length : 0;
    const result = await fetchJSON(`/api/transcript?${buildTranscriptQueryParams(offset)}`);
    if (isCurrentRequest()) {
      applyTranscriptResult(result, append);
    }
  } catch (error) {
    if (isCurrentRequest()) {
      showTranscriptError(error.message);
    }
  } finally {
    if (isCurrentRequest()) {
      state.transcriptLoading = false;
    }
  }
}

function loadMoreTranscript() {
  return loadTranscript(true);
}

let transcriptSearchTimeout = null;
function debouncedTranscriptSearch() {
  clearTimeout(transcriptSearchTimeout);
  transcriptSearchTimeout = setTimeout(() => loadTranscript(false), 300);
}

function updateTranscriptStats() {
  let status = `Showing ${state.transcriptEntries.length}`;
  if (state.transcriptHasMore) {
    status += ` of ${state.transcriptTotalCount}`;
  }
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
  if (!hasFilters) {
    return;
  }

  elements.transcriptActiveFilters.innerHTML = "";

  state.transcriptIncludeSources.forEach((name) => {
    elements.transcriptActiveFilters.appendChild(
      createActiveFilter("source", name, false, () =>
        toggleTranscriptSourceFilter(name, false, false)
      )
    );
  });

  state.transcriptExcludeSources.forEach((name) => {
    elements.transcriptActiveFilters.appendChild(
      createActiveFilter("source", name, true, () =>
        toggleTranscriptSourceFilter(name, false, false)
      )
    );
  });

  state.transcriptIncludeTypes.forEach((name) => {
    elements.transcriptActiveFilters.appendChild(
      createActiveFilter("entry_type", name, false, () =>
        toggleTranscriptTypeFilter(name, false, false)
      )
    );
  });

  state.transcriptExcludeTypes.forEach((name) => {
    elements.transcriptActiveFilters.appendChild(
      createActiveFilter("entry_type", name, true, () =>
        toggleTranscriptTypeFilter(name, false, false)
      )
    );
  });
}

function formatTranscriptContent(entry) {
  if (entry.content !== null && entry.content !== undefined) {
    if (typeof entry.content === "string") {
      return { kind: "text", value: entry.content };
    }
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

const ZOOM_BUTTON_SVG = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 3 21 3 21 9"></polyline><polyline points="9 21 3 21 3 15"></polyline><line x1="21" y1="3" x2="14" y2="10"></line><line x1="3" y1="21" x2="10" y2="14"></line></svg>`;

function createTranscriptTypeHtml(entry, entryType) {
  if (entry.isComposite) {
    let html = `<span class="transcript-type clickable" data-type="${escapeHtml(entryType)}">tool call + result</span>`;
    if (entry.tool_name) {
      html += `<span class="transcript-tool-name">${escapeHtml(entry.tool_name)}</span>`;
    }
    return html;
  }
  return `<span class="transcript-type clickable" data-type="${escapeHtml(entryType)}">${escapeHtml(entryType)}</span>`;
}

function createTranscriptMetadataHtml(entry) {
  let html = "";
  if (entry.timestamp) {
    html += `<span class="transcript-timestamp">${escapeHtml(entry.timestamp)}</span>`;
  }
  const source = entry.transcript_source || "";
  if (source) {
    const seq =
      entry.sequence_number != null ? `#${escapeHtml(String(entry.sequence_number))}` : "";
    html += `<span class="transcript-source clickable" data-source="${escapeHtml(source)}">${escapeHtml(source)}${seq}</span>`;
  }
  if (entry.prompt_name) {
    html += `<span class="transcript-prompt mono" title="${escapeHtml(entry.prompt_name)}">${escapeHtml(entry.prompt_name)}</span>`;
  }
  return html;
}

function createContentHtml(content, emptyMessage = "(no content)") {
  if (!content.value) {
    return `<div class="transcript-message muted">${emptyMessage}</div>`;
  }
  if (content.kind === "json") {
    return `<pre class="transcript-json">${escapeHtml(content.value)}</pre>`;
  }
  return `<div class="transcript-message">${escapeHtml(content.value)}</div>`;
}

function createToolResultHtml(toolResult) {
  if (!toolResult) {
    return "";
  }
  const resultContent = formatTranscriptContent(toolResult);
  return `<div class="transcript-result-divider">↓ result</div>${createContentHtml(resultContent, "(no result)")}`;
}

function createDetailsHtml(entry) {
  const payload = entry.parsed || entry.raw_json;
  if (!payload) {
    return "";
  }
  return `<details class="transcript-details"><summary>Details</summary><pre>${escapeHtml(JSON.stringify(payload, null, 2))}</pre></details>`;
}

/**
 * Creates a single transcript entry DOM element.
 * Used by both virtual scroller and fallback rendering.
 */
function createTranscriptEntryElement(entry, index) {
  const entryType = entry.entry_type || "unknown";
  const role = entry.role || "";
  const cssClass = role ? `role-${role}` : `type-${entryType}`;

  const container = document.createElement("div");
  container.className = `transcript-entry ${cssClass} compact${entry.isComposite ? " combined" : ""}`;
  container.dataset.entryIndex = index;

  const headerHtml = `<div class="transcript-header"><button class="zoom-button" type="button" data-zoom-index="${index}" title="Expand entry" aria-label="Expand entry">${ZOOM_BUTTON_SVG}</button>${createTranscriptTypeHtml(entry, entryType)}${createTranscriptMetadataHtml(entry)}</div>`;

  const content = formatTranscriptContent(entry);
  container.innerHTML =
    headerHtml +
    createContentHtml(content) +
    (entry.isComposite ? createToolResultHtml(entry.toolResult) : "") +
    createDetailsHtml(entry);

  return container;
}

/**
 * Renders empty state for transcript when using virtual scroller.
 */
function renderTranscriptEmptyState() {
  if (state.transcriptEntries.length === 0) {
    // Show empty state inside the virtual scroller container
    if (state.transcriptScroller) {
      state.transcriptScroller.reset();
    }
    elements.transcriptList.innerHTML =
      '<div class="logs-empty">No transcript entries match filters</div>';
  }
}

/**
 * Initializes the transcript virtual scroller.
 * Called once after DOM is ready.
 */
function initTranscriptVirtualScroller() {
  if (state.transcriptScroller) {
    state.transcriptScroller.destroy();
  }

  state.transcriptScroller = new VirtualScroller({
    container: elements.transcriptList,
    estimatedItemHeight: 120, // Transcript entries are typically taller
    bufferSize: 15,
    renderItem: createTranscriptEntryElement,
    onLoadMore: loadMoreTranscript,
    onLoadError: (error) =>
      showToast(`Failed to load more transcript entries: ${error.message}`, "error"),
  });
}

/**
 * Fallback render function for transcript (without virtual scrolling).
 * Used when virtual scroller is not initialized.
 */
function renderTranscript() {
  // If virtual scroller is available, use it instead
  if (state.transcriptScroller) {
    if (state.transcriptEntries.length === 0) {
      renderTranscriptEmptyState();
    } else {
      state.transcriptScroller.setData(
        state.transcriptEntries,
        state.transcriptTotalCount,
        state.transcriptHasMore
      );
    }
    return;
  }

  // Fallback: render all items (original behavior)
  elements.transcriptList.innerHTML = "";

  if (state.transcriptEntries.length === 0) {
    elements.transcriptList.innerHTML =
      '<div class="logs-empty">No transcript entries match filters</div>';
    return;
  }

  state.transcriptEntries.forEach((entry, index) => {
    elements.transcriptList.appendChild(createTranscriptEntryElement(entry, index));
  });
}

function handleTranscriptSourceClick(e, sourceEl) {
  const src = sourceEl.dataset.source;
  if (!src) {
    return;
  }
  toggleTranscriptSourceFilter(src, !e.shiftKey, e.shiftKey);
}

function handleTranscriptTypeClick(e, typeEl) {
  const typ = typeEl.dataset.type;
  if (!typ) {
    return;
  }
  toggleTranscriptTypeFilter(typ, !e.shiftKey, e.shiftKey);
}

// Transcript event delegation - handles clicks on dynamically created elements
elements.transcriptList.addEventListener("click", (e) => {
  const sourceEl = e.target.closest(".transcript-source.clickable");
  if (sourceEl) {
    handleTranscriptSourceClick(e, sourceEl);
    return;
  }
  const typeEl = e.target.closest(".transcript-type.clickable");
  if (typeEl) {
    handleTranscriptTypeClick(e, typeEl);
  }
});

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
  if (state.transcriptScroller) {
    state.transcriptScroller.scrollToBottom();
  } else {
    elements.transcriptList.scrollTop = elements.transcriptList.scrollHeight;
  }
});

// ============================================================================
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

function updateLogsData(entries, append) {
  if (append) {
    state.filteredLogs = state.filteredLogs.concat(entries);
  } else {
    state.filteredLogs = entries;
  }
}

function updateLogsDisplay(entries, append) {
  if (state.logsScroller) {
    if (append) {
      state.logsScroller.appendData(entries, state.logsTotalCount, state.logsHasMore);
    } else {
      state.logsScroller.setData(state.filteredLogs, state.logsTotalCount, state.logsHasMore);
    }
    renderLogsEmptyState();
  } else {
    renderLogs();
  }
  updateLogsStats();
}

function showLogsError(message) {
  elements.logsList.innerHTML = `<p class="muted">Failed to load logs: ${message}</p>`;
}

function applyLogsResult(result, append) {
  const entries = result.entries || [];
  updateLogsData(entries, append);
  state.logsTotalCount = result.total || state.filteredLogs.length;
  state.logsHasMore = state.filteredLogs.length < state.logsTotalCount;
  updateLogsDisplay(entries, append);
}

async function loadLogs(append = false) {
  const requestId = ++state.logsRequestId;
  const isCurrentRequest = () => requestId === state.logsRequestId;
  try {
    state.logsLoading = true;
    const offset = append ? state.filteredLogs.length : 0;
    const result = await fetchJSON(`/api/logs?${buildLogsQueryParams(offset)}`);
    if (isCurrentRequest()) {
      applyLogsResult(result, append);
    }
  } catch (error) {
    if (isCurrentRequest()) {
      showLogsError(error.message);
    }
  } finally {
    if (isCurrentRequest()) {
      state.logsLoading = false;
    }
  }
}

function loadMoreLogs() {
  return loadLogs(true);
}

// Debounced search to avoid too many API calls
let logsSearchTimeout = null;
function debouncedLogsSearch() {
  clearTimeout(logsSearchTimeout);
  logsSearchTimeout = setTimeout(() => loadLogs(false), 300);
}

function updateLogsStats() {
  let status = `Showing ${state.filteredLogs.length}`;
  if (state.logsHasMore) {
    status += ` of ${state.logsTotalCount}`;
  }
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
  if (isIncluded) {
    chip.classList.add("included");
  }
  if (isExcluded) {
    chip.classList.add("excluded");
  }

  const displayName = name.split(".").pop() || name;
  let prefix = "";
  if (isIncluded) {
    prefix = "+ ";
  }
  if (isExcluded) {
    prefix = "− ";
  }
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

  if (!hasFilters) {
    return;
  }

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

function createLogHeaderHtml(log) {
  let html = `<span class="log-level">${(log.level || "INFO").toUpperCase()}</span>`;
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
  return `<div class="log-header">${html}</div>`;
}

function createLogBodyHtml(log) {
  let html = "";
  if (log.message) {
    html += `<div class="log-message">${escapeHtml(log.message)}</div>`;
  }
  if (log.context && Object.keys(log.context).length > 0) {
    html += `<pre class="log-context">${escapeHtml(JSON.stringify(log.context, null, 2))}</pre>`;
  }
  if (log.exc_info) {
    html += `<pre class="log-exception">${escapeHtml(log.exc_info)}</pre>`;
  }
  return html;
}

/**
 * Creates a single log entry DOM element.
 * Used by both virtual scroller and fallback rendering.
 */
function createLogEntryElement(log, index) {
  const entry = document.createElement("div");
  entry.className = `log-entry log-${(log.level || "INFO").toLowerCase()}`;
  entry.dataset.index = index;
  entry.innerHTML = createLogHeaderHtml(log) + createLogBodyHtml(log);
  return entry;
}

/**
 * Renders empty state for logs when using virtual scroller.
 */
function renderLogsEmptyState() {
  if (state.filteredLogs.length === 0) {
    // Show empty state inside the virtual scroller container
    if (state.logsScroller) {
      state.logsScroller.reset();
    }
    elements.logsList.innerHTML = '<div class="logs-empty">No log entries match filters</div>';
  }
}

/**
 * Initializes the logs virtual scroller.
 * Called once after DOM is ready.
 */
function initLogsVirtualScroller() {
  if (state.logsScroller) {
    state.logsScroller.destroy();
  }

  state.logsScroller = new VirtualScroller({
    container: elements.logsList,
    estimatedItemHeight: 80, // Log entries are typically shorter
    bufferSize: 20,
    renderItem: createLogEntryElement,
    onLoadMore: loadMoreLogs,
    onLoadError: (error) => showToast(`Failed to load more logs: ${error.message}`, "error"),
  });
}

/**
 * Fallback render function for logs (without virtual scrolling).
 * Used when virtual scroller is not initialized.
 */
function renderLogs() {
  // If virtual scroller is available, use it instead
  if (state.logsScroller) {
    if (state.filteredLogs.length === 0) {
      renderLogsEmptyState();
    } else {
      state.logsScroller.setData(state.filteredLogs, state.logsTotalCount, state.logsHasMore);
    }
    return;
  }

  // Fallback: render all items (original behavior)
  elements.logsList.innerHTML = "";

  if (state.filteredLogs.length === 0) {
    elements.logsList.innerHTML = '<div class="logs-empty">No log entries match filters</div>';
    return;
  }

  state.filteredLogs.forEach((log, index) => {
    elements.logsList.appendChild(createLogEntryElement(log, index));
  });
}

function handleLoggerClick(e, loggerEl) {
  const logger = loggerEl.dataset.logger;
  if (logger) {
    toggleLoggerFilter(logger, !e.shiftKey, e.shiftKey);
  }
}

function handleEventClick(e, eventEl) {
  const event = eventEl.dataset.event;
  if (event) {
    toggleEventFilter(event, !e.shiftKey, e.shiftKey);
  }
}

// Logs event delegation - handles clicks on dynamically created elements
elements.logsList.addEventListener("click", (e) => {
  const loggerEl = e.target.closest(".log-logger.clickable");
  if (loggerEl) {
    handleLoggerClick(e, loggerEl);
    return;
  }
  const eventEl = e.target.closest(".log-event-name.clickable");
  if (eventEl) {
    handleEventClick(e, eventEl);
  }
});

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
  state.logsEventChipFilter = elements.logsEventFilter.value;
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
  document.querySelectorAll(".level-checkbox input").forEach((cb) => {
    cb.checked = true;
  });

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
  if (state.logsScroller) {
    state.logsScroller.scrollToBottom();
  } else {
    elements.logsList.scrollTop = elements.logsList.scrollHeight;
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
    item.className = `file-item${fullPath === state.selectedFile ? " active" : ""}`;
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

    if (result.type === "image") {
      // Whitelist of allowed MIME types to prevent XSS via crafted MIME strings
      const allowedMimeTypes = new Set([
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
        "image/svg+xml",
        "image/x-icon",
        "image/bmp",
      ]);
      const mimeType = allowedMimeTypes.has(result.mime_type) ? result.mime_type : "image/png";
      const dataUrl = `data:${mimeType};base64,${result.content}`;
      elements.filesystemViewer.innerHTML = `<div class="image-container"><img src="${dataUrl}" alt="${escapeHtml(displayPath)}" class="filesystem-image" /></div>`;
      state.fileContent = null;
    } else if (result.type === "binary") {
      // biome-ignore lint/nursery/noSecrets: HTML string, not a secret
      elements.filesystemViewer.innerHTML = '<p class="muted">Binary file cannot be displayed</p>';
      state.fileContent = null;
    } else {
      const content =
        result.type === "json" ? JSON.stringify(result.content, null, 2) : result.content;
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
// ENVIRONMENT
// ============================================================================

async function loadEnvironment() {
  try {
    const data = await fetchJSON("/api/environment");
    state.environmentData = data;
    state.hasEnvironmentData =
      data.system !== null || data.python !== null || data.git !== null || data.container !== null;
    renderEnvironment();
  } catch (error) {
    elements.environmentData.innerHTML = `<p class="muted">Failed to load environment data: ${error.message}</p>`;
  }
}

function renderEnvSection(title, content) {
  return `<div class="environment-section"><h3 class="section-title">${title}</h3><dl class="key-value-list">${content}</dl></div>`;
}

function renderSystemSection(sys) {
  if (!sys) {
    return "";
  }
  return renderEnvSection(
    "System",
    `<dt>OS</dt><dd>${escapeHtml(sys.os_name)} ${escapeHtml(sys.os_release)}</dd>` +
      `<dt>Kernel</dt><dd class="mono">${escapeHtml(sys.kernel_version)}</dd>` +
      `<dt>Architecture</dt><dd>${escapeHtml(sys.architecture)}</dd>` +
      `<dt>Processor</dt><dd>${escapeHtml(sys.processor)}</dd>` +
      `<dt>CPU Count</dt><dd>${sys.cpu_count}</dd>` +
      `<dt>Memory</dt><dd>${formatBytes(sys.memory_total_bytes)}</dd>` +
      `<dt>Hostname</dt><dd class="mono">${escapeHtml(sys.hostname)}</dd>`
  );
}

function renderPythonSection(py) {
  if (!py) {
    return "";
  }
  let content = `<dt>Version</dt><dd class="mono">${escapeHtml(py.version)}</dd>`;
  if (py.version_info) {
    content += `<dt>Version Info</dt><dd class="mono">${JSON.stringify(py.version_info)}</dd>`;
  }
  content +=
    `<dt>Implementation</dt><dd>${escapeHtml(py.implementation)}</dd>` +
    `<dt>Executable</dt><dd class="mono">${escapeHtml(py.executable)}</dd>` +
    `<dt>Prefix</dt><dd class="mono">${escapeHtml(py.prefix)}</dd>` +
    `<dt>Base Prefix</dt><dd class="mono">${escapeHtml(py.base_prefix)}</dd>` +
    `<dt>Virtualenv</dt><dd>${py.is_virtualenv ? "Yes" : "No"}</dd>`;
  return renderEnvSection("Python", content);
}

function renderGitSection(git) {
  if (!git) {
    return "";
  }
  let content =
    `<dt>Repo Root</dt><dd class="mono">${escapeHtml(git.repo_root)}</dd>` +
    `<dt>Branch</dt><dd class="mono">${escapeHtml(git.branch)}</dd>` +
    `<dt>Commit</dt><dd class="mono">${escapeHtml(git.commit_short)}</dd>` +
    `<dt>Full SHA</dt><dd class="mono small">${escapeHtml(git.commit_sha)}</dd>` +
    `<dt>Dirty</dt><dd>${git.is_dirty ? "Yes" : "No"}</dd>`;
  if (git.remotes && Object.keys(git.remotes).length > 0) {
    const remoteItems = Object.entries(git.remotes)
      .map(
        ([name, url]) =>
          `<div><span class="mono">${escapeHtml(name)}:</span> ${escapeHtml(url)}</div>`
      )
      .join("");
    content += `<dt>Remotes</dt><dd><div class="nested-list">${remoteItems}</div></dd>`;
  }
  if (git.tags && git.tags.length > 0) {
    content += `<dt>Tags</dt><dd class="mono">${git.tags.map((t) => escapeHtml(t)).join(", ")}</dd>`;
  }
  return renderEnvSection("Git Repository", content);
}

function renderContainerSection(container) {
  if (!container) {
    return "";
  }
  let content =
    `<dt>Runtime</dt><dd>${escapeHtml(container.runtime)}</dd>` +
    `<dt>Container ID</dt><dd class="mono">${escapeHtml(container.container_id)}</dd>` +
    `<dt>Image</dt><dd class="mono">${escapeHtml(container.image)}</dd>`;
  if (container.image_digest) {
    content += `<dt>Image Digest</dt><dd class="mono small">${escapeHtml(container.image_digest)}</dd>`;
  }
  if (container.cgroup_path) {
    content += `<dt>Cgroup Path</dt><dd class="mono">${escapeHtml(container.cgroup_path)}</dd>`;
  }
  return renderEnvSection("Container", content);
}

function renderEnvVarsSection(envVars) {
  if (!envVars || Object.keys(envVars).length === 0) {
    return "";
  }
  const content = Object.entries(envVars)
    .map(
      ([key, value]) =>
        `<dt class="mono">${escapeHtml(key)}</dt><dd class="mono small">${escapeHtml(value)}</dd>`
    )
    .join("");
  return renderEnvSection("Environment Variables", content);
}

function renderEnvironment() {
  if (!state.hasEnvironmentData) {
    elements.environmentEmptyState.classList.remove("hidden");
    elements.environmentContent.classList.add("hidden");
    return;
  }
  elements.environmentEmptyState.classList.add("hidden");
  elements.environmentContent.classList.remove("hidden");

  const data = state.environmentData;
  elements.environmentData.innerHTML =
    renderSystemSection(data.system) +
    renderPythonSection(data.python) +
    renderGitSection(data.git) +
    renderContainerSection(data.container) +
    renderEnvVarsSection(data.env_vars);
}

elements.environmentCopy.addEventListener("click", async () => {
  if (!state.environmentData) {
    showToast("No environment data to copy", "error");
    return;
  }
  try {
    await navigator.clipboard.writeText(JSON.stringify(state.environmentData, null, 2));
    showToast("Copied environment data to clipboard", "success");
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
elements.shortcutsOverlay
  .querySelector(".shortcuts-backdrop")
  .addEventListener("click", closeShortcuts);
elements.helpButton.addEventListener("click", openShortcuts);

// ============================================================================
// ZOOM MODAL
// ============================================================================

/**
 * Opens the zoom modal for a transcript entry.
 */
function openTranscriptZoom(index) {
  const entry = state.transcriptEntries[index];
  if (!entry) {
    return;
  }
  state.zoomOpen = true;
  state.zoomType = "transcript";
  state.zoomIndex = index;
  state.zoomEntry = entry;
  renderZoomModal();
  elements.zoomModal.classList.remove("hidden");
}

/**
 * Closes the zoom modal.
 */
function closeZoomModal() {
  state.zoomOpen = false;
  state.zoomType = null;
  state.zoomIndex = -1;
  state.zoomEntry = null;
  elements.zoomModal.classList.add("hidden");
}

/**
 * Navigates to the previous entry in the zoom modal.
 */
function zoomPrev() {
  if (!state.zoomOpen || state.zoomIndex <= 0) {
    return;
  }
  openTranscriptZoom(state.zoomIndex - 1);
}

/**
 * Navigates to the next entry in the zoom modal.
 */
function zoomNext() {
  if (!state.zoomOpen) {
    return;
  }
  const maxIndex = state.transcriptEntries.length - 1;
  if (state.zoomIndex >= maxIndex) {
    return;
  }
  openTranscriptZoom(state.zoomIndex + 1);
}

/**
 * Renders the zoom modal content based on current state.
 */
function renderZoomModal() {
  if (!state.zoomEntry) {
    return;
  }

  renderTranscriptZoom();

  // Update navigation button states
  updateZoomNavigation();
}

/**
 * Checks if an entry is a tool call (assistant message with tool_name).
 */
function isToolCall(entry) {
  return entry.entry_type === "assistant" && entry.tool_name && entry.tool_name !== "";
}

/**
 * Parse entry's data from parsed or raw_json fields.
 */
function parseEntryData(entry) {
  const parsed = entry.parsed;
  if (typeof parsed === "string") {
    try {
      return JSON.parse(parsed);
    } catch {
      return null;
    }
  }
  if (parsed) {
    return parsed;
  }
  if (entry.raw_json) {
    try {
      return JSON.parse(entry.raw_json);
    } catch {
      return null;
    }
  }
  return null;
}

/**
 * Get message content array from parsed entry data.
 */
function getMessageContent(parsed) {
  const content = parsed?.message?.content;
  return Array.isArray(content) ? content : null;
}

/**
 * Checks if an entry is a tool result (user message containing tool_result).
 */
function isToolResult(entry) {
  if (entry.entry_type !== "user") {
    return false;
  }
  const content = getMessageContent(parseEntryData(entry));
  return content ? content.some((item) => item.type === "tool_result") : false;
}

/**
 * Finds tool ID in a content array (tool_use.id or tool_result.tool_use_id).
 */
function findToolIdInContent(content) {
  for (const item of content) {
    if (item.type === "tool_use" && item.id) {
      return item.id;
    }
    if (item.type === "tool_result" && item.tool_use_id) {
      return item.tool_use_id;
    }
  }
  return null;
}

/**
 * Extracts tool_use_id from an entry's nested message.content structure.
 */
function extractToolUseId(entry) {
  if (entry.tool_use_id) {
    return entry.tool_use_id;
  }
  const content = getMessageContent(parseEntryData(entry));
  return content ? findToolIdInContent(content) : null;
}

/**
 * Find matching tool result for a tool call entry.
 */
function findMatchingToolResult(entries, startIndex, toolId, usedIndices) {
  const searchLimit = Math.min(startIndex + 10, entries.length);
  for (let j = startIndex + 1; j < searchLimit; j++) {
    if (usedIndices.has(j)) {
      continue;
    }
    const candidate = entries[j];
    if (isToolResult(candidate) && extractToolUseId(candidate) === toolId) {
      usedIndices.add(j);
      return candidate;
    }
  }
  return null;
}

/**
 * Process a single entry, potentially combining it with a tool result.
 */
function processEntry(entry, entries, index, usedIndices) {
  if (!isToolCall(entry)) {
    return entry;
  }
  const resultEntry = findMatchingToolResult(entries, index, extractToolUseId(entry), usedIndices);
  if (resultEntry) {
    return { ...entry, isComposite: true, toolResult: resultEntry };
  }
  return entry;
}

/**
 * Preprocesses transcript entries to combine tool calls with their results.
 */
function preprocessTranscriptEntries(entries) {
  const processed = [];
  const usedIndices = new Set();
  for (let i = 0; i < entries.length; i++) {
    if (!usedIndices.has(i)) {
      processed.push(processEntry(entries[i], entries, i, usedIndices));
    }
  }

  return processed;
}

/**
 * Helper: Renders content (text or JSON) in a scrollable container.
 */
function renderContentSection(content, parentElement) {
  const section = document.createElement("div");
  section.className = "zoom-section";

  if (content.kind === "json") {
    const jsonContainer = document.createElement("div");
    jsonContainer.className = "zoom-json-tree";
    try {
      const parsed = JSON.parse(content.value);
      jsonContainer.appendChild(renderZoomJsonTree(parsed, "", 0));
    } catch {
      jsonContainer.innerHTML = `<pre>${escapeHtml(content.value)}</pre>`;
    }
    section.appendChild(jsonContainer);
  } else {
    const scrollContainer = document.createElement("div");
    scrollContainer.className = "zoom-text-scroll";
    scrollContainer.innerHTML = `<div class="zoom-text-content">${escapeHtml(content.value)}</div>`;
    section.appendChild(scrollContainer);
  }

  parentElement.appendChild(section);
}

/**
 * Helper: Renders raw JSON tree with label.
 */
function renderRawJsonSection(payload, label, parentElement) {
  if (!payload) {
    return;
  }

  const section = document.createElement("div");
  section.className = "zoom-section";
  const labelDiv = document.createElement("div");
  labelDiv.className = "zoom-section-label";
  labelDiv.textContent = label;
  section.appendChild(labelDiv);

  const jsonContainer = document.createElement("div");
  jsonContainer.className = "zoom-json-tree";
  jsonContainer.appendChild(renderZoomJsonTree(payload, "", 0));
  section.appendChild(jsonContainer);

  parentElement.appendChild(section);
}

/**
 * Renders a transcript entry in the zoom modal.
 */
function renderTranscriptZoom() {
  const entry = state.zoomEntry;

  if (entry.isComposite) {
    renderCompositeEntryZoom(entry);
  } else {
    renderRegularEntryZoom(entry);
  }
}

function setZoomHeader(typeText, typeClass, entry) {
  elements.zoomModalType.textContent = typeText;
  elements.zoomModalType.className = `zoom-type-badge ${typeClass}`;
  elements.zoomModalTimestamp.textContent = entry.timestamp || "";
  const source = entry.transcript_source || "";
  const seq = entry.sequence_number != null ? `#${entry.sequence_number}` : "";
  elements.zoomModalSource.textContent = source ? `${source}${seq}` : "";
  elements.zoomModalSource.style.display = source ? "" : "none";
}

function createZoomSection(label, content) {
  const section = document.createElement("div");
  section.className = "zoom-section";
  section.innerHTML = `<div class="zoom-section-label">${label}</div><div class="zoom-text-content">${content}</div>`;
  return section;
}

function createCombinedHeader(badgeText, labelText, cssClass) {
  const header = document.createElement("div");
  header.className = `zoom-combined-header ${cssClass}`;
  header.innerHTML = `<span class="zoom-combined-badge">${badgeText}</span> ${labelText}`;
  return header;
}

/**
 * Renders a composite tool call + result entry in zoom view.
 */
function renderCompositeEntryZoom(entry) {
  setZoomHeader(`TOOL: ${entry.tool_name || "unknown"}`, "zoom-type-tool_use", entry);

  elements.zoomContent.innerHTML = "";
  elements.zoomContent.appendChild(
    createCombinedHeader("CALL", "Tool Input", "zoom-combined-call")
  );
  const callContent = formatTranscriptContent(entry);
  if (callContent.value) {
    renderContentSection(callContent, elements.zoomContent);
  }
  renderRawJsonSection(entry.parsed || entry.raw_json, "Call Raw JSON", elements.zoomContent);

  elements.zoomDetails.innerHTML = "";
  elements.zoomDetails.appendChild(
    createCombinedHeader("RESULT", "Tool Output", "zoom-combined-result")
  );
  if (entry.toolResult) {
    const resultContent = formatTranscriptContent(entry.toolResult);
    if (resultContent.value) {
      renderContentSection(resultContent, elements.zoomDetails);
    }
    renderRawJsonSection(
      entry.toolResult.parsed || entry.toolResult.raw_json,
      "Result Raw JSON",
      elements.zoomDetails
    );
  }
  if (entry.tool_use_id) {
    const metaSection = createZoomSection(
      "Tool Use ID",
      `<span class="mono">${escapeHtml(entry.tool_use_id)}</span>`
    );
    metaSection.classList.add("zoom-tool-meta");
    elements.zoomDetails.appendChild(metaSection);
  }
}

/**
 * Renders a regular (non-composite) entry in zoom view.
 */
function renderRegularEntryZoom(entry) {
  const entryType = entry.entry_type || "unknown";
  setZoomHeader(entryType.toUpperCase(), `zoom-type-${entryType}`, entry);

  elements.zoomContent.innerHTML = "";
  const content = formatTranscriptContent(entry);
  if (content.value) {
    renderContentSection(content, elements.zoomContent);
  }

  elements.zoomDetails.innerHTML = "";
  if (entry.tool_name) {
    elements.zoomDetails.appendChild(createZoomSection("Tool", escapeHtml(entry.tool_name)));
  }
  if (entry.tool_use_id) {
    elements.zoomDetails.appendChild(
      createZoomSection("Tool Use ID", escapeHtml(entry.tool_use_id))
    );
  }
  if (entry.prompt_name) {
    elements.zoomDetails.appendChild(createZoomSection("Prompt", escapeHtml(entry.prompt_name)));
  }
  renderRawJsonSection(entry.parsed || entry.raw_json, "Raw JSON", elements.zoomDetails);
  if (elements.zoomDetails.children.length === 0) {
    elements.zoomDetails.innerHTML = `<div class="zoom-empty">(no additional details)</div>`;
  }
}

/**
 * Creates the key prefix HTML for JSON tree nodes.
 */
function zoomJsonKeyHtml(key) {
  return key ? `<span class="zoom-json-key">${escapeHtml(key)}</span>: ` : "";
}

/**
 * Renders a primitive JSON value (null, boolean, number).
 */
function renderZoomJsonPrimitive(node, key, value, cssClass) {
  const valueHtml = value === null ? "null" : String(value);
  node.innerHTML = `${zoomJsonKeyHtml(key)}<span class="${cssClass}">${valueHtml}</span>`;
}

/**
 * Attempts to parse a string as JSON and render it if successful.
 * Returns the wrapper node if parsed, null otherwise.
 */
function tryRenderParsedJsonString(value, key, depth) {
  if (value.length <= 1) {
    return null;
  }
  const looksLikeJson =
    (value.startsWith("{") && value.endsWith("}")) ||
    (value.startsWith("[") && value.endsWith("]"));
  if (!looksLikeJson) {
    return null;
  }
  try {
    const parsed = JSON.parse(value);
    const wrapper = document.createElement("div");
    wrapper.className = "zoom-json-node";
    if (key) {
      const keyLabel = document.createElement("span");
      keyLabel.innerHTML = `<span class="zoom-json-key">${escapeHtml(key)}</span>: <span class="zoom-json-parsed-hint">(parsed JSON string)</span>`;
      wrapper.appendChild(keyLabel);
    }
    wrapper.appendChild(renderZoomJsonTree(parsed, "", depth));
    return wrapper;
  } catch {
    return null;
  }
}

/**
 * Renders a string value in the JSON tree.
 */
function renderZoomJsonString(node, key, value, depth) {
  const parsedNode = tryRenderParsedJsonString(value, key, depth);
  if (parsedNode) {
    return parsedNode;
  }
  const escaped = escapeHtml(value);
  const formatted = escaped.replace(/\n/g, "<br>");
  node.innerHTML = `${zoomJsonKeyHtml(key)}<span class="zoom-json-string">"${formatted}"</span>`;
  return node;
}

/**
 * Attaches toggle click handler for collapsible JSON tree nodes.
 */
function attachZoomJsonToggle(header, children) {
  header.addEventListener("click", () => {
    const isOpen = children.style.display !== "none";
    children.style.display = isOpen ? "none" : "block";
    header.querySelector(".zoom-json-toggle").textContent = isOpen ? "▶" : "▼";
    header.querySelector(".zoom-json-close-bracket").style.display = isOpen ? "inline" : "none";
  });
}

/**
 * Creates the header element for collapsible JSON tree nodes.
 */
function createZoomJsonHeader(key, openBracket, count, countLabel, isExpanded) {
  const header = document.createElement("div");
  header.className = "zoom-json-header";
  const toggle = isExpanded ? "▼" : "▶";
  const closeBracket = openBracket === "[" ? "]" : "}";
  header.innerHTML = `<span class="zoom-json-toggle">${toggle}</span>${zoomJsonKeyHtml(key)}<span class="zoom-json-bracket">${openBracket}</span><span class="zoom-json-count">${count} ${countLabel}</span><span class="zoom-json-bracket zoom-json-close-bracket" style="display: ${isExpanded ? "none" : "inline"}">${closeBracket}</span>`;
  return header;
}

/**
 * Creates children container and close bracket for collapsible nodes.
 */
function createZoomJsonChildren(closeBracket, isExpanded) {
  const children = document.createElement("div");
  children.className = "zoom-json-children";
  children.style.display = isExpanded ? "block" : "none";

  const closeBracketEl = document.createElement("div");
  closeBracketEl.className = "zoom-json-close";
  closeBracketEl.innerHTML = `<span class="zoom-json-bracket">${closeBracket}</span>`;
  closeBracketEl.style.display = isExpanded ? "block" : "none";

  return { children, closeBracketEl };
}

/**
 * Renders an array value in the JSON tree.
 */
function renderZoomJsonArray(node, key, value, depth) {
  const isExpanded = depth < 2;
  const header = createZoomJsonHeader(key, "[", value.length, "items", isExpanded);
  node.appendChild(header);

  const { children, closeBracketEl } = createZoomJsonChildren("]", isExpanded);
  for (let i = 0; i < value.length; i++) {
    children.appendChild(renderZoomJsonTree(value[i], String(i), depth + 1));
  }
  children.appendChild(closeBracketEl);
  node.appendChild(children);
  attachZoomJsonToggle(header, children);
  return node;
}

/**
 * Renders an object value in the JSON tree.
 */
function renderZoomJsonObject(node, key, value, depth) {
  const keys = Object.keys(value);
  const isExpanded = depth < 2;
  const header = createZoomJsonHeader(key, "{", keys.length, "keys", isExpanded);
  node.appendChild(header);

  const { children, closeBracketEl } = createZoomJsonChildren("}", isExpanded);
  for (const k of keys) {
    children.appendChild(renderZoomJsonTree(value[k], k, depth + 1));
  }
  children.appendChild(closeBracketEl);
  node.appendChild(children);
  attachZoomJsonToggle(header, children);
  return node;
}

/**
 * Renders a collapsible JSON tree for the zoom modal details panel.
 */
function renderZoomJsonTree(value, key, depth) {
  const node = document.createElement("div");
  node.className = "zoom-json-node";

  if (value === null) {
    renderZoomJsonPrimitive(node, key, null, "zoom-json-null");
    return node;
  }
  if (typeof value === "boolean") {
    renderZoomJsonPrimitive(node, key, value, "zoom-json-bool");
    return node;
  }
  if (typeof value === "number") {
    renderZoomJsonPrimitive(node, key, value, "zoom-json-number");
    return node;
  }
  if (typeof value === "string") {
    return renderZoomJsonString(node, key, value, depth);
  }
  if (Array.isArray(value)) {
    return renderZoomJsonArray(node, key, value, depth);
  }
  if (typeof value === "object") {
    return renderZoomJsonObject(node, key, value, depth);
  }

  node.textContent = String(value);
  return node;
}

/**
 * Updates the state of prev/next navigation buttons.
 */
function updateZoomNavigation() {
  if (!state.zoomOpen) {
    return;
  }

  const maxIndex = state.transcriptEntries.length - 1;

  elements.zoomPrev.disabled = state.zoomIndex <= 0;
  elements.zoomNext.disabled = state.zoomIndex >= maxIndex;
}

/**
 * Copies the current zoom entry as JSON to clipboard.
 */
async function copyZoomEntry() {
  if (!state.zoomEntry) {
    showToast("No entry to copy", "error");
    return;
  }
  try {
    await navigator.clipboard.writeText(JSON.stringify(state.zoomEntry, null, 2));
    showToast("Copied entry to clipboard", "success");
  } catch {
    showToast("Failed to copy", "error");
  }
}

// Zoom modal event listeners
elements.zoomClose.addEventListener("click", closeZoomModal);
elements.zoomModal.querySelector(".zoom-modal-backdrop").addEventListener("click", closeZoomModal);
elements.zoomCopy.addEventListener("click", copyZoomEntry);
elements.zoomPrev.addEventListener("click", zoomPrev);
elements.zoomNext.addEventListener("click", zoomNext);

// Event delegation for zoom buttons in transcript list
elements.transcriptList.addEventListener("click", (e) => {
  const zoomBtn = e.target.closest(".zoom-button[data-zoom-index]");
  if (zoomBtn) {
    e.preventDefault();
    e.stopPropagation();
    const index = Number.parseInt(zoomBtn.dataset.zoomIndex, 10);
    openTranscriptZoom(index);
  }
});

// ============================================================================
// KEYBOARD SHORTCUTS
// ============================================================================

/**
 * Handles escape key to close modals/dialogs.
 * Returns true if the event was handled.
 */
function handleEscapeKey(e) {
  if (state.zoomOpen) {
    e.preventDefault();
    closeZoomModal();
    return true;
  }
  if (state.shortcutsOpen) {
    e.preventDefault();
    closeShortcuts();
    return true;
  }
  return false;
}

/**
 * Handles navigation keys when zoom modal is open.
 * Returns true if the event was handled.
 */
function handleZoomModalKeys(e) {
  const nextKeys = ["j", "J", "ArrowDown", "ArrowRight"];
  const prevKeys = ["k", "K", "ArrowUp", "ArrowLeft"];
  if (nextKeys.includes(e.key)) {
    e.preventDefault();
    zoomNext();
    return true;
  }
  if (prevKeys.includes(e.key)) {
    e.preventDefault();
    zoomPrev();
    return true;
  }
  return true; // Block other keys while zoom modal is open
}

/**
 * Checks if target is an input element that should block shortcuts.
 */
function isInputElement(target) {
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
}

/**
 * Single key shortcuts mapping.
 */
const KEYBOARD_SHORTCUTS = {
  r: reloadBundle,
  R: reloadBundle,
  d: toggleTheme,
  D: toggleTheme,
  "[": toggleSidebar,
  "/": focusCurrentSearch,
  j: navigateNext,
  J: navigateNext,
  k: navigatePrev,
  K: navigatePrev,
};

/**
 * Tab view names indexed by number key.
 */
const TAB_VIEWS = ["sessions", "transcript", "logs", "filesystem", "environment"];

/**
 * Handles number key shortcuts for tab switching.
 */
function handleTabShortcut(e, key) {
  if (key >= "1" && key <= "5") {
    e.preventDefault();
    switchView(TAB_VIEWS[Number.parseInt(key, 10) - 1]);
    return true;
  }
  return false;
}

/**
 * Handles arrow key shortcuts for bundle navigation.
 */
function handleArrowShortcut(e, key) {
  if (key === "ArrowLeft" || key === "ArrowRight") {
    e.preventDefault();
    navigateBundle(key === "ArrowRight" ? 1 : -1);
    return true;
  }
  return false;
}

/**
 * Handles global keyboard shortcuts.
 * Returns true if the event was handled.
 */
function handleGlobalShortcuts(e) {
  const key = e.key;

  if (handleTabShortcut(e, key)) {
    return true;
  }

  if (KEYBOARD_SHORTCUTS[key]) {
    e.preventDefault();
    KEYBOARD_SHORTCUTS[key]();
    return true;
  }

  // Help shortcut
  if (key === "?" || (e.shiftKey && key === "/")) {
    e.preventDefault();
    openShortcuts();
    return true;
  }

  return handleArrowShortcut(e, key);
}

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    handleEscapeKey(e);
    return;
  }

  if (state.zoomOpen) {
    handleZoomModalKeys(e);
    return;
  }

  if (state.shortcutsOpen || isInputElement(e.target)) {
    return;
  }

  handleGlobalShortcuts(e);
});

function focusCurrentSearch() {
  if (state.activeView === "sessions") {
    elements.itemSearch.focus();
  } else if (state.activeView === "transcript") {
    elements.transcriptSearch.focus();
  } else if (state.activeView === "logs") {
    elements.logsSearch.focus();
  } else if (state.activeView === "filesystem") {
    elements.filesystemFilter.focus();
  }
}

function navigateNext() {
  if (state.activeView === "sessions") {
    focusItem(state.focusedItemIndex + 1);
  } else if (state.activeView === "transcript") {
    scrollTranscriptBy(1);
  } else if (state.activeView === "logs") {
    scrollLogsBy(1);
  }
}

function navigatePrev() {
  if (state.activeView === "sessions") {
    focusItem(state.focusedItemIndex - 1);
  } else if (state.activeView === "transcript") {
    scrollTranscriptBy(-1);
  } else if (state.activeView === "logs") {
    scrollLogsBy(-1);
  }
}

function scrollTranscriptBy(delta) {
  const entries = elements.transcriptList.querySelectorAll(".transcript-entry");
  if (entries.length === 0) {
    return;
  }
  const scrollHeight = entries[0].offsetHeight;
  elements.transcriptList.scrollBy({ top: scrollHeight * delta, behavior: "smooth" });
}

function scrollLogsBy(delta) {
  const entries = elements.logsList.querySelectorAll(".log-entry");
  if (entries.length === 0) {
    return;
  }
  const scrollHeight = entries[0].offsetHeight;
  elements.logsList.scrollBy({ top: scrollHeight * delta, behavior: "smooth" });
}

function focusItem(index) {
  const items = elements.jsonViewer.querySelectorAll(".item-card");
  if (items.length === 0) {
    return;
  }
  const newIndex = Math.max(0, Math.min(index, items.length - 1));
  state.focusedItemIndex = newIndex;
  items.forEach((item, i) => item.classList.toggle("focused", i === newIndex));
  items[newIndex].scrollIntoView({ behavior: "smooth", block: "center" });
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

function getMarkdownView(path) {
  return state.markdownViews.get(pathKey(path)) || "html";
}
function setMarkdownView(path, view) {
  state.markdownViews.set(pathKey(path), view);
}

function shouldOpen(path, depth) {
  const key = pathKey(path);
  if (state.closedPaths.has(key)) {
    return false;
  }
  if (state.openPaths.has(key)) {
    return true;
  }
  return depth < state.expandDepth;
}

function setOpen(path, open) {
  const key = pathKey(path);
  if (open) {
    state.openPaths.add(key);
    state.closedPaths.delete(key);
  } else {
    state.openPaths.delete(key);
    state.closedPaths.add(key);
  }
}

function applyDepth(items, depth) {
  state.openPaths = new Set();
  state.closedPaths = new Set();
  const walk = (value, path, d) => {
    if (!(Array.isArray(value) || isObject(value))) {
      return;
    }
    if (d < depth) {
      setOpen(path, true);
    }
    if (Array.isArray(value)) {
      value.forEach((c, i) => walk(c, path.concat(String(i)), d + 1));
    } else {
      Object.entries(value).forEach(([k, v]) => walk(v, path.concat(k), d + 1));
    }
  };
  items.forEach((item, i) => walk(item, [`item-${i}`], 0));
}

function setOpenForAll(items, open) {
  state.openPaths = new Set();
  state.closedPaths = new Set();
  const update = (value, path) => {
    if (!(Array.isArray(value) || isObject(value))) {
      return;
    }
    setOpen(path, open);
    if (Array.isArray(value)) {
      value.forEach((c, i) => update(c, path.concat(String(i))));
    } else {
      Object.entries(value).forEach(([k, v]) => update(v, path.concat(k)));
    }
  };
  items.forEach((item, i) => update(item, [`item-${i}`]));
}

function getFilteredItems() {
  const query = state.searchQuery.toLowerCase().trim();
  if (!query) {
    return state.currentItems.map((item, index) => ({ item, index }));
  }
  return state.currentItems
    .map((item, index) => ({ item, index, text: JSON.stringify(item).toLowerCase() }))
    .filter((e) => e.text.includes(query))
    .map(({ item, index }) => ({ item, index }));
}

/**
 * Creates the type badge text for tree nodes.
 */
function getTreeTypeBadge(value, type) {
  if (type === "array") {
    return `array (${value.length})`;
  }
  if (type === "object" && value !== null) {
    return `object (${Object.keys(value).length})`;
  }
  return type;
}

/**
 * Renders a markdown leaf node with toggle between rendered and raw views.
 */
function renderMarkdownLeaf(wrapper, markdown, path) {
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
  attachMarkdownToggleHandlers(wrapper, path);
}

/**
 * Attaches click handlers for markdown view toggle buttons.
 */
function attachMarkdownToggleHandlers(wrapper, path) {
  const buttons = wrapper.querySelectorAll(".markdown-toggle button");
  const sections = wrapper.querySelectorAll(".markdown-section");
  buttons[0].addEventListener("click", () => {
    setMarkdownView(path, "html");
    buttons[0].classList.add("active");
    buttons[1].classList.remove("active");
    sections[0].style.display = "flex";
    sections[1].style.display = "none";
  });
  buttons[1].addEventListener("click", () => {
    setMarkdownView(path, "raw");
    buttons[1].classList.add("active");
    buttons[0].classList.remove("active");
    sections[1].style.display = "flex";
    sections[0].style.display = "none";
  });
}

/**
 * Renders a non-expandable leaf node (markdown or simple value).
 */
function renderTreeLeaf(node, header, body, value, markdown, path) {
  const wrapper = document.createElement("div");
  wrapper.className = "leaf-wrapper";

  if (markdown) {
    renderMarkdownLeaf(wrapper, markdown, path);
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

/**
 * Creates the expand/collapse controls for tree nodes.
 */
function createTreeControls(path, depth) {
  const controls = document.createElement("div");
  controls.className = "tree-controls";

  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "tree-toggle";
  toggle.textContent = shouldOpen(path, depth) ? "Collapse" : "Expand";
  controls.appendChild(toggle);

  toggle.addEventListener("click", () => {
    setOpen(path, !shouldOpen(path, depth));
    renderItems(state.currentItems);
  });

  return controls;
}

/**
 * Renders simple array as compact chips.
 */
function renderSimpleArrayChips(container, value) {
  container.classList.add("compact-array");
  value.forEach((child) => {
    const chip = document.createElement("span");
    chip.className = "array-chip";
    chip.textContent = String(child);
    container.appendChild(chip);
  });
}

/**
 * Populates tree children container based on value type and expansion state.
 */
function populateTreeChildren(container, value, path, depth) {
  const childCount = Array.isArray(value) ? value.length : Object.keys(value).length;
  if (childCount === 0) {
    // biome-ignore lint/nursery/noSecrets: HTML string, not a secret
    container.innerHTML = '<span class="muted">(empty)</span>';
    return;
  }
  if (!shouldOpen(path, depth)) {
    container.style.display = "none";
    return;
  }
  if (Array.isArray(value)) {
    if (isSimpleArray(value)) {
      renderSimpleArrayChips(container, value);
    } else {
      value.forEach((child, i) => {
        container.appendChild(renderTree(child, path.concat(String(i)), depth + 1, `[${i}]`));
      });
    }
  } else {
    Object.entries(value).forEach(([key, child]) => {
      container.appendChild(renderTree(child, path.concat(key), depth + 1, key));
    });
  }
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
  badge.textContent = getTreeTypeBadge(value, type);
  header.appendChild(badge);

  const body = document.createElement("div");
  body.className = "tree-body";

  const expandable = !markdown && (Array.isArray(value) || isObject(value));

  if (!expandable) {
    return renderTreeLeaf(node, header, body, value, markdown, path);
  }

  const childCount = Array.isArray(value) ? value.length : Object.keys(value).length;
  if (childCount > 0) {
    header.appendChild(createTreeControls(path, depth));
  }

  const childrenContainer = document.createElement("div");
  childrenContainer.className = "tree-children";
  populateTreeChildren(childrenContainer, value, path, depth);

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
    if (exists) {
      await selectSlice(state.selectedSlice);
    } else if (meta.slices.length > 0) {
      await selectSlice(meta.slices[0].slice_type);
    }
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  setLoading(true);
  try {
    // Initialize virtual scrollers for logs and transcript
    initLogsVirtualScroller();
    initTranscriptVirtualScroller();

    await refreshMeta();
    await refreshBundles();
  } catch (error) {
    showToast(`Load failed: ${error.message}`, "error");
  } finally {
    setLoading(false);
  }
});
