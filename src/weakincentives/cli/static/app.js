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
  elements.filesystemView.classList.toggle("hidden", viewName !== "filesystem");
  elements.environmentView.classList.toggle("hidden", viewName !== "environment");

  // Initialize virtual scrollers if needed (may have been destroyed by resetViewState)
  if (viewName === "transcript" && !state.transcriptScroller) {
    initTranscriptVirtualScroller();
  } else if (viewName === "logs" && !state.logsScroller) {
    initLogsVirtualScroller();
  }

  // Load data if needed
  if (viewName === "transcript" && state.transcriptEntries.length === 0) {
    loadTranscriptFacets();
    loadTranscript();
  } else if (viewName === "logs" && state.filteredLogs.length === 0) {
    loadLogFacets();
    loadLogs();
  } else if (viewName === "filesystem" && state.allFiles.length === 0) {
    loadFilesystem();
  } else if (viewName === "environment" && state.environmentData === null) {
    loadEnvironment();
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

async function loadTranscript(append = false) {
  // Track this request (incrementing invalidates any in-flight requests)
  const requestId = ++state.transcriptRequestId;

  try {
    state.transcriptLoading = true;
    const offset = append ? state.transcriptEntries.length : 0;
    const query = buildTranscriptQueryParams(offset);
    const result = await fetchJSON(`/api/transcript?${query}`);

    // Ignore stale responses (filters changed while loading)
    if (requestId !== state.transcriptRequestId) {
      return;
    }

    const entries = result.entries || [];

    // Preprocess to combine tool calls with their results
    let processed;
    if (append) {
      // When appending, reprocess everything to handle cross-boundary pairs
      const rawEntries = [...state.transcriptRawEntries, ...entries];
      state.transcriptRawEntries = rawEntries;
      processed = preprocessTranscriptEntries(rawEntries);
    } else {
      state.transcriptRawEntries = entries;
      processed = preprocessTranscriptEntries(entries);
    }

    state.transcriptEntries = processed;
    state.transcriptTotalCount = result.total || state.transcriptRawEntries.length;
    state.transcriptHasMore = state.transcriptRawEntries.length < state.transcriptTotalCount;

    // Use virtual scroller if available
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
  } catch (error) {
    // Only show error if this is still the current request
    if (requestId === state.transcriptRequestId) {
      elements.transcriptList.innerHTML = `<p class="muted">Failed to load transcript: ${error.message}</p>`;
    }
  } finally {
    // Only clear loading state if this is still the current request
    if (requestId === state.transcriptRequestId) {
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

/**
 * Creates a single transcript entry DOM element.
 * Used by both virtual scroller and fallback rendering.
 */
function createTranscriptEntryElement(entry, index) {
  const isComposite = entry.isComposite === true;
  const toolResult = entry.toolResult;

  const entryType = entry.entry_type || "unknown";
  const role = entry.role || "";
  const cssClass = role ? `role-${role}` : `type-${entryType}`;

  const container = document.createElement("div");
  container.className = `transcript-entry ${cssClass} compact${isComposite ? " combined" : ""}`;
  container.dataset.entryIndex = index;

  const source = entry.transcript_source || "";
  const seq =
    entry.sequence_number !== null && entry.sequence_number !== undefined
      ? String(entry.sequence_number)
      : "";
  const promptName = entry.prompt_name || "";

  const content = formatTranscriptContent(entry);

  let html = `<div class="transcript-header">`;
  // Zoom button
  html += `<button class="zoom-button" type="button" data-zoom-index="${index}" title="Expand entry" aria-label="Expand entry">`;
  html += `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">`;
  html += `<polyline points="15 3 21 3 21 9"></polyline>`;
  html += `<polyline points="9 21 3 21 3 15"></polyline>`;
  html += '<line x1="21" y1="3" x2="14" y2="10"></line>';
  html += '<line x1="3" y1="21" x2="10" y2="14"></line>';
  html += "</svg></button>";

  // Show combined label for tool call + result
  if (isComposite) {
    html += `<span class="transcript-type clickable" data-type="${escapeHtml(entryType)}">tool call + result</span>`;
    if (entry.tool_name) {
      html += `<span class="transcript-tool-name">${escapeHtml(entry.tool_name)}</span>`;
    }
  } else {
    html += `<span class="transcript-type clickable" data-type="${escapeHtml(entryType)}">${escapeHtml(entryType)}</span>`;
  }

  if (entry.timestamp) {
    html += `<span class="transcript-timestamp">${escapeHtml(entry.timestamp)}</span>`;
  }
  if (source) {
    html += `<span class="transcript-source clickable" data-source="${escapeHtml(source)}">${escapeHtml(source)}${seq ? `#${escapeHtml(seq)}` : ""}</span>`;
  }
  if (promptName) {
    html += `<span class="transcript-prompt mono" title="${escapeHtml(promptName)}">${escapeHtml(promptName)}</span>`;
  }
  html += "</div>";

  // Render tool call content
  if (content.value) {
    if (content.kind === "json") {
      html += `<pre class="transcript-json">${escapeHtml(content.value)}</pre>`;
    } else {
      html += `<div class="transcript-message">${escapeHtml(content.value)}</div>`;
    }
  } else {
    html += `<div class="transcript-message muted">(no content)</div>`;
  }

  // Render tool result content if this is a composite entry
  if (isComposite && toolResult) {
    const resultContent = formatTranscriptContent(toolResult);
    html += `<div class="transcript-result-divider">↓ result</div>`;
    if (resultContent.value) {
      if (resultContent.kind === "json") {
        html += `<pre class="transcript-json">${escapeHtml(resultContent.value)}</pre>`;
      } else {
        html += `<div class="transcript-message">${escapeHtml(resultContent.value)}</div>`;
      }
    } else {
      html += `<div class="transcript-message muted">(no result)</div>`;
    }
  }

  const detailsPayload = entry.parsed || entry.raw_json;
  if (detailsPayload) {
    html += `<details class="transcript-details"><summary>Details</summary>`;
    html += `<pre>${escapeHtml(JSON.stringify(detailsPayload, null, 2))}</pre>`;
    html += "</details>";
  }

  container.innerHTML = html;

  // Event listeners handled via delegation on container (see setupTranscriptEventDelegation)
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

// Transcript event delegation - handles clicks on dynamically created elements
elements.transcriptList.addEventListener("click", (e) => {
  const sourceEl = e.target.closest(".transcript-source.clickable");
  if (sourceEl) {
    const src = sourceEl.dataset.source;
    if (src) {
      if (e.shiftKey) {
        toggleTranscriptSourceFilter(src, false, true);
      } else {
        toggleTranscriptSourceFilter(src, true, false);
      }
    }
    return;
  }

  const typeEl = e.target.closest(".transcript-type.clickable");
  if (typeEl) {
    const typ = typeEl.dataset.type;
    if (typ) {
      if (e.shiftKey) {
        toggleTranscriptTypeFilter(typ, false, true);
      } else {
        toggleTranscriptTypeFilter(typ, true, false);
      }
    }
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

async function loadLogs(append = false) {
  // Track this request (incrementing invalidates any in-flight requests)
  const requestId = ++state.logsRequestId;

  try {
    state.logsLoading = true;
    const offset = append ? state.filteredLogs.length : 0;
    const query = buildLogsQueryParams(offset);
    const result = await fetchJSON(`/api/logs?${query}`);

    // Ignore stale responses (filters changed while loading)
    if (requestId !== state.logsRequestId) {
      return;
    }

    const entries = result.entries || [];

    if (append) {
      state.filteredLogs = state.filteredLogs.concat(entries);
    } else {
      state.filteredLogs = entries;
    }
    state.logsTotalCount = result.total || state.filteredLogs.length;
    state.logsHasMore = state.filteredLogs.length < state.logsTotalCount;

    // Use virtual scroller if available
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
  } catch (error) {
    // Only show error if this is still the current request
    if (requestId === state.logsRequestId) {
      elements.logsList.innerHTML = `<p class="muted">Failed to load logs: ${error.message}</p>`;
    }
  } finally {
    // Only clear loading state if this is still the current request
    if (requestId === state.logsRequestId) {
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

/**
 * Creates a single log entry DOM element.
 * Used by both virtual scroller and fallback rendering.
 */
function createLogEntryElement(log, index) {
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
  html += "</div>";

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

  // Event listeners handled via delegation on container (see logs event delegation below)
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

// Logs event delegation - handles clicks on dynamically created elements
elements.logsList.addEventListener("click", (e) => {
  const loggerEl = e.target.closest(".log-logger.clickable");
  if (loggerEl) {
    const logger = loggerEl.dataset.logger;
    if (logger) {
      if (e.shiftKey) {
        toggleLoggerFilter(logger, false, true);
      } else {
        toggleLoggerFilter(logger, true, false);
      }
    }
    return;
  }

  const eventEl = e.target.closest(".log-event-name.clickable");
  if (eventEl) {
    const event = eventEl.dataset.event;
    if (event) {
      if (e.shiftKey) {
        toggleEventFilter(event, false, true);
      } else {
        toggleEventFilter(event, true, false);
      }
    }
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
      const dataUrl = `data:${result.mime_type};base64,${result.content}`;
      elements.filesystemViewer.innerHTML = `<div class="image-container"><img src="${dataUrl}" alt="${escapeHtml(displayPath)}" class="filesystem-image" /></div>`;
      state.fileContent = null;
    } else if (result.type === "binary") {
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

function renderEnvironment() {
  if (!state.hasEnvironmentData) {
    elements.environmentEmptyState.classList.remove("hidden");
    elements.environmentContent.classList.add("hidden");
    return;
  }

  elements.environmentEmptyState.classList.add("hidden");
  elements.environmentContent.classList.remove("hidden");

  const data = state.environmentData;
  let html = "";

  // System section
  if (data.system) {
    html += '<div class="environment-section">';
    html += '<h3 class="section-title">System</h3>';
    html += '<dl class="key-value-list">';
    html += `<dt>OS</dt><dd>${escapeHtml(data.system.os_name)} ${escapeHtml(data.system.os_release)}</dd>`;
    html += `<dt>Kernel</dt><dd class="mono">${escapeHtml(data.system.kernel_version)}</dd>`;
    html += `<dt>Architecture</dt><dd>${escapeHtml(data.system.architecture)}</dd>`;
    html += `<dt>Processor</dt><dd>${escapeHtml(data.system.processor)}</dd>`;
    html += `<dt>CPU Count</dt><dd>${data.system.cpu_count}</dd>`;
    html += `<dt>Memory</dt><dd>${formatBytes(data.system.memory_total_bytes)}</dd>`;
    html += `<dt>Hostname</dt><dd class="mono">${escapeHtml(data.system.hostname)}</dd>`;
    html += "</dl></div>";
  }

  // Python section
  if (data.python) {
    html += '<div class="environment-section">';
    html += '<h3 class="section-title">Python</h3>';
    html += '<dl class="key-value-list">';
    html += `<dt>Version</dt><dd class="mono">${escapeHtml(data.python.version)}</dd>`;
    if (data.python.version_info) {
      html += `<dt>Version Info</dt><dd class="mono">${JSON.stringify(data.python.version_info)}</dd>`;
    }
    html += `<dt>Implementation</dt><dd>${escapeHtml(data.python.implementation)}</dd>`;
    html += `<dt>Executable</dt><dd class="mono">${escapeHtml(data.python.executable)}</dd>`;
    html += `<dt>Prefix</dt><dd class="mono">${escapeHtml(data.python.prefix)}</dd>`;
    html += `<dt>Base Prefix</dt><dd class="mono">${escapeHtml(data.python.base_prefix)}</dd>`;
    html += `<dt>Virtualenv</dt><dd>${data.python.is_virtualenv ? "Yes" : "No"}</dd>`;
    html += "</dl></div>";
  }

  // Git section
  if (data.git) {
    html += '<div class="environment-section">';
    html += '<h3 class="section-title">Git Repository</h3>';
    html += '<dl class="key-value-list">';
    html += `<dt>Repo Root</dt><dd class="mono">${escapeHtml(data.git.repo_root)}</dd>`;
    html += `<dt>Branch</dt><dd class="mono">${escapeHtml(data.git.branch)}</dd>`;
    html += `<dt>Commit</dt><dd class="mono">${escapeHtml(data.git.commit_short)}</dd>`;
    html += `<dt>Full SHA</dt><dd class="mono small">${escapeHtml(data.git.commit_sha)}</dd>`;
    html += `<dt>Dirty</dt><dd>${data.git.is_dirty ? "Yes" : "No"}</dd>`;
    if (data.git.remotes && Object.keys(data.git.remotes).length > 0) {
      html += '<dt>Remotes</dt><dd><div class="nested-list">';
      Object.entries(data.git.remotes).forEach(([name, url]) => {
        html += `<div><span class="mono">${escapeHtml(name)}:</span> ${escapeHtml(url)}</div>`;
      });
      html += "</div></dd>";
    }
    if (data.git.tags && data.git.tags.length > 0) {
      html += `<dt>Tags</dt><dd class="mono">${data.git.tags.map((t) => escapeHtml(t)).join(", ")}</dd>`;
    }
    html += "</dl></div>";
  }

  // Container section
  if (data.container) {
    html += '<div class="environment-section">';
    html += '<h3 class="section-title">Container</h3>';
    html += '<dl class="key-value-list">';
    html += `<dt>Runtime</dt><dd>${escapeHtml(data.container.runtime)}</dd>`;
    html += `<dt>Container ID</dt><dd class="mono">${escapeHtml(data.container.container_id)}</dd>`;
    html += `<dt>Image</dt><dd class="mono">${escapeHtml(data.container.image)}</dd>`;
    if (data.container.image_digest) {
      html += `<dt>Image Digest</dt><dd class="mono small">${escapeHtml(data.container.image_digest)}</dd>`;
    }
    if (data.container.cgroup_path) {
      html += `<dt>Cgroup Path</dt><dd class="mono">${escapeHtml(data.container.cgroup_path)}</dd>`;
    }
    html += "</dl></div>";
  }

  // Environment Variables section
  if (data.env_vars && Object.keys(data.env_vars).length > 0) {
    html += '<div class="environment-section">';
    html += '<h3 class="section-title">Environment Variables</h3>';
    html += '<dl class="key-value-list">';
    Object.entries(data.env_vars).forEach(([key, value]) => {
      html += `<dt class="mono">${escapeHtml(key)}</dt><dd class="mono small">${escapeHtml(value)}</dd>`;
    });
    html += "</dl></div>";
  }

  elements.environmentData.innerHTML = html;
}

function formatBytes(bytes) {
  if (bytes === 0) {
    return "0 B";
  }
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${Math.round((bytes / k ** i) * 100) / 100} ${sizes[i]}`;
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
 * Checks if an entry is a tool result (user message containing tool_result).
 */
function isToolResult(entry) {
  if (entry.entry_type !== "user") {
    return false;
  }

  // Parse the entry to check for tool_result in message.content
  let parsed = entry.parsed;
  if (typeof parsed === "string") {
    try {
      parsed = JSON.parse(parsed);
    } catch {
      return false;
    }
  } else if (!parsed && entry.raw_json) {
    try {
      parsed = JSON.parse(entry.raw_json);
    } catch {
      return false;
    }
  }

  if (!parsed) {
    return false;
  }

  const content = parsed.message?.content;
  if (!Array.isArray(content)) {
    return false;
  }

  return content.some((item) => item.type === "tool_result");
}

/**
 * Extracts tool_use_id from an entry's nested message.content structure.
 * For tool calls: uses top-level tool_use_id or looks for content[*].id where type === "tool_use"
 * For tool results: looks for content[*].tool_use_id where type === "tool_result"
 */
function extractToolUseId(entry) {
  // First try top-level tool_use_id if it exists (tool calls have this)
  if (entry.tool_use_id) {
    return entry.tool_use_id;
  }

  // Try to get from parsed or raw_json
  let parsed = entry.parsed;
  if (typeof parsed === "string") {
    try {
      parsed = JSON.parse(parsed);
    } catch {
      return null;
    }
  } else if (!parsed && entry.raw_json) {
    try {
      parsed = JSON.parse(entry.raw_json);
    } catch {
      return null;
    }
  }

  if (!parsed) {
    return null;
  }

  // Look in message.content array
  const content = parsed.message?.content;
  if (!Array.isArray(content)) {
    return null;
  }

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
 * Preprocesses transcript entries to combine tool calls with their results.
 * Returns a new array where tool call + result pairs are merged into composite entries.
 * Each composite entry has an `isComposite` flag and `toolResult` property.
 */
function preprocessTranscriptEntries(entries) {
  const processed = [];
  const usedIndices = new Set();

  for (let i = 0; i < entries.length; i++) {
    if (usedIndices.has(i)) {
      continue;
    }

    const entry = entries[i];

    // Check if this is a tool call
    if (isToolCall(entry)) {
      const toolId = extractToolUseId(entry);

      // Look for matching result in next few entries (usually immediately after)
      let resultEntry = null;
      for (let j = i + 1; j < Math.min(i + 10, entries.length); j++) {
        if (usedIndices.has(j)) {
          continue;
        }
        const candidate = entries[j];
        if (isToolResult(candidate) && extractToolUseId(candidate) === toolId) {
          resultEntry = candidate;
          usedIndices.add(j);
          break;
        }
      }

      // Create composite entry if result found
      if (resultEntry) {
        processed.push({
          ...entry,
          isComposite: true,
          toolResult: resultEntry,
        });
      } else {
        processed.push(entry);
      }
    } else {
      processed.push(entry);
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

/**
 * Renders a composite tool call + result entry in zoom view.
 */
function renderCompositeEntryZoom(entry) {
  const toolName = entry.tool_name || "unknown";

  // Header
  elements.zoomModalType.textContent = `TOOL: ${toolName}`;
  elements.zoomModalType.className = "zoom-type-badge zoom-type-tool_use";
  elements.zoomModalTimestamp.textContent = entry.timestamp || "";

  const source = entry.transcript_source || "";
  const seq =
    entry.sequence_number !== null && entry.sequence_number !== undefined
      ? `#${entry.sequence_number}`
      : "";
  elements.zoomModalSource.textContent = source ? `${source}${seq}` : "";
  elements.zoomModalSource.style.display = source ? "" : "none";

  // Left panel: Tool Call
  elements.zoomContent.innerHTML = "";
  const callHeader = document.createElement("div");
  callHeader.className = "zoom-combined-header zoom-combined-call";
  callHeader.innerHTML = '<span class="zoom-combined-badge">CALL</span> Tool Input';
  elements.zoomContent.appendChild(callHeader);

  const callContent = formatTranscriptContent(entry);
  if (callContent.value) {
    renderContentSection(callContent, elements.zoomContent);
  }
  renderRawJsonSection(entry.parsed || entry.raw_json, "Call Raw JSON", elements.zoomContent);

  // Right panel: Tool Result
  elements.zoomDetails.innerHTML = "";
  const resultHeader = document.createElement("div");
  resultHeader.className = "zoom-combined-header zoom-combined-result";
  resultHeader.innerHTML = '<span class="zoom-combined-badge">RESULT</span> Tool Output';
  elements.zoomDetails.appendChild(resultHeader);

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

  // Tool metadata
  if (entry.tool_use_id) {
    const metaSection = document.createElement("div");
    metaSection.className = "zoom-section zoom-tool-meta";
    metaSection.innerHTML = `<div class="zoom-section-label">Tool Use ID</div><div class="zoom-text-content mono">${escapeHtml(entry.tool_use_id)}</div>`;
    elements.zoomDetails.appendChild(metaSection);
  }
}

/**
 * Renders a regular (non-composite) entry in zoom view.
 */
function renderRegularEntryZoom(entry) {
  const entryType = entry.entry_type || "unknown";

  // Header
  elements.zoomModalType.textContent = entryType.toUpperCase();
  elements.zoomModalType.className = `zoom-type-badge zoom-type-${entryType}`;
  elements.zoomModalTimestamp.textContent = entry.timestamp || "";

  const source = entry.transcript_source || "";
  const seq =
    entry.sequence_number !== null && entry.sequence_number !== undefined
      ? `#${entry.sequence_number}`
      : "";
  elements.zoomModalSource.textContent = source ? `${source}${seq}` : "";
  elements.zoomModalSource.style.display = source ? "" : "none";

  // Content panel
  elements.zoomContent.innerHTML = "";
  const content = formatTranscriptContent(entry);
  if (content.value) {
    renderContentSection(content, elements.zoomContent);
  }

  // Details panel
  elements.zoomDetails.innerHTML = "";

  if (entry.tool_name) {
    const toolSection = document.createElement("div");
    toolSection.className = "zoom-section";
    toolSection.innerHTML = `<div class="zoom-section-label">Tool</div><div class="zoom-text-content">${escapeHtml(entry.tool_name)}</div>`;
    elements.zoomDetails.appendChild(toolSection);
  }

  if (entry.tool_use_id) {
    const toolIdSection = document.createElement("div");
    toolIdSection.className = "zoom-section";
    toolIdSection.innerHTML = `<div class="zoom-section-label">Tool Use ID</div><div class="zoom-text-content">${escapeHtml(entry.tool_use_id)}</div>`;
    elements.zoomDetails.appendChild(toolIdSection);
  }

  if (entry.prompt_name) {
    const promptSection = document.createElement("div");
    promptSection.className = "zoom-section";
    promptSection.innerHTML = `<div class="zoom-section-label">Prompt</div><div class="zoom-text-content">${escapeHtml(entry.prompt_name)}</div>`;
    elements.zoomDetails.appendChild(promptSection);
  }

  renderRawJsonSection(entry.parsed || entry.raw_json, "Raw JSON", elements.zoomDetails);

  if (elements.zoomDetails.children.length === 0) {
    elements.zoomDetails.innerHTML = `<div class="zoom-empty">(no additional details)</div>`;
  }
}

/**
 * Renders a collapsible JSON tree for the zoom modal details panel.
 */
function renderZoomJsonTree(value, key, depth) {
  const node = document.createElement("div");
  node.className = "zoom-json-node";

  if (value === null) {
    node.innerHTML = key
      ? `<span class="zoom-json-key">${escapeHtml(key)}</span>: <span class="zoom-json-null">null</span>`
      : '<span class="zoom-json-null">null</span>';
    return node;
  }

  if (typeof value === "boolean") {
    node.innerHTML = key
      ? `<span class="zoom-json-key">${escapeHtml(key)}</span>: <span class="zoom-json-bool">${value}</span>`
      : `<span class="zoom-json-bool">${value}</span>`;
    return node;
  }

  if (typeof value === "number") {
    node.innerHTML = key
      ? `<span class="zoom-json-key">${escapeHtml(key)}</span>: <span class="zoom-json-number">${value}</span>`
      : `<span class="zoom-json-number">${value}</span>`;
    return node;
  }

  if (typeof value === "string") {
    // Try to parse as JSON - handle double-serialized JSON
    if (
      value.length > 1 &&
      ((value.startsWith("{") && value.endsWith("}")) ||
        (value.startsWith("[") && value.endsWith("]")))
    ) {
      try {
        const parsed = JSON.parse(value);
        // Successfully parsed - render as object/array with a "parsed" indicator
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
        // Not valid JSON, render as string
      }
    }

    // Regular string - no truncation, show full content
    const escaped = escapeHtml(value);
    const formatted = escaped.replace(/\n/g, "<br>");
    node.innerHTML = key
      ? `<span class="zoom-json-key">${escapeHtml(key)}</span>: <span class="zoom-json-string">"${formatted}"</span>`
      : `<span class="zoom-json-string">"${formatted}"</span>`;
    return node;
  }

  if (Array.isArray(value)) {
    const isExpanded = depth < 2;
    const header = document.createElement("div");
    header.className = "zoom-json-header";
    header.innerHTML = `<span class="zoom-json-toggle">${isExpanded ? "▼" : "▶"}</span>${key ? `<span class="zoom-json-key">${escapeHtml(key)}</span>: ` : ""}<span class="zoom-json-bracket">[</span><span class="zoom-json-count">${value.length} items</span><span class="zoom-json-bracket zoom-json-close-bracket" style="display: ${isExpanded ? "none" : "inline"}">]</span>`;
    node.appendChild(header);

    const children = document.createElement("div");
    children.className = "zoom-json-children";
    children.style.display = isExpanded ? "block" : "none";

    for (let i = 0; i < value.length; i++) {
      children.appendChild(renderZoomJsonTree(value[i], String(i), depth + 1));
    }

    const closeBracket = document.createElement("div");
    closeBracket.className = "zoom-json-close";
    closeBracket.innerHTML = '<span class="zoom-json-bracket">]</span>';
    closeBracket.style.display = isExpanded ? "block" : "none";
    children.appendChild(closeBracket);

    node.appendChild(children);

    header.addEventListener("click", () => {
      const isOpen = children.style.display !== "none";
      children.style.display = isOpen ? "none" : "block";
      header.querySelector(".zoom-json-toggle").textContent = isOpen ? "▶" : "▼";
      header.querySelector(".zoom-json-close-bracket").style.display = isOpen ? "inline" : "none";
    });

    return node;
  }

  if (typeof value === "object") {
    const keys = Object.keys(value);
    const isExpanded = depth < 2;
    const header = document.createElement("div");
    header.className = "zoom-json-header";
    header.innerHTML = `<span class="zoom-json-toggle">${isExpanded ? "▼" : "▶"}</span>${key ? `<span class="zoom-json-key">${escapeHtml(key)}</span>: ` : ""}<span class="zoom-json-bracket">{</span><span class="zoom-json-count">${keys.length} keys</span><span class="zoom-json-bracket zoom-json-close-bracket" style="display: ${isExpanded ? "none" : "inline"}">}</span>`;
    node.appendChild(header);

    const children = document.createElement("div");
    children.className = "zoom-json-children";
    children.style.display = isExpanded ? "block" : "none";

    for (const k of keys) {
      children.appendChild(renderZoomJsonTree(value[k], k, depth + 1));
    }

    const closeBracket = document.createElement("div");
    closeBracket.className = "zoom-json-close";
    closeBracket.innerHTML = '<span class="zoom-json-bracket">}</span>';
    closeBracket.style.display = isExpanded ? "block" : "none";
    children.appendChild(closeBracket);

    node.appendChild(children);

    header.addEventListener("click", () => {
      const isOpen = children.style.display !== "none";
      children.style.display = isOpen ? "none" : "block";
      header.querySelector(".zoom-json-toggle").textContent = isOpen ? "▶" : "▼";
      header.querySelector(".zoom-json-close-bracket").style.display = isOpen ? "inline" : "none";
    });

    return node;
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

document.addEventListener("keydown", (e) => {
  // Escape - close dialogs
  if (e.key === "Escape") {
    if (state.zoomOpen) {
      e.preventDefault();
      closeZoomModal();
      return;
    }
    if (state.shortcutsOpen) {
      e.preventDefault();
      closeShortcuts();
    }
    return;
  }

  // Handle zoom modal navigation
  if (state.zoomOpen) {
    if (e.key === "j" || e.key === "J" || e.key === "ArrowDown" || e.key === "ArrowRight") {
      e.preventDefault();
      zoomNext();
      return;
    }
    if (e.key === "k" || e.key === "K" || e.key === "ArrowUp" || e.key === "ArrowLeft") {
      e.preventDefault();
      zoomPrev();
      return;
    }
    // Other keys are blocked while zoom modal is open
    return;
  }

  // Don't process if dialog open or typing in input
  if (state.shortcutsOpen) {
    return;
  }
  const tag = e.target.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") {
    return;
  }

  // Number keys for tabs
  if (e.key >= "1" && e.key <= "5") {
    e.preventDefault();
    const views = ["sessions", "transcript", "logs", "filesystem", "environment"];
    switchView(views[Number.parseInt(e.key, 10) - 1]);
    return;
  }

  // Global shortcuts
  if (e.key === "r" || e.key === "R") {
    e.preventDefault();
    reloadBundle();
    return;
  }
  if (e.key === "d" || e.key === "D") {
    e.preventDefault();
    toggleTheme();
    return;
  }
  if (e.key === "[") {
    e.preventDefault();
    toggleSidebar();
    return;
  }
  if (e.key === "?" || (e.shiftKey && e.key === "/")) {
    e.preventDefault();
    openShortcuts();
    return;
  }
  if (e.key === "/") {
    e.preventDefault();
    focusCurrentSearch();
    return;
  }

  // J/K navigation
  if (e.key === "j" || e.key === "J") {
    e.preventDefault();
    navigateNext();
    return;
  }
  if (e.key === "k" || e.key === "K") {
    e.preventDefault();
    navigatePrev();
    return;
  }

  // Arrow keys for bundles
  if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
    e.preventDefault();
    navigateBundle(e.key === "ArrowRight" ? 1 : -1);
  }
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

const isObject = (v) => typeof v === "object" && v !== null;
const isPrimitive = (v) => v === null || (typeof v !== "object" && !Array.isArray(v));
const isSimpleArray = (v) => Array.isArray(v) && v.every(isPrimitive);

function getMarkdownPayload(value) {
  if (
    !value ||
    typeof value !== "object" ||
    Array.isArray(value) ||
    !Object.prototype.hasOwnProperty.call(value, MARKDOWN_KEY)
  ) {
    return null;
  }
  const payload = value[MARKDOWN_KEY];
  return payload && typeof payload.text === "string" && typeof payload.html === "string"
    ? payload
    : null;
}

function valueType(value) {
  if (getMarkdownPayload(value)) {
    return "markdown";
  }
  if (Array.isArray(value)) {
    return "array";
  }
  if (value === null) {
    return "null";
  }
  return typeof value;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function pathKey(path) {
  return path.join(".");
}
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
  badge.textContent =
    type === "array"
      ? `array (${value.length})`
      : type === "object" && value !== null
        ? `object (${Object.keys(value).length})`
        : type;
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
      renderItems(state.currentItems);
    });

    header.appendChild(controls);
  }

  const childrenContainer = document.createElement("div");
  childrenContainer.className = "tree-children";

  if (!hasChildren) {
    childrenContainer.innerHTML = '<span class="muted">(empty)</span>';
  } else if (shouldOpen(path, depth)) {
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
          childrenContainer.appendChild(
            renderTree(child, path.concat(String(i)), depth + 1, `[${i}]`)
          );
        });
      }
    } else {
      Object.entries(value).forEach(([key, child]) => {
        childrenContainer.appendChild(renderTree(child, path.concat(key), depth + 1, key));
      });
    }
  } else {
    childrenContainer.style.display = "none";
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
