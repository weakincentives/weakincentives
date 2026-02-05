// ============================================================================
// Store - Centralized application state
//
// All view modules share a single state object returned by createStore().
// Each view receives the store at initialization and accesses state via
// store.getState(). State is a plain mutable object for performance in a
// browser context — the centralization ensures all modules read/write the
// same source of truth.
// ============================================================================

/**
 * Creates the initial application state.
 * @returns {object}
 */
export function createInitialState() {
  return {
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
    transcriptRawEntries: [],
    transcriptEntries: [],
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
    transcriptRequestId: 0,
    transcriptLoadRetries: 0,
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
    logsLimit: 50,
    logsTotalCount: 0,
    logsHasMore: false,
    logsLoading: false,
    logsRequestId: 0,
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
    zoomType: null,
    zoomIndex: -1,
    zoomEntry: null,
    // Virtual scrollers (initialized after DOM ready)
    logsScroller: null,
    transcriptScroller: null,
  };
}

/**
 * Creates a store that holds centralized application state.
 *
 * @returns {{ getState: () => object }}
 */
export function createStore() {
  const state = createInitialState();

  function getState() {
    return state;
  }

  return { getState };
}
