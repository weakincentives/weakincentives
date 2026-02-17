// ============================================================================
// State - Centralized application state definition
//
// All view modules share a single state object created by createInitialState().
// Each view receives the state object at initialization. State is a plain
// mutable object — the centralization ensures all modules read/write the same
// source of truth while keeping the state shape defined in one place.
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
    theme: localStorage.getItem("wink-theme") || "dark",
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
    hasTranscript: true,
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
    // Command palette
    commandPaletteOpen: false,
    // Sidebar resize
    sidebarWidth: Number.parseInt(localStorage.getItem("wink-sidebar-width"), 10) || 280,
    // Virtual scrollers (initialized after DOM ready)
    logsScroller: null,
    transcriptScroller: null,
  };
}
