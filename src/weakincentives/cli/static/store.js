// ============================================================================
// Store - Centralized state management with dispatch/subscribe pattern
//
// Mirrors the event-driven pattern from the Python codebase (sessions and
// reducers). State changes flow through typed actions processed by reducers.
// Views subscribe to state slices and receive updates when those slices change.
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
 * Creates a store with dispatch/subscribe pattern.
 *
 * Actions are plain objects with a `type` string field. Reducers are functions
 * that receive (state, action) and mutate state directly (the state object is
 * mutable for performance in a browser context, unlike the frozen Python
 * dataclasses). Subscribers are notified after each dispatch.
 *
 * @returns {{ getState, dispatch, subscribe }}
 */
export function createStore() {
  const state = createInitialState();
  const subscribers = new Map();
  let nextId = 0;

  function getState() {
    return state;
  }

  /**
   * Dispatch an action. Notifies all subscribers after state is updated.
   * Callers mutate state directly before or via the action handler in views.
   * The dispatch call serves as the notification mechanism.
   *
   * @param {{ type: string, [key: string]: unknown }} action
   */
  function dispatch(action) {
    for (const callback of subscribers.values()) {
      callback(action, state);
    }
  }

  /**
   * Subscribe to state changes. Returns an unsubscribe function.
   *
   * @param {function({ type: string }, object): void} callback
   * @returns {function(): void} unsubscribe
   */
  function subscribe(callback) {
    const id = nextId++;
    subscribers.set(id, callback);
    return () => subscribers.delete(id);
  }

  return { getState, dispatch, subscribe };
}
