// ============================================================================
// Logs View - Handles log display, filtering, and interaction
// ============================================================================

import { createActiveFilter, createFilterChip } from "../components/filter-chips.js";
import { VirtualScroller } from "../components/virtual-scroller.js";
import { escapeHtml } from "../lib.js";

// ============================================================================
// Query building
// ============================================================================

function buildLogsQueryParams(state, offset = 0) {
  const params = new URLSearchParams();
  params.set("offset", offset);
  params.set("limit", state.logsLimit);

  if (state.logsLevels.size > 0 && state.logsLevels.size < 4) {
    params.set("level", Array.from(state.logsLevels).join(","));
  }

  if (state.logsSearch.trim()) {
    params.set("search", state.logsSearch.trim());
  }

  if (state.logsIncludeLoggers.size > 0) {
    params.set("logger", Array.from(state.logsIncludeLoggers).join(","));
  }
  if (state.logsExcludeLoggers.size > 0) {
    params.set("exclude_logger", Array.from(state.logsExcludeLoggers).join(","));
  }

  if (state.logsIncludeEvents.size > 0) {
    params.set("event", Array.from(state.logsIncludeEvents).join(","));
  }
  if (state.logsExcludeEvents.size > 0) {
    params.set("exclude_event", Array.from(state.logsExcludeEvents).join(","));
  }

  return params.toString();
}

// ============================================================================
// DOM rendering helpers
// ============================================================================

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

function createLogEntryElement(log, index) {
  const entry = document.createElement("div");
  entry.className = `log-entry log-${(log.level || "INFO").toLowerCase()}`;
  entry.dataset.index = index;
  entry.innerHTML = createLogHeaderHtml(log) + createLogBodyHtml(log);
  return entry;
}

// ============================================================================
// Logs View initialization
// ============================================================================

/**
 * Initializes the logs view. Wires up DOM events and manages log data.
 *
 * @param {object} deps - Dependencies
 * @param {object} deps.state - The shared application state
 * @param {function} deps.fetchJSON - API fetch helper
 * @param {function} deps.showToast - Toast notification helper
 * @returns {{ loadLogs, loadLogFacets, loadMoreLogs, scrollBy, initVirtualScroller, reset }}
 */
export function initLogsView({ state, fetchJSON, showToast }) {
  const els = {
    search: document.getElementById("logs-search"),
    clearFilters: document.getElementById("logs-clear-filters"),
    showing: document.getElementById("logs-showing"),
    copy: document.getElementById("logs-copy"),
    scrollBottom: document.getElementById("logs-scroll-bottom"),
    list: document.getElementById("logs-list"),
    loggerFilter: document.getElementById("logs-logger-filter"),
    loggerChips: document.getElementById("logs-logger-chips"),
    eventFilter: document.getElementById("logs-event-filter"),
    eventChips: document.getElementById("logs-event-chips"),
    activeFilters: document.getElementById("logs-active-filters"),
    activeFiltersGroup: document.getElementById("logs-active-filters-group"),
  };

  let searchTimeout = null;

  // -- Data loading --

  function updateData(entries, append) {
    if (append) {
      state.filteredLogs = state.filteredLogs.concat(entries);
    } else {
      state.filteredLogs = entries;
    }
  }

  function updateDisplay(entries, append) {
    if (state.logsScroller) {
      if (append) {
        state.logsScroller.appendData(entries, state.logsTotalCount, state.logsHasMore);
      } else {
        state.logsScroller.setData(state.filteredLogs, state.logsTotalCount, state.logsHasMore);
      }
      renderEmptyState();
    } else {
      renderLogs();
    }
    updateStats();
  }

  function showError(message) {
    els.list.innerHTML = `<p class="muted">Failed to load logs: ${message}</p>`;
  }

  function applyResult(result, append) {
    const entries = result.entries || [];
    updateData(entries, append);
    state.logsTotalCount = result.total || state.filteredLogs.length;
    state.logsHasMore = state.filteredLogs.length < state.logsTotalCount;
    updateDisplay(entries, append);
  }

  async function fetchLogs(append, isCurrentRequest) {
    const offset = append ? state.filteredLogs.length : 0;
    const result = await fetchJSON(`/api/logs?${buildLogsQueryParams(state, offset)}`);
    if (isCurrentRequest()) {
      applyResult(result, append);
    }
  }

  async function loadLogs(append = false) {
    const requestId = ++state.logsRequestId;
    const isCurrentRequest = () => requestId === state.logsRequestId;
    try {
      state.logsLoading = true;
      await fetchLogs(append, isCurrentRequest);
    } catch (error) {
      if (isCurrentRequest()) {
        showError(error.message);
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

  async function loadLogFacets() {
    try {
      state.logsFacets = await fetchJSON("/api/logs/facets");
      renderFilterChips();
    } catch (error) {
      console.warn("Failed to load log facets:", error);
    }
  }

  // -- Stats --

  function updateStats() {
    let status = `Showing ${state.filteredLogs.length}`;
    if (state.logsHasMore) {
      status += ` of ${state.logsTotalCount}`;
    }
    els.showing.textContent = status;
  }

  // -- Filter chips --

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
    renderFilterChips();
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
    renderFilterChips();
    loadLogs(false);
  }

  function renderFilterChips() {
    const loggerFilter = state.logsLoggerChipFilter.toLowerCase();
    const eventFilter = state.logsEventChipFilter.toLowerCase();

    els.loggerChips.innerHTML = "";
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
        els.loggerChips.appendChild(chip);
      });

    els.eventChips.innerHTML = "";
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
        els.eventChips.appendChild(chip);
      });

    renderActiveFilters();
  }

  function renderActiveFilters() {
    const hasFilters =
      state.logsIncludeLoggers.size > 0 ||
      state.logsExcludeLoggers.size > 0 ||
      state.logsIncludeEvents.size > 0 ||
      state.logsExcludeEvents.size > 0;

    els.activeFiltersGroup.style.display = hasFilters ? "flex" : "none";

    if (!hasFilters) {
      return;
    }

    els.activeFilters.innerHTML = "";

    state.logsIncludeLoggers.forEach((name) => {
      els.activeFilters.appendChild(
        createActiveFilter("logger", name, false, () => toggleLoggerFilter(name, false, false))
      );
    });

    state.logsExcludeLoggers.forEach((name) => {
      els.activeFilters.appendChild(
        createActiveFilter("logger", name, true, () => toggleLoggerFilter(name, false, false))
      );
    });

    state.logsIncludeEvents.forEach((name) => {
      els.activeFilters.appendChild(
        createActiveFilter("event", name, false, () => toggleEventFilter(name, false, false))
      );
    });

    state.logsExcludeEvents.forEach((name) => {
      els.activeFilters.appendChild(
        createActiveFilter("event", name, true, () => toggleEventFilter(name, false, false))
      );
    });
  }

  // -- Rendering --

  function renderEmptyState() {
    if (state.filteredLogs.length === 0) {
      if (state.logsScroller) {
        state.logsScroller.reset();
      }
      els.list.innerHTML = '<div class="logs-empty">No log entries match filters</div>';
    }
  }

  function renderLogs() {
    if (state.logsScroller) {
      if (state.filteredLogs.length === 0) {
        renderEmptyState();
      } else {
        state.logsScroller.setData(state.filteredLogs, state.logsTotalCount, state.logsHasMore);
      }
      return;
    }

    els.list.innerHTML = "";

    if (state.filteredLogs.length === 0) {
      els.list.innerHTML = '<div class="logs-empty">No log entries match filters</div>';
      return;
    }

    state.filteredLogs.forEach((log, index) => {
      els.list.appendChild(createLogEntryElement(log, index));
    });
  }

  // -- Virtual scroller --

  function initVirtualScroller() {
    if (state.logsScroller) {
      state.logsScroller.destroy();
    }

    state.logsScroller = new VirtualScroller({
      container: els.list,
      estimatedItemHeight: 80,
      bufferSize: 20,
      renderItem: createLogEntryElement,
      onLoadMore: loadMoreLogs,
      onLoadError: (error) => showToast(`Failed to load more logs: ${error.message}`, "error"),
    });
  }

  // -- Scroll helper --

  function scrollBy(delta) {
    const entries = els.list.querySelectorAll(".log-entry");
    if (entries.length === 0) {
      return;
    }
    const scrollHeight = entries[0].offsetHeight;
    els.list.scrollBy({ top: scrollHeight * delta, behavior: "smooth" });
  }

  // -- Debounced search --

  function debouncedSearch() {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => loadLogs(false), 300);
  }

  // -- Event delegation --

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

  // -- Wire up DOM events --

  els.list.addEventListener("click", (e) => {
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

  els.search.addEventListener("input", () => {
    state.logsSearch = els.search.value;
    debouncedSearch();
  });

  els.loggerFilter.addEventListener("input", () => {
    state.logsLoggerChipFilter = els.loggerFilter.value;
    renderFilterChips();
  });

  els.eventFilter.addEventListener("input", () => {
    state.logsEventChipFilter = els.eventFilter.value;
    renderFilterChips();
  });

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

  els.clearFilters.addEventListener("click", () => {
    state.logsSearch = "";
    state.logsLevels = new Set(["DEBUG", "INFO", "WARNING", "ERROR"]);
    state.logsIncludeLoggers.clear();
    state.logsExcludeLoggers.clear();
    state.logsIncludeEvents.clear();
    state.logsExcludeEvents.clear();
    state.logsLoggerChipFilter = "";
    state.logsEventChipFilter = "";

    els.search.value = "";
    els.loggerFilter.value = "";
    els.eventFilter.value = "";
    document.querySelectorAll(".level-checkbox input").forEach((cb) => {
      cb.checked = true;
    });

    renderFilterChips();
    loadLogs(false);
  });

  els.copy.addEventListener("click", async () => {
    const text = JSON.stringify(state.filteredLogs, null, 2);
    try {
      await navigator.clipboard.writeText(text);
      showToast("Copied filtered logs", "success");
    } catch {
      showToast("Failed to copy", "error");
    }
  });

  els.scrollBottom.addEventListener("click", () => {
    if (state.logsScroller) {
      state.logsScroller.scrollToBottom();
    } else {
      els.list.scrollTop = els.list.scrollHeight;
    }
  });

  function reset() {
    clearTimeout(searchTimeout);
    searchTimeout = null;
    // Invalidate any in-flight requests by bumping the request ID
    state.logsRequestId++;
    state.logsLoading = false;
  }

  return {
    loadLogs,
    loadLogFacets,
    loadMoreLogs,
    scrollBy,
    initVirtualScroller,
    reset,
  };
}
