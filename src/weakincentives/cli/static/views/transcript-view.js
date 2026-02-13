// ============================================================================
// Transcript View - Handles transcript display, filtering, and interaction
// ============================================================================

import { createActiveFilter, createFilterChip } from "../components/filter-chips.js";
import { VirtualScroller } from "../components/virtual-scroller.js";
import { escapeHtml } from "../lib.js";

// ============================================================================
// Query building
// ============================================================================

function buildTranscriptQueryParams(state, offset = 0) {
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

// ============================================================================
// Transcript entry preprocessing (nest tool_result under tool_use)
// ============================================================================

function indexToolResults(entries) {
  const map = new Map();
  for (const entry of entries) {
    if (entry.entry_type === "tool_result" && entry.tool_use_id) {
      map.set(entry.tool_use_id, entry);
    }
  }
  return map;
}

function isConsumedResult(entry, consumed) {
  return entry.entry_type === "tool_result" && entry.tool_use_id && consumed.has(entry.tool_use_id);
}

function tryNestToolUse(entry, resultsByCallId, consumed) {
  if (entry.entry_type !== "tool_use" || !entry.tool_use_id) {
    return null;
  }
  const result = resultsByCallId.get(entry.tool_use_id);
  if (!result) {
    return null;
  }
  consumed.add(entry.tool_use_id);
  return { ...entry, isComposite: true, toolResult: result };
}

/**
 * Match tool_result entries to their tool_use by tool_use_id and nest them.
 * Consumed tool_result entries are removed from the flat list.  Unmatched
 * tool_result entries (e.g. when the tool_use was filtered out) stay as-is.
 */
export function preprocessTranscriptEntries(entries) {
  const resultsByCallId = indexToolResults(entries);
  const consumed = new Set();
  const processed = [];

  for (const entry of entries) {
    if (isConsumedResult(entry, consumed)) {
      continue;
    }
    const nested = tryNestToolUse(entry, resultsByCallId, consumed);
    processed.push(nested || entry);
  }

  return processed;
}

// ============================================================================
// Content formatting
// ============================================================================

function tryParseJson(str) {
  if (typeof str !== "string" || str.length < 2) {
    return null;
  }
  const ch = str[0];
  if (ch !== "{" && ch !== "[") {
    return null;
  }
  try {
    return JSON.parse(str);
  } catch {
    return null;
  }
}

function formatStringContent(entry) {
  if (entry.content_html) {
    return { kind: "markdown", value: entry.content, html: entry.content_html };
  }
  const parsed = tryParseJson(entry.content);
  if (parsed !== null) {
    return { kind: "json", value: JSON.stringify(parsed, null, 2) };
  }
  return { kind: "text", value: entry.content };
}

function prettyJson(value) {
  if (typeof value === "string") {
    const obj = tryParseJson(value);
    return obj !== null ? JSON.stringify(obj, null, 2) : value;
  }
  return JSON.stringify(value, null, 2);
}

export function formatTranscriptContent(entry) {
  if (entry.content !== null && entry.content !== undefined) {
    if (typeof entry.content === "string") {
      return formatStringContent(entry);
    }
    return { kind: "json", value: JSON.stringify(entry.content, null, 2) };
  }
  if (entry.parsed !== null && entry.parsed !== undefined) {
    return { kind: "json", value: prettyJson(entry.parsed) };
  }
  if (entry.raw_json !== null && entry.raw_json !== undefined) {
    return { kind: "json", value: prettyJson(entry.raw_json) };
  }
  return { kind: "text", value: "" };
}

// ============================================================================
// DOM rendering helpers
// ============================================================================

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
  if (content.kind === "markdown" && content.html) {
    return `<div class="transcript-message transcript-markdown">${content.html}</div>`;
  }
  return `<div class="transcript-message">${escapeHtml(content.value)}</div>`;
}

function safeParseParsed(entry) {
  const raw = entry.parsed || entry.raw_json;
  if (!raw) {
    return null;
  }
  if (typeof raw === "object") {
    return raw;
  }
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function createToolResultHtml(toolResult) {
  if (!toolResult) {
    return "";
  }
  const parsed = safeParseParsed(toolResult);
  const output = parsed?.output;
  if (output) {
    return `<div class="transcript-nested-result"><div class="transcript-result-divider">↓ result</div><pre class="transcript-json">${escapeHtml(prettyJson(output))}</pre></div>`;
  }
  const resultContent = formatTranscriptContent(toolResult);
  return `<div class="transcript-nested-result"><div class="transcript-result-divider">↓ result</div>${createContentHtml(resultContent, "(no result)")}</div>`;
}

function createPayloadHtml(entry) {
  const raw = entry.parsed || entry.raw_json;
  if (!raw) {
    return "";
  }
  const text = prettyJson(raw);
  if (!text) {
    return "";
  }
  return `<div class="transcript-payload"><div class="transcript-payload-label">Payload</div><pre>${escapeHtml(text)}</pre></div>`;
}

function createCompositeInputHtml(entry) {
  const useParsed = safeParseParsed(entry);
  const resultParsed = entry.toolResult ? safeParseParsed(entry.toolResult) : null;
  const input = resultParsed?.input || useParsed?.input;
  if (!input) {
    return "";
  }
  return `<div class="transcript-tool-input"><div class="transcript-input-label">↓ input</div><pre class="transcript-json">${escapeHtml(prettyJson(input))}</pre></div>`;
}

export function createTranscriptEntryElement(entry, index) {
  const entryType = entry.entry_type || "unknown";
  const role = entry.role || "";
  const cssClass = role ? `role-${role}` : `type-${entryType}`;

  const container = document.createElement("div");
  container.className = `transcript-entry ${cssClass}${entry.isComposite ? " combined" : ""}`;
  container.dataset.entryIndex = index;

  const headerHtml = `<div class="transcript-header">${createTranscriptTypeHtml(entry, entryType)}${createTranscriptMetadataHtml(entry)}</div>`;

  if (entry.isComposite) {
    container.innerHTML =
      headerHtml + createCompositeInputHtml(entry) + createToolResultHtml(entry.toolResult);
    return container;
  }

  const content = formatTranscriptContent(entry);
  container.innerHTML = headerHtml + createContentHtml(content) + createPayloadHtml(entry);

  return container;
}

// ============================================================================
// Transcript View initialization
// ============================================================================

/**
 * Initializes the transcript view. Wires up DOM events and manages transcript data.
 *
 * @param {object} deps - Dependencies
 * @param {object} deps.state - The shared application state
 * @param {function} deps.fetchJSON - API fetch helper
 * @param {function} deps.showToast - Toast notification helper
 * @returns {{ loadTranscript, loadTranscriptFacets, loadMoreTranscript, scrollBy, initVirtualScroller, reset }}
 */
export function initTranscriptView({ state, fetchJSON, showToast }) {
  const els = {
    search: document.getElementById("transcript-search"),
    clearFilters: document.getElementById("transcript-clear-filters"),
    showing: document.getElementById("transcript-showing"),
    copy: document.getElementById("transcript-copy"),
    scrollBottom: document.getElementById("transcript-scroll-bottom"),
    list: document.getElementById("transcript-list"),
    sourceFilter: document.getElementById("transcript-source-filter"),
    sourceChips: document.getElementById("transcript-source-chips"),
    typeFilter: document.getElementById("transcript-type-filter"),
    typeChips: document.getElementById("transcript-type-chips"),
    activeFilters: document.getElementById("transcript-active-filters"),
    activeFiltersGroup: document.getElementById("transcript-active-filters-group"),
  };

  let searchTimeout = null;

  // -- Data loading --

  function processEntries(entries, append) {
    if (append) {
      const rawEntries = [...state.transcriptRawEntries, ...entries];
      state.transcriptRawEntries = rawEntries;
      return preprocessTranscriptEntries(rawEntries);
    }
    state.transcriptRawEntries = entries;
    return preprocessTranscriptEntries(entries);
  }

  function updateDisplay() {
    if (state.transcriptScroller) {
      state.transcriptScroller.setData(
        state.transcriptEntries,
        state.transcriptTotalCount,
        state.transcriptHasMore
      );
      renderEmptyState();
    } else {
      renderTranscript();
    }
    updateStats();
  }

  function showError(message) {
    els.list.innerHTML = `<p class="muted">Failed to load transcript: ${escapeHtml(message)}</p>`;
  }

  function applyResult(result, append) {
    state.transcriptEntries = processEntries(result.entries || [], append);
    state.transcriptTotalCount = result.total || state.transcriptRawEntries.length;
    state.transcriptHasMore = state.transcriptRawEntries.length < state.transcriptTotalCount;
    updateDisplay();
  }

  async function fetchTranscript(append, isCurrentRequest) {
    const offset = append ? state.transcriptRawEntries.length : 0;
    const result = await fetchJSON(`/api/transcript?${buildTranscriptQueryParams(state, offset)}`);
    if (isCurrentRequest()) {
      state.transcriptLoadRetries = 0;
      applyResult(result, append);
    }
  }

  function handleTranscriptError(error, isCurrentRequest) {
    if (isCurrentRequest()) {
      state.transcriptLoadRetries++;
      showError(error.message);
    }
  }

  function handleTranscriptDone(isCurrentRequest) {
    if (isCurrentRequest()) {
      state.transcriptLoading = false;
    }
  }

  async function loadTranscript(append = false) {
    const requestId = ++state.transcriptRequestId;
    const isCurrentRequest = () => requestId === state.transcriptRequestId;
    if (!append) {
      state.transcriptLoadRetries = 0;
    }
    state.transcriptLoading = true;
    try {
      await fetchTranscript(append, isCurrentRequest);
    } catch (error) {
      handleTranscriptError(error, isCurrentRequest);
    } finally {
      handleTranscriptDone(isCurrentRequest);
    }
  }

  function loadMoreTranscript() {
    return loadTranscript(true);
  }

  async function loadTranscriptFacets() {
    try {
      state.transcriptFacets = await fetchJSON("/api/transcript/facets");
      renderFilterChips();
    } catch (error) {
      console.warn("Failed to load transcript facets:", error);
    }
  }

  // -- Stats --

  function updateStats() {
    let status = `Showing ${state.transcriptEntries.length}`;
    if (state.transcriptHasMore) {
      status += ` of ${state.transcriptTotalCount}`;
    }
    els.showing.textContent = status;
  }

  // -- Filter chips --

  function toggleSourceFilter(name, include, exclude) {
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
    renderFilterChips();
    loadTranscript(false);
  }

  function toggleTypeFilter(name, include, exclude) {
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
    renderFilterChips();
    loadTranscript(false);
  }

  function renderFilterChips() {
    const sourceFilter = state.transcriptSourceChipFilter.toLowerCase();
    const typeFilter = state.transcriptTypeChipFilter.toLowerCase();

    els.sourceChips.innerHTML = "";
    (state.transcriptFacets.sources || [])
      .filter((item) => !sourceFilter || item.name.toLowerCase().includes(sourceFilter))
      .forEach((item) => {
        const chip = createFilterChip(
          item.name,
          item.count,
          state.transcriptIncludeSources.has(item.name),
          state.transcriptExcludeSources.has(item.name),
          (name, include, exclude) => toggleSourceFilter(name, include, exclude)
        );
        els.sourceChips.appendChild(chip);
      });

    els.typeChips.innerHTML = "";
    (state.transcriptFacets.entry_types || [])
      .filter((item) => !typeFilter || item.name.toLowerCase().includes(typeFilter))
      .forEach((item) => {
        const chip = createFilterChip(
          item.name,
          item.count,
          state.transcriptIncludeTypes.has(item.name),
          state.transcriptExcludeTypes.has(item.name),
          (name, include, exclude) => toggleTypeFilter(name, include, exclude)
        );
        els.typeChips.appendChild(chip);
      });

    renderActiveFilters();
  }

  function renderActiveFilters() {
    const hasFilters =
      state.transcriptIncludeSources.size > 0 ||
      state.transcriptExcludeSources.size > 0 ||
      state.transcriptIncludeTypes.size > 0 ||
      state.transcriptExcludeTypes.size > 0;

    els.activeFiltersGroup.style.display = hasFilters ? "flex" : "none";
    if (!hasFilters) {
      return;
    }

    els.activeFilters.innerHTML = "";

    state.transcriptIncludeSources.forEach((name) => {
      els.activeFilters.appendChild(
        createActiveFilter("source", name, false, () => toggleSourceFilter(name, false, false))
      );
    });

    state.transcriptExcludeSources.forEach((name) => {
      els.activeFilters.appendChild(
        createActiveFilter("source", name, true, () => toggleSourceFilter(name, false, false))
      );
    });

    state.transcriptIncludeTypes.forEach((name) => {
      els.activeFilters.appendChild(
        createActiveFilter("entry_type", name, false, () => toggleTypeFilter(name, false, false))
      );
    });

    state.transcriptExcludeTypes.forEach((name) => {
      els.activeFilters.appendChild(
        createActiveFilter("entry_type", name, true, () => toggleTypeFilter(name, false, false))
      );
    });
  }

  // -- Rendering --

  function renderEmptyState() {
    if (state.transcriptEntries.length === 0) {
      if (state.transcriptScroller) {
        state.transcriptScroller.reset();
      }
      els.list.innerHTML = '<div class="logs-empty">No transcript entries match filters</div>';
    }
  }

  function renderTranscript() {
    if (state.transcriptScroller) {
      if (state.transcriptEntries.length === 0) {
        renderEmptyState();
      } else {
        state.transcriptScroller.setData(
          state.transcriptEntries,
          state.transcriptTotalCount,
          state.transcriptHasMore
        );
      }
      return;
    }

    els.list.innerHTML = "";

    if (state.transcriptEntries.length === 0) {
      els.list.innerHTML = '<div class="logs-empty">No transcript entries match filters</div>';
      return;
    }

    state.transcriptEntries.forEach((entry, index) => {
      els.list.appendChild(createTranscriptEntryElement(entry, index));
    });
  }

  // -- Virtual scroller --

  function initVirtualScroller() {
    if (state.transcriptScroller) {
      state.transcriptScroller.destroy();
    }

    state.transcriptScroller = new VirtualScroller({
      container: els.list,
      estimatedItemHeight: 120,
      bufferSize: 15,
      renderItem: createTranscriptEntryElement,
      onLoadMore: loadMoreTranscript,
      onLoadError: (error) =>
        showToast(`Failed to load more transcript entries: ${error.message}`, "error"),
    });
  }

  // -- Scroll helper --

  function scrollBy(delta) {
    const entries = els.list.querySelectorAll(".transcript-entry");
    if (entries.length === 0) {
      return;
    }
    const scrollHeight = entries[0].offsetHeight;
    els.list.scrollBy({ top: scrollHeight * delta, behavior: "smooth" });
  }

  // -- Debounced search --

  function debouncedSearch() {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => loadTranscript(false), 300);
  }

  // -- Event delegation --

  function handleSourceClick(e, sourceEl) {
    const src = sourceEl.dataset.source;
    if (!src) {
      return;
    }
    toggleSourceFilter(src, !e.shiftKey, e.shiftKey);
  }

  function handleTypeClick(e, typeEl) {
    const typ = typeEl.dataset.type;
    if (!typ) {
      return;
    }
    toggleTypeFilter(typ, !e.shiftKey, e.shiftKey);
  }

  // -- Wire up DOM events --

  els.list.addEventListener("click", (e) => {
    const sourceEl = e.target.closest(".transcript-source.clickable");
    if (sourceEl) {
      handleSourceClick(e, sourceEl);
      return;
    }
    const typeEl = e.target.closest(".transcript-type.clickable");
    if (typeEl) {
      handleTypeClick(e, typeEl);
    }
  });

  els.search.addEventListener("input", () => {
    state.transcriptSearch = els.search.value;
    debouncedSearch();
  });

  els.sourceFilter.addEventListener("input", () => {
    state.transcriptSourceChipFilter = els.sourceFilter.value;
    renderFilterChips();
  });

  els.typeFilter.addEventListener("input", () => {
    state.transcriptTypeChipFilter = els.typeFilter.value;
    renderFilterChips();
  });

  els.clearFilters.addEventListener("click", () => {
    state.transcriptSearch = "";
    state.transcriptIncludeSources.clear();
    state.transcriptExcludeSources.clear();
    state.transcriptIncludeTypes.clear();
    state.transcriptExcludeTypes.clear();
    state.transcriptSourceChipFilter = "";
    state.transcriptTypeChipFilter = "";

    els.search.value = "";
    els.sourceFilter.value = "";
    els.typeFilter.value = "";

    renderFilterChips();
    loadTranscript(false);
  });

  els.copy.addEventListener("click", async () => {
    const text = JSON.stringify(state.transcriptEntries, null, 2);
    try {
      await navigator.clipboard.writeText(text);
      showToast("Copied transcript entries", "success");
    } catch {
      showToast("Failed to copy", "error");
    }
  });

  els.scrollBottom.addEventListener("click", () => {
    if (state.transcriptScroller) {
      state.transcriptScroller.scrollToBottom();
    } else {
      els.list.scrollTop = els.list.scrollHeight;
    }
  });

  function reset() {
    clearTimeout(searchTimeout);
    searchTimeout = null;
    // Invalidate any in-flight requests by bumping the request ID
    state.transcriptRequestId++;
    state.transcriptLoading = false;
  }

  return {
    loadTranscript,
    loadTranscriptFacets,
    loadMoreTranscript,
    scrollBy,
    initVirtualScroller,
    reset,
    get list() {
      return els.list;
    },
  };
}
