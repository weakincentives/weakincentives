// ============================================================================
// Transcript View - Handles transcript display, filtering, and interaction
// ============================================================================

import { createActiveFilter, createFilterChip } from "../components/filter-chips.js";
import { VirtualScroller } from "../components/virtual-scroller.js";
import { escapeHtml } from "../lib.js";

const ZOOM_BUTTON_SVG = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 3 21 3 21 9"></polyline><polyline points="9 21 3 21 3 15"></polyline><line x1="21" y1="3" x2="14" y2="10"></line><line x1="3" y1="21" x2="10" y2="14"></line></svg>`;

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
// Transcript entry preprocessing (tool call + result combining)
// ============================================================================

function isToolCall(entry) {
  return entry.entry_type === "assistant" && entry.tool_name && entry.tool_name !== "";
}

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

function getMessageContent(parsed) {
  const content = parsed?.message?.content;
  return Array.isArray(content) ? content : null;
}

function isToolResult(entry) {
  if (entry.entry_type !== "user") {
    return false;
  }
  const content = getMessageContent(parseEntryData(entry));
  return content ? content.some((item) => item.type === "tool_result") : false;
}

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

function extractToolUseId(entry) {
  if (entry.tool_use_id) {
    return entry.tool_use_id;
  }
  const content = getMessageContent(parseEntryData(entry));
  return content ? findToolIdInContent(content) : null;
}

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

export function preprocessTranscriptEntries(entries) {
  const processed = [];
  const usedIndices = new Set();
  for (let i = 0; i < entries.length; i++) {
    if (!usedIndices.has(i)) {
      processed.push(processEntry(entries[i], entries, i, usedIndices));
    }
  }
  return processed;
}

// ============================================================================
// Content formatting
// ============================================================================

export function formatTranscriptContent(entry) {
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

export function createTranscriptEntryElement(entry, index) {
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

// ============================================================================
// Transcript View initialization
// ============================================================================

/**
 * Initializes the transcript view. Subscribes to store and wires up DOM events.
 *
 * @param {object} deps - Dependencies
 * @param {object} deps.store - The application store
 * @param {function} deps.fetchJSON - API fetch helper
 * @param {function} deps.showToast - Toast notification helper
 * @returns {{ loadTranscript, loadTranscriptFacets, loadMoreTranscript, scrollBy, initVirtualScroller }}
 */
export function initTranscriptView({ store, fetchJSON, showToast }) {
  const state = store.getState();

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
    els.list.innerHTML = `<p class="muted">Failed to load transcript: ${message}</p>`;
  }

  function applyResult(result, append) {
    state.transcriptEntries = processEntries(result.entries || [], append);
    state.transcriptTotalCount = result.total || state.transcriptRawEntries.length;
    state.transcriptHasMore = state.transcriptRawEntries.length < state.transcriptTotalCount;
    updateDisplay();
  }

  async function fetchTranscript(append, isCurrentRequest) {
    const offset = append ? state.transcriptEntries.length : 0;
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

  return {
    loadTranscript,
    loadTranscriptFacets,
    loadMoreTranscript,
    scrollBy,
    initVirtualScroller,
    get list() {
      return els.list;
    },
  };
}
