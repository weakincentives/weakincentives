// ============================================================================
// Zoom Modal - Detail view for transcript and log entries
// ============================================================================

import { renderZoomJsonTree } from "../components/zoom-json-tree.js";
import { escapeHtml } from "../lib.js";
import { formatTranscriptContent } from "./transcript-view.js";

const MAX_TRANSCRIPT_LOAD_RETRIES = 3;

// ============================================================================
// Content rendering helpers
// ============================================================================

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

// ============================================================================
// Zoom modal rendering
// ============================================================================

function setZoomHeader(els, typeText, typeClass, entry) {
  els.modalType.textContent = typeText;
  els.modalType.className = `zoom-type-badge ${typeClass}`;
  els.modalTimestamp.textContent = entry.timestamp || "";
  const source = entry.transcript_source || "";
  const seq = entry.sequence_number != null ? `#${entry.sequence_number}` : "";
  els.modalSource.textContent = source ? `${source}${seq}` : "";
  els.modalSource.style.display = source ? "" : "none";
}

function renderCompositeEntryZoom(els, entry) {
  setZoomHeader(els, `TOOL: ${entry.tool_name || "unknown"}`, "zoom-type-tool_use", entry);

  els.content.innerHTML = "";
  els.content.appendChild(createCombinedHeader("CALL", "Tool Input", "zoom-combined-call"));
  const callContent = formatTranscriptContent(entry);
  if (callContent.value) {
    renderContentSection(callContent, els.content);
  }
  renderRawJsonSection(entry.parsed || entry.raw_json, "Call Raw JSON", els.content);

  els.details.innerHTML = "";
  els.details.appendChild(createCombinedHeader("RESULT", "Tool Output", "zoom-combined-result"));
  if (entry.toolResult) {
    const resultContent = formatTranscriptContent(entry.toolResult);
    if (resultContent.value) {
      renderContentSection(resultContent, els.details);
    }
    renderRawJsonSection(
      entry.toolResult.parsed || entry.toolResult.raw_json,
      "Result Raw JSON",
      els.details
    );
  }
  if (entry.tool_use_id) {
    const metaSection = createZoomSection(
      "Tool Use ID",
      `<span class="mono">${escapeHtml(entry.tool_use_id)}</span>`
    );
    metaSection.classList.add("zoom-tool-meta");
    els.details.appendChild(metaSection);
  }
}

function renderRegularEntryZoom(els, entry) {
  const entryType = entry.entry_type || "unknown";
  setZoomHeader(els, entryType.toUpperCase(), `zoom-type-${entryType}`, entry);

  els.content.innerHTML = "";
  const content = formatTranscriptContent(entry);
  if (content.value) {
    renderContentSection(content, els.content);
  }

  els.details.innerHTML = "";
  if (entry.tool_name) {
    els.details.appendChild(createZoomSection("Tool", escapeHtml(entry.tool_name)));
  }
  if (entry.tool_use_id) {
    els.details.appendChild(createZoomSection("Tool Use ID", escapeHtml(entry.tool_use_id)));
  }
  if (entry.prompt_name) {
    els.details.appendChild(createZoomSection("Prompt", escapeHtml(entry.prompt_name)));
  }
  renderRawJsonSection(entry.parsed || entry.raw_json, "Raw JSON", els.details);
  if (els.details.children.length === 0) {
    els.details.innerHTML = `<div class="zoom-empty">(no additional details)</div>`;
  }
}

// ============================================================================
// Zoom Modal initialization
// ============================================================================

/**
 * Initializes the zoom modal for transcript entry detail view.
 *
 * @param {object} deps - Dependencies
 * @param {object} deps.state - The shared application state
 * @param {object} deps.transcriptView - The transcript view module
 * @param {function} deps.showToast - Toast notification helper
 * @returns {{ openTranscriptZoom, closeZoomModal, zoomPrev, zoomNext, isOpen, isNextPending }}
 */
export function initZoomModal({ state, transcriptView, showToast }) {
  const els = {
    modal: document.getElementById("zoom-modal"),
    modalType: document.getElementById("zoom-modal-type"),
    modalTimestamp: document.getElementById("zoom-modal-timestamp"),
    modalSource: document.getElementById("zoom-modal-source"),
    content: document.getElementById("zoom-content"),
    details: document.getElementById("zoom-details"),
    close: document.getElementById("zoom-close"),
    copy: document.getElementById("zoom-copy"),
    prev: document.getElementById("zoom-prev"),
    next: document.getElementById("zoom-next"),
  };

  let zoomNextPending = false;

  // -- Core operations --

  function openTranscriptZoom(index) {
    const entry = state.transcriptEntries[index];
    if (!entry) {
      return;
    }
    state.zoomOpen = true;
    state.zoomType = "transcript";
    state.zoomIndex = index;
    state.zoomEntry = entry;
    state.transcriptLoadRetries = 0;
    renderModal();
    els.modal.classList.remove("hidden");
  }

  function closeZoomModal() {
    state.zoomOpen = false;
    state.zoomType = null;
    state.zoomIndex = -1;
    state.zoomEntry = null;
    els.modal.classList.add("hidden");
  }

  function zoomPrev() {
    if (!state.zoomOpen || state.zoomIndex <= 0) {
      return;
    }
    openTranscriptZoom(state.zoomIndex - 1);
  }

  async function zoomNextWithLoad(startIndex, nextIndex) {
    zoomNextPending = true;
    try {
      await transcriptView.loadMoreTranscript();
      const indexUnchanged = state.zoomOpen && state.zoomIndex === startIndex;
      const hasNewEntry = indexUnchanged && nextIndex < state.transcriptEntries.length;
      if (hasNewEntry) {
        openTranscriptZoom(nextIndex);
      }
    } catch (error) {
      showToast(`Failed to load more entries: ${error.message}`, "error");
    } finally {
      zoomNextPending = false;
      updateNavigation();
    }
  }

  async function zoomNext() {
    if (!state.zoomOpen || zoomNextPending) {
      return;
    }
    const startIndex = state.zoomIndex;
    const nextIndex = startIndex + 1;
    const hasNextEntry = nextIndex < state.transcriptEntries.length;

    if (hasNextEntry) {
      openTranscriptZoom(nextIndex);
      return;
    }
    const canLoadMore =
      state.transcriptHasMore &&
      !state.transcriptLoading &&
      state.transcriptLoadRetries < MAX_TRANSCRIPT_LOAD_RETRIES;
    if (canLoadMore) {
      await zoomNextWithLoad(startIndex, nextIndex);
    }
  }

  // -- Rendering --

  function renderModal() {
    if (!state.zoomEntry) {
      return;
    }

    if (state.zoomEntry.isComposite) {
      renderCompositeEntryZoom(els, state.zoomEntry);
    } else {
      renderRegularEntryZoom(els, state.zoomEntry);
    }

    updateNavigation();
  }

  function updateNavigation() {
    if (!state.zoomOpen) {
      return;
    }

    const maxIndex = state.transcriptEntries.length - 1;
    const canLoadMore =
      state.transcriptHasMore && state.transcriptLoadRetries < MAX_TRANSCRIPT_LOAD_RETRIES;

    els.prev.disabled = state.zoomIndex <= 0;
    els.next.disabled = state.zoomIndex >= maxIndex && !canLoadMore;
  }

  async function copyEntry() {
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

  // -- Wire up DOM events --

  els.close.addEventListener("click", closeZoomModal);
  els.modal.querySelector(".zoom-modal-backdrop").addEventListener("click", closeZoomModal);
  els.copy.addEventListener("click", copyEntry);
  els.prev.addEventListener("click", zoomPrev);
  els.next.addEventListener("click", () => {
    zoomNext().catch((error) => showToast(`Navigation failed: ${error.message}`, "error"));
  });

  // Event delegation for zoom buttons in transcript list
  transcriptView.list.addEventListener("click", (e) => {
    const zoomBtn = e.target.closest(".zoom-button[data-zoom-index]");
    if (zoomBtn) {
      e.preventDefault();
      e.stopPropagation();
      const index = Number.parseInt(zoomBtn.dataset.zoomIndex, 10);
      openTranscriptZoom(index);
    }
  });

  return {
    openTranscriptZoom,
    closeZoomModal,
    zoomPrev,
    zoomNext,
    isOpen() {
      return state.zoomOpen;
    },
    isNextPending() {
      return zoomNextPending;
    },
  };
}
