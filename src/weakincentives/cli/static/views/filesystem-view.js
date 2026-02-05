// ============================================================================
// Filesystem View - Workspace snapshot file browser
// ============================================================================

import { escapeHtml } from "../lib.js";

const FILESYSTEM_PREFIX = "filesystem/";

/**
 * Initializes the filesystem view. Wires up DOM events and provides data loading.
 *
 * @param {object} deps - Dependencies
 * @param {object} deps.state - The shared application state
 * @param {function} deps.fetchJSON - API fetch helper
 * @param {function} deps.showToast - Toast notification helper
 * @returns {{ loadFilesystem }}
 */
export function initFilesystemView({ state, fetchJSON, showToast }) {
  const els = {
    filter: document.getElementById("filesystem-filter"),
    list: document.getElementById("filesystem-list"),
    emptyState: document.getElementById("filesystem-empty-state"),
    noSnapshot: document.getElementById("filesystem-no-snapshot"),
    content: document.getElementById("filesystem-content"),
    currentPath: document.getElementById("filesystem-current-path"),
    viewer: document.getElementById("filesystem-viewer"),
    copy: document.getElementById("filesystem-copy"),
  };

  // -- Data loading --

  async function loadFilesystem() {
    try {
      const files = await fetchJSON("/api/files");
      state.allFiles = files;

      state.filesystemFiles = files
        .filter((f) => f.startsWith(FILESYSTEM_PREFIX))
        .map((f) => f.slice(FILESYSTEM_PREFIX.length));

      state.hasFilesystemSnapshot = state.filesystemFiles.length > 0;
      renderList();
    } catch (error) {
      els.list.innerHTML = `<p class="muted">Failed to load files: ${escapeHtml(error.message)}</p>`;
    }
  }

  // -- Rendering --

  function renderList() {
    if (!state.hasFilesystemSnapshot) {
      els.list.innerHTML = "";
      els.emptyState.classList.add("hidden");
      els.noSnapshot.classList.remove("hidden");
      els.content.classList.add("hidden");
      return;
    }

    els.noSnapshot.classList.add("hidden");

    const filter = state.filesystemFilter.toLowerCase();
    const filtered = state.filesystemFiles.filter((f) => f.toLowerCase().includes(filter));

    els.list.innerHTML = "";

    if (filtered.length === 0) {
      els.list.innerHTML = '<p class="muted">No files match filter</p>';
      return;
    }

    filtered.forEach((displayPath) => {
      const item = document.createElement("div");
      const fullPath = FILESYSTEM_PREFIX + displayPath;
      item.className = `file-item${fullPath === state.selectedFile ? " active" : ""}`;
      item.textContent = displayPath;
      item.addEventListener("click", () => selectFile(fullPath, displayPath));
      els.list.appendChild(item);
    });
  }

  const ALLOWED_MIME_TYPES = new Set([
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    "image/svg+xml",
    "image/x-icon",
    "image/bmp",
  ]);

  function renderImageFile(result, displayPath) {
    const mimeType = ALLOWED_MIME_TYPES.has(result.mime_type) ? result.mime_type : "image/png";
    const container = document.createElement("div");
    container.className = "image-container";
    const img = document.createElement("img");
    img.src = `data:${mimeType};base64,${result.content}`;
    img.alt = displayPath;
    img.className = "filesystem-image";
    container.appendChild(img);
    els.viewer.innerHTML = "";
    els.viewer.appendChild(container);
    state.fileContent = null;
  }

  function renderTextFile(result) {
    const content =
      result.type === "json" ? JSON.stringify(result.content, null, 2) : result.content;
    state.fileContent = content;
    els.viewer.innerHTML = `<pre class="file-content">${escapeHtml(content)}</pre>`;
  }

  function renderMarkdownFile(result) {
    state.fileContent = result.content;
    const container = document.createElement("div");
    container.className = "filesystem-markdown";

    const toggle = document.createElement("div");
    toggle.className = "markdown-toggle";
    const renderedBtn = document.createElement("button");
    renderedBtn.type = "button";
    renderedBtn.textContent = "Rendered";
    renderedBtn.className = "active";
    const rawBtn = document.createElement("button");
    rawBtn.type = "button";
    rawBtn.textContent = "Raw";
    toggle.appendChild(renderedBtn);
    toggle.appendChild(rawBtn);

    const renderedSection = document.createElement("div");
    renderedSection.className = "markdown-section";
    renderedSection.innerHTML = `<div class="markdown-rendered">${result.html}</div>`;

    const rawSection = document.createElement("div");
    rawSection.className = "markdown-section";
    rawSection.style.display = "none";
    rawSection.innerHTML = `<pre class="markdown-raw">${escapeHtml(result.content)}</pre>`;

    renderedBtn.addEventListener("click", () => {
      renderedBtn.classList.add("active");
      rawBtn.classList.remove("active");
      renderedSection.style.display = "flex";
      rawSection.style.display = "none";
    });
    rawBtn.addEventListener("click", () => {
      rawBtn.classList.add("active");
      renderedBtn.classList.remove("active");
      rawSection.style.display = "flex";
      renderedSection.style.display = "none";
    });

    container.appendChild(toggle);
    container.appendChild(renderedSection);
    container.appendChild(rawSection);
    els.viewer.innerHTML = "";
    els.viewer.appendChild(container);
  }

  function renderFileResult(result, displayPath) {
    if (result.type === "image") {
      renderImageFile(result, displayPath);
    } else if (result.type === "markdown") {
      renderMarkdownFile(result);
    } else if (result.type === "binary") {
      // biome-ignore lint/nursery/noSecrets: HTML string, not a secret
      els.viewer.innerHTML = '<p class="muted">Binary file cannot be displayed</p>';
      state.fileContent = null;
    } else {
      renderTextFile(result);
    }
  }

  async function selectFile(fullPath, displayPath) {
    state.selectedFile = fullPath;
    renderList();

    try {
      const result = await fetchJSON(`/api/files/${encodeURIComponent(fullPath)}`);

      els.emptyState.classList.add("hidden");
      els.noSnapshot.classList.add("hidden");
      els.content.classList.remove("hidden");
      els.currentPath.textContent = displayPath;

      renderFileResult(result, displayPath);
    } catch (error) {
      els.viewer.innerHTML = `<p class="muted">Failed to load file: ${escapeHtml(error.message)}</p>`;
    }
  }

  // -- Wire up DOM events --

  els.filter.addEventListener("input", () => {
    state.filesystemFilter = els.filter.value;
    renderList();
  });

  els.copy.addEventListener("click", async () => {
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

  return {
    loadFilesystem,
  };
}
