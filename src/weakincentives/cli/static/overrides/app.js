const state = {
  config: null,
  prompts: [],
  selectedPrompt: null,
  promptDetail: null,
  loading: false,
  theme: localStorage.getItem("wink-theme") || "light",
  shortcutsOpen: false,
  renderedPanelExpanded: true,
  rawView: false,
  editingSection: null,
  editingTool: null,
  confirmCallback: null,
  filterQuery: "",
};

const elements = {
  snapshotPath: document.getElementById("snapshot-path"),
  tagBadge: document.getElementById("tag-badge"),
  promptFilter: document.getElementById("prompt-filter"),
  promptList: document.getElementById("prompt-list"),
  emptyState: document.getElementById("empty-state"),
  promptContent: document.getElementById("prompt-content"),
  promptTitle: document.getElementById("prompt-title"),
  promptMeta: document.getElementById("prompt-meta"),
  seedBadge: document.getElementById("seed-badge"),
  renderedPanel: document.getElementById("rendered-panel"),
  renderedToggle: document.getElementById("rendered-toggle"),
  renderedContent: document.getElementById("rendered-content"),
  renderedHtml: document.getElementById("rendered-html"),
  renderedRaw: document.getElementById("rendered-raw"),
  rawToggle: document.getElementById("raw-toggle"),
  sectionCount: document.getElementById("section-count"),
  sectionsContainer: document.getElementById("sections-container"),
  toolCount: document.getElementById("tool-count"),
  toolsContainer: document.getElementById("tools-container"),
  loadingOverlay: document.getElementById("loading-overlay"),
  toastContainer: document.getElementById("toast-container"),
  themeToggle: document.getElementById("theme-toggle"),
  reloadButton: document.getElementById("reload-button"),
  helpButton: document.getElementById("help-button"),
  shortcutsOverlay: document.getElementById("shortcuts-overlay"),
  shortcutsClose: document.getElementById("shortcuts-close"),
  confirmModal: document.getElementById("confirm-modal"),
  confirmModalTitle: document.getElementById("confirm-modal-title"),
  confirmModalMessage: document.getElementById("confirm-modal-message"),
  confirmModalCancel: document.getElementById("confirm-modal-cancel"),
  confirmModalConfirm: document.getElementById("confirm-modal-confirm"),
};

// --- Theme Management ---

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  state.theme = theme;
  localStorage.setItem("wink-theme", theme);
}

function toggleTheme() {
  const next = state.theme === "dark" ? "light" : "dark";
  applyTheme(next);
}

applyTheme(state.theme);
elements.themeToggle.addEventListener("click", toggleTheme);

// --- Loading State ---

function setLoading(loading) {
  state.loading = loading;
  elements.loadingOverlay.classList.toggle("hidden", !loading);
}

// --- Toast Notifications ---

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

// --- API Helpers ---

async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return response.json();
}

async function fetchWithLoading(url, options) {
  setLoading(true);
  try {
    return await fetchJSON(url, options);
  } finally {
    setLoading(false);
  }
}

// --- Confirm Modal ---

function showConfirm(title, message, callback) {
  elements.confirmModalTitle.textContent = title;
  elements.confirmModalMessage.textContent = message;
  state.confirmCallback = callback;
  elements.confirmModal.classList.remove("hidden");
}

function hideConfirm() {
  elements.confirmModal.classList.add("hidden");
  state.confirmCallback = null;
}

elements.confirmModalCancel.addEventListener("click", hideConfirm);
elements.confirmModal.querySelector(".confirm-modal-backdrop").addEventListener("click", hideConfirm);
elements.confirmModalConfirm.addEventListener("click", () => {
  if (state.confirmCallback) {
    state.confirmCallback();
  }
  hideConfirm();
});

// --- Shortcuts Overlay ---

function openShortcuts() {
  state.shortcutsOpen = true;
  elements.shortcutsOverlay.classList.remove("hidden");
}

function closeShortcuts() {
  state.shortcutsOpen = false;
  elements.shortcutsOverlay.classList.add("hidden");
}

elements.shortcutsClose.addEventListener("click", closeShortcuts);
elements.shortcutsOverlay.querySelector(".shortcuts-backdrop").addEventListener("click", closeShortcuts);
elements.helpButton.addEventListener("click", openShortcuts);

// --- Config Loading ---

async function loadConfig() {
  const config = await fetchJSON("/api/config");
  state.config = config;
  elements.snapshotPath.textContent = config.snapshot_path;
  elements.tagBadge.textContent = `tag: ${config.tag}`;
}

// --- Prompt List ---

async function loadPrompts() {
  const prompts = await fetchJSON("/api/prompts");
  state.prompts = prompts;
  renderPromptList();
}

function renderPromptList() {
  const container = elements.promptList;
  container.innerHTML = "";

  const query = state.filterQuery.toLowerCase();
  const filtered = state.prompts.filter((prompt) => {
    if (!query) return true;
    const searchText = `${prompt.ns}:${prompt.key} ${prompt.name || ""}`.toLowerCase();
    return searchText.includes(query);
  });

  if (filtered.length === 0) {
    const empty = document.createElement("li");
    empty.className = "prompt-item muted";
    empty.textContent = query ? "No prompts match filter" : "No prompts found";
    container.appendChild(empty);
    return;
  }

  filtered.forEach((prompt) => {
    const item = document.createElement("li");
    const isSelected =
      state.selectedPrompt &&
      state.selectedPrompt.ns === prompt.ns &&
      state.selectedPrompt.key === prompt.key;

    item.className = "prompt-item" + (isSelected ? " active" : "");
    if (!prompt.is_seeded) {
      item.classList.add("not-seeded");
    }

    const title = document.createElement("div");
    title.className = "prompt-item-title";
    title.textContent = prompt.name || `${prompt.ns}:${prompt.key}`;

    if (prompt.stale_count > 0) {
      const icon = document.createElement("span");
      icon.className = "warning-icon";
      icon.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>`;
      icon.title = `${prompt.stale_count} stale override(s)`;
      title.appendChild(icon);
    }

    const subtitle = document.createElement("div");
    subtitle.className = "prompt-item-subtitle";
    subtitle.textContent = `${prompt.ns}:${prompt.key}`;

    const meta = document.createElement("div");
    meta.className = "prompt-item-meta";

    const sectionPill = document.createElement("span");
    sectionPill.className = "pill pill-quiet";
    sectionPill.textContent = `${prompt.section_count} sections`;
    meta.appendChild(sectionPill);

    const toolPill = document.createElement("span");
    toolPill.className = "pill pill-quiet";
    toolPill.textContent = `${prompt.tool_count} tools`;
    meta.appendChild(toolPill);

    if (!prompt.is_seeded) {
      const seedPill = document.createElement("span");
      seedPill.className = "pill badge-not-seeded";
      seedPill.textContent = "Not Seeded";
      meta.appendChild(seedPill);
    } else if (prompt.has_overrides) {
      const overridePill = document.createElement("span");
      overridePill.className = "pill badge-overridden";
      overridePill.textContent = "Overridden";
      meta.appendChild(overridePill);
    }

    item.appendChild(title);
    item.appendChild(subtitle);
    item.appendChild(meta);

    item.addEventListener("click", () => selectPrompt(prompt.ns, prompt.key));
    container.appendChild(item);
  });
}

elements.promptFilter.addEventListener("input", () => {
  state.filterQuery = elements.promptFilter.value;
  renderPromptList();
});

// --- Prompt Detail ---

async function selectPrompt(ns, key) {
  state.selectedPrompt = { ns, key };
  renderPromptList();

  try {
    const detail = await fetchWithLoading(
      `/api/prompts/${encodeURIComponent(ns)}/${encodeURIComponent(key)}`
    );
    state.promptDetail = detail;
    renderPromptDetail();
  } catch (error) {
    showToast(`Failed to load prompt: ${error.message}`, "error");
  }
}

function renderPromptDetail() {
  const detail = state.promptDetail;
  if (!detail) {
    elements.emptyState.classList.remove("hidden");
    elements.promptContent.classList.add("hidden");
    return;
  }

  elements.emptyState.classList.add("hidden");
  elements.promptContent.classList.remove("hidden");

  // Header
  elements.promptTitle.textContent = detail.name || `${detail.ns}:${detail.key}`;
  elements.promptMeta.textContent = `${detail.ns}:${detail.key} | ${detail.created_at}`;

  // Seed badge
  if (!detail.is_seeded) {
    elements.seedBadge.textContent = "Seed Required";
    elements.seedBadge.className = "pill badge-not-seeded";
    elements.seedBadge.classList.remove("hidden");
  } else {
    elements.seedBadge.classList.add("hidden");
  }

  // Rendered prompt
  elements.renderedHtml.innerHTML = detail.rendered_prompt.html;
  elements.renderedRaw.textContent = detail.rendered_prompt.text;
  updateRawToggle();

  // Sections
  elements.sectionCount.textContent = `${detail.sections.length}`;
  renderSections(detail.sections, detail.is_seeded);

  // Tools
  elements.toolCount.textContent = `${detail.tools.length}`;
  renderTools(detail.tools, detail.is_seeded);
}

// --- Rendered Panel Toggle ---

function toggleRenderedPanel() {
  state.renderedPanelExpanded = !state.renderedPanelExpanded;
  elements.renderedPanel.classList.toggle("expanded", state.renderedPanelExpanded);
}

function toggleRawView() {
  state.rawView = !state.rawView;
  updateRawToggle();
}

function updateRawToggle() {
  elements.rawToggle.textContent = state.rawView ? "HTML" : "Raw";
  elements.renderedHtml.classList.toggle("hidden", state.rawView);
  elements.renderedRaw.classList.toggle("hidden", !state.rawView);
}

elements.renderedToggle.addEventListener("click", (e) => {
  if (e.target.closest(".panel-toggle-actions")) return;
  toggleRenderedPanel();
});
elements.rawToggle.addEventListener("click", (e) => {
  e.stopPropagation();
  toggleRawView();
});

// Initialize panel state
elements.renderedPanel.classList.add("expanded");

// --- Sections Rendering ---

function renderSections(sections, isSeeded) {
  const container = elements.sectionsContainer;
  container.innerHTML = "";

  if (sections.length === 0) {
    const empty = document.createElement("div");
    empty.className = "accordion-empty";
    empty.innerHTML = `
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
        <polyline points="14 2 14 8 20 8"></polyline>
      </svg>
      <p>No sections available</p>
    `;
    container.appendChild(empty);
    return;
  }

  sections.forEach((section) => {
    const item = createSectionAccordion(section, isSeeded);
    container.appendChild(item);
  });
}

function createSectionAccordion(section, isSeeded) {
  const item = document.createElement("div");
  item.className = "accordion-item";
  item.dataset.path = section.path.join("/");

  const header = document.createElement("div");
  header.className = "accordion-header";

  const left = document.createElement("div");
  left.className = "accordion-header-left";
  left.innerHTML = `
    <svg class="chevron" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <polyline points="9 18 15 12 9 6"></polyline>
    </svg>
    <span class="accordion-number">${section.number}</span>
    <span class="accordion-title">${section.path.join(" / ")}</span>
  `;

  const right = document.createElement("div");
  right.className = "accordion-header-right";

  if (section.is_stale) {
    const staleBadge = document.createElement("span");
    staleBadge.className = "pill badge-stale";
    staleBadge.textContent = "Stale";
    right.appendChild(staleBadge);
  }

  if (section.is_overridden) {
    const overrideBadge = document.createElement("span");
    overrideBadge.className = "pill badge-overridden";
    overrideBadge.textContent = "Overridden";
    right.appendChild(overrideBadge);
  } else {
    const originalBadge = document.createElement("span");
    originalBadge.className = "pill pill-quiet";
    originalBadge.textContent = "Original";
    right.appendChild(originalBadge);
  }

  header.appendChild(left);
  header.appendChild(right);

  const body = document.createElement("div");
  body.className = "accordion-body";

  if (!isSeeded) {
    body.innerHTML = `
      <div class="not-seeded-message">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="12" y1="8" x2="12" y2="12"></line>
          <line x1="12" y1="16" x2="12.01" y2="16"></line>
        </svg>
        <p>This prompt must be seeded before overrides can be edited.</p>
      </div>
    `;
  } else {
    body.appendChild(createSectionEditor(section));
  }

  header.addEventListener("click", () => {
    item.classList.toggle("expanded");
  });

  item.appendChild(header);
  item.appendChild(body);
  return item;
}

function createSectionEditor(section) {
  const container = document.createElement("div");
  container.className = "editor-container";

  const textarea = document.createElement("textarea");
  textarea.className = "editor-textarea";
  textarea.value = section.current_body || "";
  textarea.placeholder = "Enter section override content...";

  const actions = document.createElement("div");
  actions.className = "editor-actions";

  if (section.is_overridden) {
    const revertBtn = document.createElement("button");
    revertBtn.type = "button";
    revertBtn.className = "ghost danger small";
    revertBtn.textContent = "Revert";
    revertBtn.addEventListener("click", () => {
      showConfirm(
        "Revert Override",
        `Revert the "${section.path.join(" / ")}" section to original?`,
        () => deleteSection(section.path)
      );
    });
    actions.appendChild(revertBtn);
  }

  const saveBtn = document.createElement("button");
  saveBtn.type = "button";
  saveBtn.className = "primary small";
  saveBtn.textContent = "Save";
  saveBtn.addEventListener("click", () => {
    saveSection(section.path, textarea.value);
  });
  actions.appendChild(saveBtn);

  container.appendChild(textarea);
  container.appendChild(actions);
  return container;
}

async function saveSection(path, body) {
  if (!state.selectedPrompt) return;

  try {
    await fetchWithLoading(
      `/api/prompts/${encodeURIComponent(state.selectedPrompt.ns)}/${encodeURIComponent(state.selectedPrompt.key)}/sections/${encodeURIComponent(path.join("/"))}`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ body }),
      }
    );
    showToast("Section override saved", "success");
    await loadPrompts();
    await selectPrompt(state.selectedPrompt.ns, state.selectedPrompt.key);
  } catch (error) {
    showToast(`Failed to save: ${error.message}`, "error");
  }
}

async function deleteSection(path) {
  if (!state.selectedPrompt) return;

  try {
    await fetchWithLoading(
      `/api/prompts/${encodeURIComponent(state.selectedPrompt.ns)}/${encodeURIComponent(state.selectedPrompt.key)}/sections/${encodeURIComponent(path.join("/"))}`,
      { method: "DELETE" }
    );
    showToast("Section override reverted", "success");
    await loadPrompts();
    await selectPrompt(state.selectedPrompt.ns, state.selectedPrompt.key);
  } catch (error) {
    showToast(`Failed to revert: ${error.message}`, "error");
  }
}

// --- Tools Rendering ---

function renderTools(tools, isSeeded) {
  const container = elements.toolsContainer;
  container.innerHTML = "";

  if (tools.length === 0) {
    const empty = document.createElement("div");
    empty.className = "accordion-empty";
    empty.innerHTML = `
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"></path>
      </svg>
      <p>No tools available</p>
    `;
    container.appendChild(empty);
    return;
  }

  tools.forEach((tool) => {
    const item = createToolAccordion(tool, isSeeded);
    container.appendChild(item);
  });
}

function createToolAccordion(tool, isSeeded) {
  const item = document.createElement("div");
  item.className = "accordion-item";
  item.dataset.toolName = tool.name;

  const header = document.createElement("div");
  header.className = "accordion-header";

  const left = document.createElement("div");
  left.className = "accordion-header-left";
  left.innerHTML = `
    <svg class="chevron" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <polyline points="9 18 15 12 9 6"></polyline>
    </svg>
    <span class="accordion-title">${tool.name}</span>
  `;

  const right = document.createElement("div");
  right.className = "accordion-header-right";

  if (tool.is_stale) {
    const staleBadge = document.createElement("span");
    staleBadge.className = "pill badge-stale";
    staleBadge.textContent = "Stale";
    right.appendChild(staleBadge);
  }

  if (tool.is_overridden) {
    const overrideBadge = document.createElement("span");
    overrideBadge.className = "pill badge-overridden";
    overrideBadge.textContent = "Overridden";
    right.appendChild(overrideBadge);
  } else {
    const originalBadge = document.createElement("span");
    originalBadge.className = "pill pill-quiet";
    originalBadge.textContent = "Original";
    right.appendChild(originalBadge);
  }

  header.appendChild(left);
  header.appendChild(right);

  const body = document.createElement("div");
  body.className = "accordion-body";

  if (!isSeeded) {
    body.innerHTML = `
      <div class="not-seeded-message">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="12" y1="8" x2="12" y2="12"></line>
          <line x1="12" y1="16" x2="12.01" y2="16"></line>
        </svg>
        <p>This prompt must be seeded before overrides can be edited.</p>
      </div>
    `;
  } else {
    body.appendChild(createToolEditor(tool));
  }

  header.addEventListener("click", () => {
    item.classList.toggle("expanded");
  });

  item.appendChild(header);
  item.appendChild(body);
  return item;
}

function createToolEditor(tool) {
  const container = document.createElement("div");
  container.className = "tool-editor";

  // Description field
  const descField = document.createElement("div");
  descField.className = "tool-field";
  descField.innerHTML = `<label>Description</label>`;

  const descInput = document.createElement("textarea");
  descInput.rows = 3;
  descInput.value = tool.current_description || "";
  descInput.placeholder = "Override tool description...";
  descField.appendChild(descInput);

  container.appendChild(descField);

  // Param descriptions
  const paramNames = Object.keys(tool.current_param_descriptions || {});
  if (paramNames.length > 0) {
    const paramField = document.createElement("div");
    paramField.className = "tool-field";
    paramField.innerHTML = `<label>Parameter Descriptions</label>`;

    const paramFields = document.createElement("div");
    paramFields.className = "param-fields";

    const paramInputs = {};
    paramNames.forEach((name) => {
      const row = document.createElement("div");
      row.className = "param-row";

      const nameSpan = document.createElement("span");
      nameSpan.className = "param-name";
      nameSpan.textContent = name;

      const input = document.createElement("input");
      input.type = "text";
      input.value = tool.current_param_descriptions[name] || "";
      input.placeholder = `Description for ${name}`;
      paramInputs[name] = input;

      row.appendChild(nameSpan);
      row.appendChild(input);
      paramFields.appendChild(row);
    });

    paramField.appendChild(paramFields);
    container.appendChild(paramField);
  }

  // Actions
  const actions = document.createElement("div");
  actions.className = "editor-actions";

  if (tool.is_overridden) {
    const revertBtn = document.createElement("button");
    revertBtn.type = "button";
    revertBtn.className = "ghost danger small";
    revertBtn.textContent = "Revert";
    revertBtn.addEventListener("click", () => {
      showConfirm(
        "Revert Override",
        `Revert the "${tool.name}" tool to original?`,
        () => deleteTool(tool.name)
      );
    });
    actions.appendChild(revertBtn);
  }

  const saveBtn = document.createElement("button");
  saveBtn.type = "button";
  saveBtn.className = "primary small";
  saveBtn.textContent = "Save";
  saveBtn.addEventListener("click", () => {
    const paramDescriptions = {};
    if (paramNames.length > 0) {
      const paramInputs = container.querySelectorAll(".param-row input");
      paramInputs.forEach((input, index) => {
        paramDescriptions[paramNames[index]] = input.value;
      });
    }
    saveTool(tool.name, descInput.value || null, paramDescriptions);
  });
  actions.appendChild(saveBtn);

  container.appendChild(actions);
  return container;
}

async function saveTool(toolName, description, paramDescriptions) {
  if (!state.selectedPrompt) return;

  try {
    await fetchWithLoading(
      `/api/prompts/${encodeURIComponent(state.selectedPrompt.ns)}/${encodeURIComponent(state.selectedPrompt.key)}/tools/${encodeURIComponent(toolName)}`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ description, param_descriptions: paramDescriptions }),
      }
    );
    showToast("Tool override saved", "success");
    await loadPrompts();
    await selectPrompt(state.selectedPrompt.ns, state.selectedPrompt.key);
  } catch (error) {
    showToast(`Failed to save: ${error.message}`, "error");
  }
}

async function deleteTool(toolName) {
  if (!state.selectedPrompt) return;

  try {
    await fetchWithLoading(
      `/api/prompts/${encodeURIComponent(state.selectedPrompt.ns)}/${encodeURIComponent(state.selectedPrompt.key)}/tools/${encodeURIComponent(toolName)}`,
      { method: "DELETE" }
    );
    showToast("Tool override reverted", "success");
    await loadPrompts();
    await selectPrompt(state.selectedPrompt.ns, state.selectedPrompt.key);
  } catch (error) {
    showToast(`Failed to revert: ${error.message}`, "error");
  }
}

// --- Reload ---

async function reloadSnapshot() {
  elements.reloadButton.classList.add("spinning");
  try {
    await fetchJSON("/api/reload", { method: "POST" });
    await loadPrompts();
    if (state.selectedPrompt) {
      await selectPrompt(state.selectedPrompt.ns, state.selectedPrompt.key);
    }
    showToast("Snapshot reloaded", "success");
  } catch (error) {
    showToast(`Reload failed: ${error.message}`, "error");
  } finally {
    elements.reloadButton.classList.remove("spinning");
  }
}

elements.reloadButton.addEventListener("click", reloadSnapshot);

// --- Keyboard Shortcuts ---

document.addEventListener("keydown", (event) => {
  // Escape handling
  if (event.key === "Escape") {
    if (state.shortcutsOpen) {
      event.preventDefault();
      closeShortcuts();
      return;
    }
    if (!elements.confirmModal.classList.contains("hidden")) {
      event.preventDefault();
      hideConfirm();
      return;
    }
    return;
  }

  // Ignore if typing in an input
  const tag = event.target.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") {
    // Ctrl/Cmd+S to save
    if ((event.metaKey || event.ctrlKey) && event.key === "s") {
      event.preventDefault();
      // Find active save button and click it
      const activeBody = document.querySelector(".accordion-item.expanded .accordion-body");
      if (activeBody) {
        const saveBtn = activeBody.querySelector("button.primary");
        if (saveBtn) saveBtn.click();
      }
    }
    return;
  }

  // Don't process other shortcuts if a dialog is open
  if (state.shortcutsOpen || !elements.confirmModal.classList.contains("hidden")) {
    return;
  }

  // Reload (R)
  if (event.key === "r" || event.key === "R") {
    event.preventDefault();
    reloadSnapshot();
    return;
  }

  // Toggle dark mode (D)
  if (event.key === "d" || event.key === "D") {
    event.preventDefault();
    toggleTheme();
    return;
  }

  // Toggle rendered prompt panel (P)
  if (event.key === "p" || event.key === "P") {
    event.preventDefault();
    toggleRenderedPanel();
    return;
  }

  // Toggle markdown/raw (M)
  if (event.key === "m" || event.key === "M") {
    event.preventDefault();
    toggleRawView();
    return;
  }

  // Shortcuts help (?)
  if (event.key === "?" || (event.shiftKey && event.key === "/")) {
    event.preventDefault();
    openShortcuts();
    return;
  }
});

// --- Initialization ---

document.addEventListener("DOMContentLoaded", async () => {
  setLoading(true);
  try {
    await loadConfig();
    await loadPrompts();
  } catch (error) {
    showToast(`Load failed: ${error.message}`, "error");
  } finally {
    setLoading(false);
  }
});
