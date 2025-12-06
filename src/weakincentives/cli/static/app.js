const state = {
  meta: null,
  selectedSlice: null,
  snapshots: [],
  entries: [],
  currentItems: [],
  openPaths: new Set(),
  closedPaths: new Set(),
  markdownViews: new Map(),
  expandDepth: 2,
  searchQuery: "",
  sliceBuckets: { state: [], event: [] },
  loading: false,
  theme: localStorage.getItem("wink-theme") || "light",
};

const MARKDOWN_KEY = "__markdown__";

const elements = {
  path: document.getElementById("snapshot-path"),
  snapshotSelect: document.getElementById("snapshot-select"),
  sessionTree: document.getElementById("session-tree"),
  created: document.getElementById("meta-created"),
  version: document.getElementById("meta-version"),
  count: document.getElementById("meta-count"),
  stateSliceList: document.getElementById("state-slice-list"),
  eventSliceList: document.getElementById("event-slice-list"),
  stateSliceCount: document.getElementById("state-slice-count"),
  eventSliceCount: document.getElementById("event-slice-count"),
  sliceFilter: document.getElementById("slice-filter"),
  sliceTitle: document.getElementById("slice-title"),
  itemCount: document.getElementById("item-count"),
  typeRow: document.getElementById("type-row"),
  tagRow: document.getElementById("tag-row"),
  jsonViewer: document.getElementById("json-viewer"),
  reload: document.getElementById("reload-button"),
  copy: document.getElementById("copy-button"),
  depthInput: document.getElementById("depth-input"),
  expandAll: document.getElementById("expand-all"),
  collapseAll: document.getElementById("collapse-all"),
  itemSearch: document.getElementById("item-search"),
  loadingOverlay: document.getElementById("loading-overlay"),
  toastContainer: document.getElementById("toast-container"),
  themeToggle: document.getElementById("theme-toggle"),
  sliceEmptyState: document.getElementById("slice-empty-state"),
  sliceContent: document.getElementById("slice-content"),
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

// Initialize theme on load
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

// --- API Fetching ---

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

function renderMeta(meta) {
  elements.path.textContent = meta.path;
  elements.created.textContent = meta.created_at;
  elements.version.textContent = meta.version;
  elements.count.textContent = `${meta.slices.length} slices`;
  elements.tagRow.innerHTML = "";

  const tags = meta.tags || {};
  const entries = Object.entries(tags);

  if (!entries.length) {
    const pill = document.createElement("span");
    pill.className = "pill pill-quiet";
    pill.textContent = "No tags set";
    elements.tagRow.appendChild(pill);
    return;
  }

  entries
    .sort(([a], [b]) => a.localeCompare(b))
    .forEach(([key, value]) => {
      const pill = document.createElement("span");
      pill.className = "pill";
      pill.textContent = `${key}: ${value}`;
      elements.tagRow.appendChild(pill);
    });
}

function renderSessionTree() {
  const container = elements.sessionTree;
  container.innerHTML = "";

  if (!state.entries.length) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "No sessions captured yet.";
    container.appendChild(empty);
    return;
  }

  const grouped = new Map();
  state.entries.forEach((entry) => {
    const bucket = grouped.get(entry.path) || [];
    bucket.push(entry);
    grouped.set(entry.path, bucket);
  });

  grouped.forEach((entries, path) => {
    entries.sort((a, b) => (a.line_number || 0) - (b.line_number || 0));
    const group = document.createElement("div");
    group.className =
      "session-group" + (state.meta && state.meta.path === path ? " active" : "");

    const header = document.createElement("div");
    header.className = "session-group-header";
    header.textContent = path.split("/").pop() || path;

    const badge = document.createElement("span");
    badge.className = "pill";
    badge.textContent = `${entries.length} session${entries.length === 1 ? "" : "s"}`;
    header.appendChild(badge);
    header.addEventListener("click", () => switchSnapshot(path));

    const list = document.createElement("div");
    list.className = "session-items";

    entries.forEach((entry) => {
      const item = document.createElement("div");
      item.className = "session-item" + (entry.selected ? " active" : "");

      const title = document.createElement("div");
      title.className = "session-item-title";
      title.textContent = entry.session_id;

      const meta = document.createElement("div");
      meta.className = "session-item-meta";
      const parts = [];
      if (entry.line_number !== undefined) parts.push(`line ${entry.line_number}`);
      if (entry.created_at) parts.push(entry.created_at);
      meta.textContent = parts.join(" · ");

      item.appendChild(title);
      item.appendChild(meta);
      item.addEventListener("click", () => {
        if (state.meta && state.meta.path === entry.path) {
          selectEntry(entry.session_id);
        } else {
          switchSnapshot(entry.path, entry.session_id);
        }
      });

      list.appendChild(item);
    });

    group.appendChild(header);
    group.appendChild(list);
    container.appendChild(group);
  });
}

async function isEventSlice(entry) {
  if (!entry.count) {
    return false;
  }
  try {
    const encoded = encodeURIComponent(entry.slice_type);
    const detail = await fetchJSON(`/api/slices/${encoded}?limit=1`);
    const sample = detail.items[0];
    if (sample && typeof sample === "object") {
      const hasEventId = Object.prototype.hasOwnProperty.call(sample, "event_id");
      const hasCreatedAt = Object.prototype.hasOwnProperty.call(sample, "created_at");
      return hasEventId && hasCreatedAt;
    }
  } catch (error) {
    console.warn("Failed to classify slice", entry.slice_type, error);
  }
  return false;
}

async function bucketSlices(entries) {
  const buckets = { state: [], event: [] };
  await Promise.all(
    entries.map(async (entry) => {
      const isEvent = await isEventSlice(entry);
      buckets[isEvent ? "event" : "state"].push(entry);
    })
  );
  return buckets;
}

function renderSliceList() {
  const filter = elements.sliceFilter.value.toLowerCase().trim();
  const renderBucket = (target, entries, emptyLabel) => {
    target.innerHTML = "";
    const filtered = entries.filter(
      (entry) =>
        entry.slice_type.toLowerCase().includes(filter) ||
        entry.item_type.toLowerCase().includes(filter)
    );

    if (!filtered.length) {
      const empty = document.createElement("li");
      empty.className = "slice-item muted";
      empty.textContent = emptyLabel;
      target.appendChild(empty);
      return;
    }

    filtered.forEach((entry) => {
      const item = document.createElement("li");
      item.className =
        "slice-item" + (entry.slice_type === state.selectedSlice ? " active" : "");

      const title = document.createElement("div");
      title.className = "slice-title";
      title.textContent = entry.slice_type;

      const subtitle = document.createElement("div");
      subtitle.className = "slice-subtitle";
      subtitle.textContent = `${entry.item_type} · ${entry.count} items`;

      item.appendChild(title);
      item.appendChild(subtitle);
      item.addEventListener("click", () => selectSlice(entry.slice_type));
      target.appendChild(item);
    });
  };

  renderBucket(
    elements.stateSliceList,
    state.sliceBuckets.state,
    "No state slices match"
  );
  renderBucket(
    elements.eventSliceList,
    state.sliceBuckets.event,
    "No event slices match"
  );
  elements.stateSliceCount.textContent = `${state.sliceBuckets.state.length} total`;
  elements.eventSliceCount.textContent = `${state.sliceBuckets.event.length} total`;
}

function showSliceContent(show) {
  elements.sliceEmptyState.classList.toggle("hidden", show);
  elements.sliceContent.classList.toggle("hidden", !show);
}

function renderSliceDetail(slice) {
  showSliceContent(true);
  elements.sliceTitle.textContent = slice.slice_type;
  elements.itemCount.textContent = `${slice.items.length} items`;
  elements.typeRow.innerHTML = "";
  state.currentItems = slice.items;
  state.markdownViews = new Map();
  state.searchQuery = "";
  elements.itemSearch.value = "";
  applyDepth(state.currentItems, state.expandDepth);

  const slicePill = document.createElement("span");
  slicePill.className = "pill";
  slicePill.textContent = `slice: ${slice.slice_type}`;

  const itemPill = document.createElement("span");
  itemPill.className = "pill";
  itemPill.textContent = `item: ${slice.item_type}`;

  elements.typeRow.appendChild(slicePill);
  elements.typeRow.appendChild(itemPill);

  renderItems(slice.items);
}

async function selectSlice(sliceType) {
  state.selectedSlice = sliceType;
  renderSliceList();
  const encoded = encodeURIComponent(sliceType);
  try {
    const detail = await fetchJSON(`/api/slices/${encoded}`);
    renderSliceDetail(detail);
  } catch (error) {
    elements.jsonViewer.textContent = error.message;
  }
}

async function refreshMeta() {
  const meta = await fetchJSON("/api/meta");
  state.meta = meta;
  state.sliceBuckets = await bucketSlices(meta.slices);
  renderMeta(meta);
  renderSliceList();
  if (!state.selectedSlice && meta.slices.length > 0) {
    await selectSlice(meta.slices[0].slice_type);
  } else if (state.selectedSlice) {
    const exists = meta.slices.some(
      (entry) => entry.slice_type === state.selectedSlice
    );
    if (exists) {
      await selectSlice(state.selectedSlice);
    } else if (meta.slices.length > 0) {
      await selectSlice(meta.slices[0].slice_type);
    }
  }
}

elements.sliceFilter.addEventListener("input", renderSliceList);
elements.copy.addEventListener("click", async () => {
  const text = JSON.stringify(
    getFilteredItems().map((entry) => entry.item),
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

// --- Reload Functionality ---

async function reloadSnapshot() {
  elements.reload.classList.add("spinning");
  try {
    await fetchJSON("/api/reload", { method: "POST" });
    await refreshMeta();
    await refreshEntries();
    showToast("Snapshot reloaded", "success");
  } catch (error) {
    showToast(`Reload failed: ${error.message}`, "error");
  } finally {
    elements.reload.classList.remove("spinning");
  }
}

elements.reload.addEventListener("click", reloadSnapshot);

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

elements.itemSearch.addEventListener("input", () => {
  state.searchQuery = elements.itemSearch.value || "";
  renderItems(state.currentItems);
});

document.addEventListener("DOMContentLoaded", async () => {
  setLoading(true);
  try {
    await refreshMeta();
    await refreshEntries();
    await refreshSnapshots();
  } catch (error) {
    showToast(`Load failed: ${error.message}`, "error");
    elements.jsonViewer.textContent = error.message;
  } finally {
    setLoading(false);
  }
});

const isObject = (value) => typeof value === "object" && value !== null;

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
  if (
    payload &&
    typeof payload === "object" &&
    typeof payload.text === "string" &&
    typeof payload.html === "string"
  ) {
    return payload;
  }

  return null;
}

function valueType(value) {
  if (getMarkdownPayload(value)) return "markdown";
  if (Array.isArray(value)) return "array";
  if (value === null) return "null";
  return typeof value;
}

// Type icons for tree visualization
const TYPE_ICONS = {
  string: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="4 7 4 4 20 4 20 7"></polyline><line x1="9" y1="20" x2="15" y2="20"></line><line x1="12" y1="4" x2="12" y2="20"></line></svg>`,
  number: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="9" x2="20" y2="9"></line><line x1="4" y1="15" x2="20" y2="15"></line><line x1="10" y1="3" x2="8" y2="21"></line><line x1="16" y1="3" x2="14" y2="21"></line></svg>`,
  boolean: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>`,
  null: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="4.93" y1="4.93" x2="19.07" y2="19.07"></line></svg>`,
  array: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="8" y1="6" x2="21" y2="6"></line><line x1="8" y1="12" x2="21" y2="12"></line><line x1="8" y1="18" x2="21" y2="18"></line><line x1="3" y1="6" x2="3.01" y2="6"></line><line x1="3" y1="12" x2="3.01" y2="12"></line><line x1="3" y1="18" x2="3.01" y2="18"></line></svg>`,
  object: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path></svg>`,
  markdown: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg>`,
  undefined: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>`,
};

function createTypeIcon(type) {
  const icon = document.createElement("span");
  icon.className = `type-icon type-${type}`;
  icon.innerHTML = TYPE_ICONS[type] || TYPE_ICONS.undefined;
  icon.title = type;
  return icon;
}

const isPrimitive = (value) =>
  value === null || (typeof value !== "object" && !Array.isArray(value));

const isSimpleArray = (value) =>
  Array.isArray(value) && value.every((item) => isPrimitive(item));

function previewValue(value, max = 24) {
  const raw =
    typeof value === "string" ? value : value === null ? "null" : String(value);
  return raw.length > max ? `${raw.slice(0, max)}…` : raw;
}

function arrayPreview(array, limit = 6) {
  const shown = array.slice(0, limit).map((entry) => previewValue(entry));
  const suffix =
    array.length > limit ? `, … +${array.length - limit} more` : "";
  return `[${shown.join(", ")}${suffix}]`;
}

function countLines(value) {
  return String(value).split("\n").length;
}

function applyTruncation(node, lineCount, limit = 25) {
  if (lineCount <= limit) {
    return null;
  }
  node.classList.add("truncated");
  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "leaf-toggle";
  toggle.textContent = "Expand";
  toggle.addEventListener("click", () => {
    const isTruncated = node.classList.toggle("truncated");
    toggle.textContent = isTruncated ? "Expand" : "Collapse";
  });
  return toggle;
}

function getFilteredItems() {
  const query = state.searchQuery.toLowerCase().trim();
  if (!query) {
    return state.currentItems.map((item, index) => ({ item, index }));
  }
  return state.currentItems
    .map((item, index) => ({
      item,
      index,
      text: JSON.stringify(item).toLowerCase(),
    }))
    .filter((entry) => entry.text.includes(query))
    .map(({ item, index }) => ({ item, index }));
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
  if (state.closedPaths.has(key)) return false;
  if (state.openPaths.has(key)) return true;
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

function expandChildren(value, path, open) {
  if (Array.isArray(value)) {
    value.forEach((child, index) => {
      const childPath = path.concat(String(index));
      setOpen(childPath, open);
      if (isObject(child)) {
        expandChildren(child, childPath, open);
      }
    });
  } else if (isObject(value)) {
    Object.entries(value).forEach(([key, child]) => {
      const childPath = path.concat(key);
      setOpen(childPath, open);
      if (isObject(child)) {
        expandChildren(child, childPath, open);
      }
    });
  }
}

function applyDepth(items, depth) {
  state.openPaths = new Set();
  state.closedPaths = new Set();
  const walk = (value, path, currentDepth) => {
    if (!(Array.isArray(value) || isObject(value))) {
      return;
    }
    if (currentDepth < depth) {
      setOpen(path, true);
    }
    if (Array.isArray(value)) {
      value.forEach((child, index) =>
        walk(child, path.concat(String(index)), currentDepth + 1)
      );
    } else {
      Object.entries(value).forEach(([key, child]) =>
        walk(child, path.concat(key), currentDepth + 1)
      );
    }
  };
  items.forEach((item, index) => walk(item, [`item-${index}`], 0));
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
      value.forEach((child, index) =>
        update(child, path.concat(String(index)))
      );
    } else {
      Object.entries(value).forEach(([key, child]) =>
        update(child, path.concat(key))
      );
    }
  };
  items.forEach((item, index) => update(item, [`item-${index}`]));
}

function renderTree(value, path, depth, label) {
  const node = document.createElement("div");
  node.className = "tree-node";

  const header = document.createElement("div");
  header.className = "tree-header";

  const markdown = getMarkdownPayload(value);
  const type = valueType(value);

  // Add type icon
  header.appendChild(createTypeIcon(type));

  const name = document.createElement("span");
  name.className = "tree-label";
  name.textContent = label;
  header.appendChild(name);

  const badge = document.createElement("span");
  badge.className = "pill pill-quiet";
  const childCount =
    type === "array"
      ? value.length
      : type === "object" && value !== null
        ? Object.keys(value).length
        : 0;
  if (type === "array") {
    badge.textContent = `array (${value.length})`;
  } else if (type === "markdown") {
    badge.textContent = "markdown";
  } else if (type === "object" && value !== null) {
    badge.textContent = `object (${Object.keys(value).length})`;
  } else {
    badge.textContent = type;
  }
  header.appendChild(badge);

  if (Array.isArray(value)) {
    const preview = document.createElement("span");
    preview.className = "tree-preview";
    preview.textContent = arrayPreview(value);
    header.appendChild(preview);
  } else if (markdown) {
    const preview = document.createElement("span");
    preview.className = "tree-preview";
    preview.textContent = previewValue(markdown.text, 64);
    header.appendChild(preview);
  }

  const body = document.createElement("div");
  body.className = "tree-body";

  const expandable = !markdown && (Array.isArray(value) || isObject(value));

  if (!expandable) {
    const wrapper = document.createElement("div");
    wrapper.className = "leaf-wrapper";

    if (markdown) {
      const view = getMarkdownView(path);
      const toggle = document.createElement("div");
      toggle.className = "markdown-toggle";

      const htmlButton = document.createElement("button");
      htmlButton.type = "button";
      htmlButton.textContent = "Rendered HTML";
      toggle.appendChild(htmlButton);

      const rawButton = document.createElement("button");
      rawButton.type = "button";
      rawButton.textContent = "Raw Markdown";
      toggle.appendChild(rawButton);

      const renderedLabel = document.createElement("p");
      renderedLabel.className = "markdown-label";
      renderedLabel.textContent = "Rendered markdown";

      const rendered = document.createElement("div");
      rendered.className = "markdown-rendered";
      rendered.innerHTML = markdown.html;

      const rawLabel = document.createElement("p");
      rawLabel.className = "markdown-label";
      rawLabel.textContent = "Raw markdown";

      const raw = document.createElement("pre");
      raw.className = "markdown-raw";
      raw.textContent = markdown.text;
      const truncationToggle = applyTruncation(raw, countLines(markdown.text));

      const renderedSection = document.createElement("div");
      renderedSection.className = "markdown-section";
      renderedSection.appendChild(renderedLabel);
      renderedSection.appendChild(rendered);

      const rawSection = document.createElement("div");
      rawSection.className = "markdown-section";
      rawSection.appendChild(rawLabel);
      rawSection.appendChild(raw);
      if (truncationToggle) {
        rawSection.appendChild(truncationToggle);
      }

      const applyView = (nextView) => {
        setMarkdownView(path, nextView);
        htmlButton.classList.toggle("active", nextView === "html");
        rawButton.classList.toggle("active", nextView === "raw");
        renderedSection.style.display = nextView === "html" ? "flex" : "none";
        rawSection.style.display = nextView === "raw" ? "flex" : "none";
      };

      htmlButton.addEventListener("click", () => applyView("html"));
      rawButton.addEventListener("click", () => applyView("raw"));
      applyView(view);

      wrapper.classList.add("markdown-leaf");
      wrapper.appendChild(toggle);
      wrapper.appendChild(renderedSection);
      wrapper.appendChild(rawSection);
    } else {
      const leaf = document.createElement("div");
      leaf.className = "tree-leaf";
      leaf.textContent = String(value);
      const toggle = applyTruncation(leaf, countLines(value));
      wrapper.appendChild(leaf);
      if (toggle) {
        wrapper.appendChild(toggle);
      }
    }
    body.appendChild(wrapper);
    node.appendChild(header);
    node.appendChild(body);
    return node;
  }

  const hasChildren = childCount > 0;
  let toggle;
  if (hasChildren) {
    const childControls = document.createElement("div");
    childControls.className = "tree-controls";
    header.appendChild(childControls);

    toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "tree-toggle";
    childControls.appendChild(toggle);

    const expandLevel = document.createElement("button");
    expandLevel.type = "button";
    expandLevel.className = "tree-mini";
    expandLevel.textContent = "Expand children";
    childControls.appendChild(expandLevel);

    const collapseLevel = document.createElement("button");
    collapseLevel.type = "button";
    collapseLevel.className = "tree-mini";
    collapseLevel.textContent = "Collapse children";
    childControls.appendChild(collapseLevel);

    expandLevel.addEventListener("click", () => {
      expandChildren(value, path, true);
      renderItems(state.currentItems);
    });

    collapseLevel.addEventListener("click", () => {
      expandChildren(value, path, false);
      renderItems(state.currentItems);
    });
  }

  const childrenContainer = document.createElement("div");
  childrenContainer.className = "tree-children";

  const renderChildren = () => {
    childrenContainer.innerHTML = "";
    if (!hasChildren) {
      const empty = document.createElement("span");
      empty.className = "muted";
      empty.textContent = "(empty)";
      childrenContainer.style.display = "block";
      childrenContainer.appendChild(empty);
      return;
    }
    if (!shouldOpen(path, depth)) {
      childrenContainer.style.display = "none";
      if (toggle) toggle.textContent = "Expand";
      return;
    }
    childrenContainer.style.display = "block";
    if (toggle) toggle.textContent = "Collapse";
    if (Array.isArray(value)) {
      if (isSimpleArray(value)) {
        childrenContainer.classList.add("compact-array");
        value.forEach((child, index) => {
          const chip = document.createElement("span");
          chip.className = "array-chip";
          chip.title = `[${index}]`;
          chip.textContent = String(child);
          const toggle = applyTruncation(chip, countLines(child));
          if (toggle) {
            const chipWrap = document.createElement("div");
            chipWrap.className = "chip-wrapper";
            chipWrap.appendChild(chip);
            chipWrap.appendChild(toggle);
            childrenContainer.appendChild(chipWrap);
          } else {
            childrenContainer.appendChild(chip);
          }
        });
      } else {
        childrenContainer.classList.remove("compact-array");
        value.forEach((child, index) => {
          childrenContainer.appendChild(
            renderTree(
              child,
              path.concat(String(index)),
              depth + 1,
              `[${index}]`
            )
          );
        });
      }
    } else {
      Object.entries(value).forEach(([key, child]) => {
        childrenContainer.appendChild(
          renderTree(child, path.concat(key), depth + 1, key)
        );
      });
    }
  };

  if (toggle) {
    toggle.addEventListener("click", () => {
      const next = !shouldOpen(path, depth);
      setOpen(path, next);
      renderItems(state.currentItems);
    });
  }

  renderChildren();

  body.appendChild(childrenContainer);
  node.appendChild(header);
  node.appendChild(body);
  return node;
}

function renderItems(items) {
  elements.jsonViewer.innerHTML = "";
  const filtered = getFilteredItems();
  const countLabel =
    state.searchQuery.trim().length > 0
      ? `${filtered.length} of ${items.length} items`
      : `${items.length} items`;
  elements.itemCount.textContent = countLabel;

  filtered.forEach(({ item, index }) => {
    const card = document.createElement("div");
    card.className = "item-card";

    const header = document.createElement("div");
    header.className = "item-header";

    const title = document.createElement("h3");
    title.textContent = `Item ${index + 1}`;
    header.appendChild(title);

    card.appendChild(header);

    const body = document.createElement("div");
    body.className = "item-body tree-root";
    body.appendChild(renderTree(item, [`item-${index}`], 0, "root"));
    card.appendChild(body);
    elements.jsonViewer.appendChild(card);
  });
}

async function refreshEntries() {
  const entries = await fetchJSON("/api/entries");
  state.entries = entries;

  renderSessionTree();
}

async function refreshSnapshots() {
  const snapshots = await fetchJSON("/api/snapshots");
  state.snapshots = snapshots;
  elements.snapshotSelect.innerHTML = "";

  snapshots.forEach((entry) => {
    const option = document.createElement("option");
    option.value = entry.path;
    option.textContent = `${entry.name} (${entry.created_at})`;
    elements.snapshotSelect.appendChild(option);
  });

  if (state.meta) {
    elements.snapshotSelect.value = state.meta.path;
  }
}

async function selectEntry(sessionId) {
  try {
    await fetchJSON("/api/select", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ session_id: sessionId }),
    });
    await refreshMeta();
    await refreshEntries();
  } catch (error) {
    alert(`Select failed: ${error.message}`);
  }
}

async function switchSnapshot(path, sessionId) {
  try {
    await fetchJSON("/api/switch", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(
        sessionId ? { path, session_id: sessionId } : { path }
      ),
    });
    await refreshMeta();
    await refreshEntries();
    await refreshSnapshots();
  } catch (error) {
    alert(`Switch failed: ${error.message}`);
  }
}

elements.snapshotSelect.addEventListener("change", (event) => {
  const target = event.target;
  if (target && target.value) {
    switchSnapshot(target.value);
  }
});

document.addEventListener("keydown", (event) => {
  // Ignore if typing in an input
  const tag = event.target.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") {
    return;
  }

  // Reload shortcut
  if (event.key === "r" || event.key === "R") {
    event.preventDefault();
    reloadSnapshot();
    return;
  }

  // Theme toggle shortcut
  if (event.key === "d" || event.key === "D") {
    event.preventDefault();
    toggleTheme();
    return;
  }

  // Snapshot navigation
  if (!state.snapshots.length) {
    return;
  }
  if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") {
    return;
  }
  event.preventDefault();

  const current = elements.snapshotSelect.value;
  const index = state.snapshots.findIndex((entry) => entry.path === current);
  if (index === -1) {
    return;
  }

  const nextIndex =
    event.key === "ArrowRight"
      ? Math.min(state.snapshots.length - 1, index + 1)
      : Math.max(0, index - 1);

  if (nextIndex !== index) {
    const next = state.snapshots[nextIndex];
    elements.snapshotSelect.value = next.path;
    switchSnapshot(next.path);
    elements.snapshotSelect.classList.add("active-switch");
    setTimeout(() => elements.snapshotSelect.classList.remove("active-switch"), 200);
  }
});
