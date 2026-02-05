// ============================================================================
// Sessions View - Slice browser with JSON tree rendering
// ============================================================================

import {
  escapeHtml,
  getMarkdownPayload,
  isObject,
  isSimpleArray,
  pathKey,
  splitQualifiedName,
  valueType,
} from "../lib.js";

// ============================================================================
// Slice classification
// ============================================================================

async function isEventSlice(entry, fetchJSON) {
  if (!entry.count) {
    return false;
  }
  try {
    const encoded = encodeURIComponent(entry.slice_type);
    const detail = await fetchJSON(`/api/slices/${encoded}?limit=1`);
    const sample = detail.items[0];
    if (sample && typeof sample === "object") {
      return (
        Object.prototype.hasOwnProperty.call(sample, "event_id") &&
        Object.prototype.hasOwnProperty.call(sample, "created_at")
      );
    }
  } catch (e) {
    console.warn("Failed to classify slice", entry.slice_type, e);
  }
  return false;
}

async function bucketSlices(entries, fetchJSON) {
  const buckets = { state: [], event: [] };
  const classifications = await Promise.all(entries.map((entry) => isEventSlice(entry, fetchJSON)));
  entries.forEach((entry, i) => {
    buckets[classifications[i] ? "event" : "state"].push(entry);
  });
  return buckets;
}

// ============================================================================
// JSON Tree rendering
// ============================================================================

function getMarkdownView(state, path) {
  return state.markdownViews.get(pathKey(path)) || "html";
}

function setMarkdownView(state, path, view) {
  state.markdownViews.set(pathKey(path), view);
}

function shouldOpen(state, path, depth) {
  const key = pathKey(path);
  if (state.closedPaths.has(key)) {
    return false;
  }
  if (state.openPaths.has(key)) {
    return true;
  }
  return depth < state.expandDepth;
}

function setOpen(state, path, open) {
  const key = pathKey(path);
  if (open) {
    state.openPaths.add(key);
    state.closedPaths.delete(key);
  } else {
    state.openPaths.delete(key);
    state.closedPaths.add(key);
  }
}

function applyDepth(state, items, depth) {
  state.openPaths = new Set();
  state.closedPaths = new Set();
  const walk = (value, path, d) => {
    if (!(Array.isArray(value) || isObject(value))) {
      return;
    }
    if (d < depth) {
      setOpen(state, path, true);
    }
    if (Array.isArray(value)) {
      value.forEach((c, i) => walk(c, path.concat(String(i)), d + 1));
    } else {
      Object.entries(value).forEach(([k, v]) => walk(v, path.concat(k), d + 1));
    }
  };
  items.forEach((item, i) => walk(item, [`item-${i}`], 0));
}

function setOpenForAll(state, items, open) {
  state.openPaths = new Set();
  state.closedPaths = new Set();
  const update = (value, path) => {
    if (!(Array.isArray(value) || isObject(value))) {
      return;
    }
    setOpen(state, path, open);
    if (Array.isArray(value)) {
      value.forEach((c, i) => update(c, path.concat(String(i))));
    } else {
      Object.entries(value).forEach(([k, v]) => update(v, path.concat(k)));
    }
  };
  items.forEach((item, i) => update(item, [`item-${i}`]));
}

function getFilteredItems(state) {
  const query = state.searchQuery.toLowerCase().trim();
  if (!query) {
    return state.currentItems.map((item, index) => ({ item, index }));
  }
  return state.currentItems
    .map((item, index) => ({ item, index, text: JSON.stringify(item).toLowerCase() }))
    .filter((e) => e.text.includes(query))
    .map(({ item, index }) => ({ item, index }));
}

function getTreeTypeBadge(value, type) {
  if (type === "array") {
    return `array (${value.length})`;
  }
  if (type === "object" && value !== null) {
    return `object (${Object.keys(value).length})`;
  }
  return type;
}

function renderMarkdownLeaf(wrapper, markdown, path, state) {
  const view = getMarkdownView(state, path);
  wrapper.classList.add("markdown-leaf");
  wrapper.innerHTML = `
    <div class="markdown-toggle">
      <button type="button" class="${view === "html" ? "active" : ""}">Rendered</button>
      <button type="button" class="${view === "raw" ? "active" : ""}">Raw</button>
    </div>
    <div class="markdown-section" style="display:${view === "html" ? "flex" : "none"}">
      <div class="markdown-rendered">${markdown.html}</div>
    </div>
    <div class="markdown-section" style="display:${view === "raw" ? "flex" : "none"}">
      <pre class="markdown-raw">${escapeHtml(markdown.text)}</pre>
    </div>
  `;
  attachMarkdownToggleHandlers(wrapper, path, state);
}

function attachMarkdownToggleHandlers(wrapper, path, state) {
  const buttons = wrapper.querySelectorAll(".markdown-toggle button");
  const sections = wrapper.querySelectorAll(".markdown-section");
  buttons[0].addEventListener("click", () => {
    setMarkdownView(state, path, "html");
    buttons[0].classList.add("active");
    buttons[1].classList.remove("active");
    sections[0].style.display = "flex";
    sections[1].style.display = "none";
  });
  buttons[1].addEventListener("click", () => {
    setMarkdownView(state, path, "raw");
    buttons[1].classList.add("active");
    buttons[0].classList.remove("active");
    sections[1].style.display = "flex";
    sections[0].style.display = "none";
  });
}

function renderTreeLeaf(node, header, body, value, markdown, path, state) {
  const wrapper = document.createElement("div");
  wrapper.className = "leaf-wrapper";

  if (markdown) {
    renderMarkdownLeaf(wrapper, markdown, path, state);
  } else {
    const leaf = document.createElement("div");
    leaf.className = "tree-leaf";
    leaf.textContent = String(value);
    wrapper.appendChild(leaf);
  }
  body.appendChild(wrapper);
  node.appendChild(header);
  node.appendChild(body);
  return node;
}

function createTreeControls(path, depth, state, renderItemsFn) {
  const controls = document.createElement("div");
  controls.className = "tree-controls";

  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "tree-toggle";
  toggle.textContent = shouldOpen(state, path, depth) ? "Collapse" : "Expand";
  controls.appendChild(toggle);

  toggle.addEventListener("click", () => {
    setOpen(state, path, !shouldOpen(state, path, depth));
    renderItemsFn(state.currentItems);
  });

  return controls;
}

function renderSimpleArrayChips(container, value) {
  container.classList.add("compact-array");
  value.forEach((child) => {
    const chip = document.createElement("span");
    chip.className = "array-chip";
    chip.textContent = String(child);
    container.appendChild(chip);
  });
}

function populateTreeChildren(container, value, path, depth, state, renderItemsFn) {
  const childCount = Array.isArray(value) ? value.length : Object.keys(value).length;
  if (childCount === 0) {
    // biome-ignore lint/nursery/noSecrets: HTML string, not a secret
    container.innerHTML = '<span class="muted">(empty)</span>';
    return;
  }
  if (!shouldOpen(state, path, depth)) {
    container.style.display = "none";
    return;
  }
  if (Array.isArray(value)) {
    if (isSimpleArray(value)) {
      renderSimpleArrayChips(container, value);
    } else {
      value.forEach((child, i) => {
        container.appendChild(
          renderTree(child, path.concat(String(i)), depth + 1, `[${i}]`, state, renderItemsFn)
        );
      });
    }
  } else {
    Object.entries(value).forEach(([key, child]) => {
      container.appendChild(
        renderTree(child, path.concat(key), depth + 1, key, state, renderItemsFn)
      );
    });
  }
}

function renderTree(value, path, depth, label, state, renderItemsFn) {
  const node = document.createElement("div");
  node.className = "tree-node";

  const header = document.createElement("div");
  header.className = "tree-header";

  const markdown = getMarkdownPayload(value);
  const type = valueType(value);

  const name = document.createElement("span");
  name.className = "tree-label";
  name.textContent = label;
  header.appendChild(name);

  const badge = document.createElement("span");
  badge.className = "pill pill-quiet";
  badge.textContent = getTreeTypeBadge(value, type);
  header.appendChild(badge);

  const body = document.createElement("div");
  body.className = "tree-body";

  const expandable = !markdown && (Array.isArray(value) || isObject(value));

  if (!expandable) {
    return renderTreeLeaf(node, header, body, value, markdown, path, state);
  }

  const childCount = Array.isArray(value) ? value.length : Object.keys(value).length;
  if (childCount > 0) {
    header.appendChild(createTreeControls(path, depth, state, renderItemsFn));
  }

  const childrenContainer = document.createElement("div");
  childrenContainer.className = "tree-children";
  populateTreeChildren(childrenContainer, value, path, depth, state, renderItemsFn);

  body.appendChild(childrenContainer);
  node.appendChild(header);
  node.appendChild(body);
  return node;
}

// ============================================================================
// Sessions View initialization
// ============================================================================

/**
 * Initializes the sessions view with slice browser and JSON tree.
 *
 * @param {object} deps - Dependencies
 * @param {object} deps.state - The shared application state
 * @param {function} deps.fetchJSON - API fetch helper
 * @param {function} deps.showToast - Toast notification helper
 * @returns {{ refreshMeta, focusSearch, focusItem, navigateNext, navigatePrev }}
 */
export function initSessionsView({ state, fetchJSON, showToast }) {
  const els = {
    sliceFilter: document.getElementById("slice-filter"),
    stateSliceList: document.getElementById("state-slice-list"),
    eventSliceList: document.getElementById("event-slice-list"),
    stateSliceCount: document.getElementById("state-slice-count"),
    eventSliceCount: document.getElementById("event-slice-count"),
    sliceEmptyState: document.getElementById("slice-empty-state"),
    sliceContent: document.getElementById("slice-content"),
    sliceTitle: document.getElementById("slice-title"),
    itemCount: document.getElementById("item-count"),
    typeRow: document.getElementById("type-row"),
    jsonViewer: document.getElementById("json-viewer"),
    itemSearch: document.getElementById("item-search"),
    depthInput: document.getElementById("depth-input"),
    expandAll: document.getElementById("expand-all"),
    collapseAll: document.getElementById("collapse-all"),
    copyButton: document.getElementById("copy-button"),
  };

  // -- Rendering --

  function renderItems(items) {
    els.jsonViewer.innerHTML = "";
    const filtered = getFilteredItems(state);

    if (state.searchQuery.trim()) {
      els.itemCount.textContent = `${filtered.length} of ${items.length} items`;
    } else {
      els.itemCount.textContent = `${items.length} items`;
    }

    filtered.forEach(({ item, index }) => {
      const card = document.createElement("div");
      card.className = "item-card";
      card.innerHTML = `<div class="item-header"><h3>Item ${index + 1}</h3></div>`;

      const body = document.createElement("div");
      body.className = "item-body tree-root";
      body.appendChild(renderTree(item, [`item-${index}`], 0, "root", state, renderItems));
      card.appendChild(body);
      els.jsonViewer.appendChild(card);
    });
  }

  // -- Slice list --

  function renderSliceList() {
    const filter = els.sliceFilter.value.toLowerCase().trim();

    const renderBucket = (target, entries, emptyLabel) => {
      target.innerHTML = "";
      const filtered = entries.filter(
        (e) =>
          (e.display_name || e.slice_type).toLowerCase().includes(filter) ||
          (e.item_display_name || e.item_type).toLowerCase().includes(filter)
      );
      if (!filtered.length) {
        const li = document.createElement("li");
        li.className = "slice-item muted";
        li.textContent = emptyLabel;
        target.appendChild(li);
        return;
      }
      filtered.forEach((entry) => {
        const li = document.createElement("li");
        li.className = `slice-item${entry.slice_type === state.selectedSlice ? " active" : ""}`;
        const { className, packagePath } = splitQualifiedName(entry.slice_type);
        li.innerHTML = `
          <div class="slice-title">${escapeHtml(className)}</div>
          <div class="slice-subtitle">${escapeHtml(packagePath)}</div>
        `;
        li.addEventListener("click", () => selectSlice(entry.slice_type));
        target.appendChild(li);
      });
    };

    renderBucket(els.stateSliceList, state.sliceBuckets.state, "No state slices");
    renderBucket(els.eventSliceList, state.sliceBuckets.event, "No event slices");
    els.stateSliceCount.textContent = `${state.sliceBuckets.state.length}`;
    els.eventSliceCount.textContent = `${state.sliceBuckets.event.length}`;
  }

  async function selectSlice(sliceType) {
    state.selectedSlice = sliceType;
    renderSliceList();

    try {
      const encoded = encodeURIComponent(sliceType);
      const detail = await fetchJSON(`/api/slices/${encoded}`);
      renderSliceDetail(detail);
    } catch (error) {
      els.jsonViewer.textContent = error.message;
    }
  }

  function renderSliceDetail(slice) {
    els.sliceEmptyState.classList.add("hidden");
    els.sliceContent.classList.remove("hidden");

    els.sliceTitle.textContent = slice.display_name || slice.slice_type;
    els.itemCount.textContent = `${slice.items.length} items`;

    els.typeRow.innerHTML = `
      <span class="pill">slice: ${slice.display_name || slice.slice_type}</span>
      <span class="pill">item: ${slice.item_display_name || slice.item_type}</span>
    `;

    state.currentItems = slice.items;
    state.markdownViews = new Map();
    state.searchQuery = "";
    els.itemSearch.value = "";

    applyDepth(state, state.currentItems, state.expandDepth);
    renderItems(state.currentItems);
  }

  // -- Meta refresh --

  function firstSliceType(slices) {
    return slices.length > 0 ? slices[0].slice_type : null;
  }

  function resolveInitialSlice(slices) {
    if (!state.selectedSlice) {
      return firstSliceType(slices);
    }
    const exists = slices.some((e) => e.slice_type === state.selectedSlice);
    return exists ? state.selectedSlice : firstSliceType(slices);
  }

  async function refreshMeta() {
    const meta = await fetchJSON("/api/meta");
    state.meta = meta;
    state.sliceBuckets = await bucketSlices(meta.slices, fetchJSON);
    renderSliceList();

    const targetSlice = resolveInitialSlice(meta.slices);
    if (targetSlice) {
      await selectSlice(targetSlice);
    }
  }

  // -- Navigation --

  function focusItem(index) {
    const items = els.jsonViewer.querySelectorAll(".item-card");
    if (items.length === 0) {
      return;
    }
    const newIndex = Math.max(0, Math.min(index, items.length - 1));
    state.focusedItemIndex = newIndex;
    items.forEach((item, i) => item.classList.toggle("focused", i === newIndex));
    items[newIndex].scrollIntoView({ behavior: "smooth", block: "center" });
  }

  // -- Wire up DOM events --

  els.sliceFilter.addEventListener("input", renderSliceList);

  els.itemSearch.addEventListener("input", () => {
    state.searchQuery = els.itemSearch.value || "";
    renderItems(state.currentItems);
  });

  els.depthInput.addEventListener("change", () => {
    const value = Number(els.depthInput.value);
    const depth = Number.isFinite(value) ? Math.max(1, Math.min(10, value)) : 1;
    state.expandDepth = depth;
    els.depthInput.value = String(depth);
    applyDepth(state, state.currentItems, depth);
    renderItems(state.currentItems);
  });

  els.expandAll.addEventListener("click", () => {
    setOpenForAll(state, state.currentItems, true);
    renderItems(state.currentItems);
  });

  els.collapseAll.addEventListener("click", () => {
    setOpenForAll(state, state.currentItems, false);
    renderItems(state.currentItems);
  });

  els.copyButton.addEventListener("click", async () => {
    const text = JSON.stringify(
      getFilteredItems(state).map((e) => e.item),
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

  return {
    refreshMeta,
    focusSearch() {
      els.itemSearch.focus();
    },
    focusItem,
    navigateNext() {
      focusItem(state.focusedItemIndex + 1);
    },
    navigatePrev() {
      focusItem(state.focusedItemIndex - 1);
    },
  };
}
