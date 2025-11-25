const state = {
  meta: null,
  selectedSlice: null,
  snapshots: [],
  entries: [],
  currentItems: [],
  openPaths: new Set(),
  closedPaths: new Set(),
  expandDepth: 2,
  searchQuery: "",
  sliceBuckets: { state: [], event: [] },
};

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
};

async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return response.json();
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

function renderSliceDetail(slice) {
  elements.sliceTitle.textContent = slice.slice_type;
  elements.itemCount.textContent = `${slice.items.length} items`;
  elements.typeRow.innerHTML = "";
  state.currentItems = slice.items;
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
elements.copy.addEventListener("click", () => {
  const text = JSON.stringify(
    getFilteredItems().map((entry) => entry.item),
    null,
    2
  );
  navigator.clipboard.writeText(text);
});

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

document.addEventListener("DOMContentLoaded", () => {
  refreshMeta()
    .then(refreshEntries)
    .then(refreshSnapshots)
    .catch((error) => {
      elements.jsonViewer.textContent = error.message;
    });
});

const isObject = (value) => typeof value === "object" && value !== null;

function valueType(value) {
  if (Array.isArray(value)) return "array";
  if (value === null) return "null";
  return typeof value;
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

  const name = document.createElement("span");
  name.className = "tree-label";
  name.textContent = label;
  header.appendChild(name);

  const badge = document.createElement("span");
  badge.className = "pill pill-quiet";
  const type = valueType(value);
  const childCount =
    type === "array"
      ? value.length
      : type === "object" && value !== null
        ? Object.keys(value).length
        : 0;
  if (type === "array") {
    badge.textContent = `array (${value.length})`;
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
  }

  const body = document.createElement("div");
  body.className = "tree-body";

  const expandable = Array.isArray(value) || isObject(value);

  if (!expandable) {
    const wrapper = document.createElement("div");
    wrapper.className = "leaf-wrapper";

    const leaf = document.createElement("div");
    leaf.className = "tree-leaf";
    leaf.textContent = String(value);
    const toggle = applyTruncation(leaf, countLines(value));
    wrapper.appendChild(leaf);
    if (toggle) {
      wrapper.appendChild(toggle);
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
