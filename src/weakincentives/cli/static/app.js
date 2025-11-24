const state = {
  meta: null,
  selectedSlice: null,
  snapshots: [],
  currentItems: [],
  openPaths: new Set(),
  closedPaths: new Set(),
  expandDepth: 2,
};

const elements = {
  path: document.getElementById("snapshot-path"),
  snapshotSelect: document.getElementById("snapshot-select"),
  created: document.getElementById("meta-created"),
  version: document.getElementById("meta-version"),
  count: document.getElementById("meta-count"),
  sliceList: document.getElementById("slice-list"),
  sliceFilter: document.getElementById("slice-filter"),
  sliceTitle: document.getElementById("slice-title"),
  itemCount: document.getElementById("item-count"),
  typeRow: document.getElementById("type-row"),
  jsonViewer: document.getElementById("json-viewer"),
  reload: document.getElementById("reload-button"),
  copy: document.getElementById("copy-button"),
  depthInput: document.getElementById("depth-input"),
  expandAll: document.getElementById("expand-all"),
  collapseAll: document.getElementById("collapse-all"),
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
}

function renderSliceList() {
  const filter = elements.sliceFilter.value.toLowerCase().trim();
  elements.sliceList.innerHTML = "";
  const slices = state.meta ? state.meta.slices : [];

  slices
    .filter(
      (entry) =>
        entry.slice_type.toLowerCase().includes(filter) ||
        entry.item_type.toLowerCase().includes(filter)
    )
    .forEach((entry) => {
      const item = document.createElement("li");
      item.className =
        "slice-item" +
        (entry.slice_type === state.selectedSlice ? " active" : "");

      const title = document.createElement("div");
      title.className = "slice-title";
      title.textContent = entry.slice_type;

      const subtitle = document.createElement("div");
      subtitle.className = "slice-subtitle";
      subtitle.textContent = `${entry.item_type} · ${entry.count} items`;

      item.appendChild(title);
      item.appendChild(subtitle);
      item.addEventListener("click", () => selectSlice(entry.slice_type));
      elements.sliceList.appendChild(item);
    });
}

function renderSliceDetail(slice) {
  elements.sliceTitle.textContent = slice.slice_type;
  elements.itemCount.textContent = `${slice.items.length} items`;
  elements.typeRow.innerHTML = "";
  state.currentItems = slice.items;
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
  const text = JSON.stringify(state.currentItems, null, 2);
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

document.addEventListener("DOMContentLoaded", () => {
  Promise.all([refreshMeta(), refreshSnapshots()]).catch((error) => {
    elements.jsonViewer.textContent = error.message;
  });
});

const isObject = (value) => typeof value === "object" && value !== null;

function valueType(value) {
  if (Array.isArray(value)) return "array";
  if (value === null) return "null";
  return typeof value;
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
  if (type === "array") {
    badge.textContent = `array (${value.length})`;
  } else if (type === "object" && value !== null) {
    badge.textContent = `object (${Object.keys(value).length})`;
  } else {
    badge.textContent = type;
  }
  header.appendChild(badge);

  const childControls = document.createElement("div");
  childControls.className = "tree-controls";
  header.appendChild(childControls);

  const body = document.createElement("div");
  body.className = "tree-body";

  const expandable = Array.isArray(value) || isObject(value);

  if (!expandable) {
    const leaf = document.createElement("div");
    leaf.className = "tree-leaf";
    leaf.textContent = String(value);
    body.appendChild(leaf);
    node.appendChild(header);
    node.appendChild(body);
    return node;
  }

  const open = shouldOpen(path, depth);
  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "tree-toggle";
  toggle.textContent = open ? "Collapse" : "Expand";
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

  const childrenContainer = document.createElement("div");
  childrenContainer.className = "tree-children";

  const renderChildren = () => {
    childrenContainer.innerHTML = "";
    if (!shouldOpen(path, depth)) {
      childrenContainer.style.display = "none";
      toggle.textContent = "Expand";
      return;
    }
    childrenContainer.style.display = "block";
    toggle.textContent = "Collapse";
    if (Array.isArray(value)) {
      value.forEach((child, index) => {
        childrenContainer.appendChild(
          renderTree(child, path.concat(String(index)), depth + 1, `[${index}]`)
        );
      });
    } else {
      Object.entries(value).forEach(([key, child]) => {
        childrenContainer.appendChild(
          renderTree(child, path.concat(key), depth + 1, key)
        );
      });
    }
  };

  toggle.addEventListener("click", () => {
    const next = !shouldOpen(path, depth);
    setOpen(path, next);
    renderItems(state.currentItems);
  });

  expandLevel.addEventListener("click", () => {
    expandChildren(value, path, true);
    renderItems(state.currentItems);
  });

  collapseLevel.addEventListener("click", () => {
    expandChildren(value, path, false);
    renderItems(state.currentItems);
  });

  renderChildren();

  body.appendChild(childrenContainer);
  node.appendChild(header);
  node.appendChild(body);
  return node;
}

function renderItems(items) {
  elements.jsonViewer.innerHTML = "";
  items.forEach((item, index) => {
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

async function switchSnapshot(path) {
  try {
    await fetchJSON("/api/switch", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ path }),
    });
    await refreshMeta();
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
