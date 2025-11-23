const state = {
  meta: null,
  selectedSlice: null,
  snapshots: [],
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
  navigator.clipboard.writeText(
    Array.from(document.querySelectorAll(".item-body pre"))
      .map((node) => node.textContent || "")
      .join("\n\n")
  );
});

document.addEventListener("DOMContentLoaded", () => {
  Promise.all([refreshMeta(), refreshSnapshots()]).catch((error) => {
    elements.jsonViewer.textContent = error.message;
  });
});

function formatItems(items) {
  return JSON.stringify(items, null, 2).replace(/\\n/g, "\n");
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

    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "item-toggle";
    toggle.textContent = "Collapse";
    header.appendChild(toggle);

    const body = document.createElement("div");
    body.className = "item-body";
    const pre = document.createElement("pre");
    pre.textContent = formatItems([item]).slice(2, -2); // strip list brackets
    body.appendChild(pre);

    const setOpen = (open) => {
      body.style.display = open ? "block" : "none";
      toggle.textContent = open ? "Collapse" : "Expand";
    };
    let open = index === 0;
    toggle.addEventListener("click", () => {
      open = !open;
      setOpen(open);
    });
    header.addEventListener("click", () => {
      open = !open;
      setOpen(open);
    });
    setOpen(open);

    card.appendChild(header);
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
