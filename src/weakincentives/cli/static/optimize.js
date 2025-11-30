const promptList = document.getElementById("prompt-list");
const promptDetail = document.getElementById("prompt-detail");
const saveSnapshotButton = document.getElementById("save-snapshot");
const resetButton = document.getElementById("reset");
const statusEl = document.getElementById("status");

let prompts = [];
let activePrompt = null;

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || response.statusText);
  }
  return response.json();
}

function showStatus(message, variant = "info") {
  statusEl.textContent = message;
  statusEl.dataset.variant = variant;
}

function renderPrompts(list) {
  promptList.innerHTML = "";
  list.forEach((entry) => {
    const li = document.createElement("li");
    li.dataset.id = entry.id;
    li.innerHTML = `<div><strong>${entry.descriptor.ns}</strong> / ${entry.descriptor.key}</div>`;
    const meta = document.createElement("div");
    meta.className = "prompt-meta";
    meta.textContent = `${Object.keys(entry.overrides.sections).length} sections, ${Object.keys(entry.overrides.tools).length} tools`;
    li.appendChild(meta);
    li.addEventListener("click", () => selectPrompt(entry.id));
    promptList.appendChild(li);
  });
}

function renderDetail(detail) {
  activePrompt = detail.id;
  promptDetail.classList.remove("hidden");
  promptDetail.innerHTML = "";

  const header = document.createElement("div");
  header.className = "section-heading";
  header.innerHTML = `<strong>${detail.descriptor.ns}</strong> / ${detail.descriptor.key}`;
  promptDetail.appendChild(header);

  const sectionsHeading = document.createElement("div");
  sectionsHeading.className = "section-heading";
  sectionsHeading.textContent = "Section overrides";
  promptDetail.appendChild(sectionsHeading);

  Object.entries(detail.overrides.sections).forEach(([path, section]) => {
    const group = document.createElement("div");
    group.className = "field-group";
    const label = document.createElement("label");
    label.textContent = `${path}`;
    group.appendChild(label);
    const textarea = document.createElement("textarea");
    textarea.value = section.body || "";
    textarea.dataset.path = path;
    textarea.placeholder = "Override body";
    group.appendChild(textarea);
    promptDetail.appendChild(group);
  });

  const toolsHeading = document.createElement("div");
  toolsHeading.className = "section-heading";
  toolsHeading.textContent = "Tool overrides";
  promptDetail.appendChild(toolsHeading);

  Object.entries(detail.overrides.tools).forEach(([name, tool]) => {
    const group = document.createElement("div");
    group.className = "field-group";
    const label = document.createElement("label");
    label.textContent = name;
    group.appendChild(label);
    const desc = document.createElement("textarea");
    desc.value = tool.description || "";
    desc.dataset.tool = name;
    desc.placeholder = "Tool description override";
    group.appendChild(desc);
    promptDetail.appendChild(group);
  });

  const saveButton = document.createElement("button");
  saveButton.textContent = "Save overrides";
  saveButton.className = "primary";
  saveButton.addEventListener("click", () => savePromptOverrides(detail.id));
  promptDetail.appendChild(saveButton);
}

async function loadPrompts() {
  try {
    prompts = await fetchJSON("/api/prompts");
    renderPrompts(prompts);
    if (prompts.length) {
      selectPrompt(prompts[0].id);
    }
  } catch (error) {
    showStatus(error.message, "error");
  }
}

async function selectPrompt(id) {
  document.querySelectorAll("#prompt-list li").forEach((el) => {
    el.classList.toggle("active", el.dataset.id === id);
  });
  try {
    const detail = await fetchJSON(`/api/prompts/${id}`);
    renderDetail(detail);
    showStatus("");
  } catch (error) {
    showStatus(error.message, "error");
  }
}

async function savePromptOverrides(id) {
  const sections = {};
  const tools = {};
  promptDetail.querySelectorAll("textarea[data-path]").forEach((el) => {
    sections[el.dataset.path] = { body: el.value };
  });
  promptDetail.querySelectorAll("textarea[data-tool]").forEach((el) => {
    tools[el.dataset.tool] = { description: el.value };
  });

  try {
    await fetchJSON(`/api/prompts/${id}/overrides`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sections, tools }),
    });
    showStatus("Overrides saved", "success");
  } catch (error) {
    showStatus(error.message, "error");
  }
}

async function saveSnapshot() {
  try {
    const result = await fetchJSON("/api/save", { method: "POST" });
    showStatus(`Snapshot saved to ${result.path}`);
  } catch (error) {
    showStatus(error.message, "error");
  }
}

async function resetSnapshot() {
  try {
    prompts = await fetchJSON("/api/reset", { method: "POST" });
    renderPrompts(prompts);
    promptDetail.classList.add("hidden");
    if (prompts.length) {
      selectPrompt(prompts[0].id);
    }
    showStatus("Snapshot reloaded", "success");
  } catch (error) {
    showStatus(error.message, "error");
  }
}

saveSnapshotButton.addEventListener("click", saveSnapshot);
resetButton.addEventListener("click", resetSnapshot);

loadPrompts();
