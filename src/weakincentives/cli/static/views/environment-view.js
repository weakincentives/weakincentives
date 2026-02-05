// ============================================================================
// Environment View - System, Python, Git & Container information display
// ============================================================================

import { escapeHtml, formatBytes } from "../lib.js";

// ============================================================================
// Section renderers
// ============================================================================

function renderEnvSection(title, content) {
  return `<div class="environment-section"><h3 class="section-title">${title}</h3><dl class="key-value-list">${content}</dl></div>`;
}

function renderSystemSection(sys) {
  if (!sys) {
    return "";
  }
  return renderEnvSection(
    "System",
    `<dt>OS</dt><dd>${escapeHtml(sys.os_name)} ${escapeHtml(sys.os_release)}</dd>` +
      `<dt>Kernel</dt><dd class="mono">${escapeHtml(sys.kernel_version)}</dd>` +
      `<dt>Architecture</dt><dd>${escapeHtml(sys.architecture)}</dd>` +
      `<dt>Processor</dt><dd>${escapeHtml(sys.processor)}</dd>` +
      `<dt>CPU Count</dt><dd>${sys.cpu_count}</dd>` +
      `<dt>Memory</dt><dd>${formatBytes(sys.memory_total_bytes)}</dd>` +
      `<dt>Hostname</dt><dd class="mono">${escapeHtml(sys.hostname)}</dd>`
  );
}

function renderPythonSection(py) {
  if (!py) {
    return "";
  }
  let content = `<dt>Version</dt><dd class="mono">${escapeHtml(py.version)}</dd>`;
  if (py.version_info) {
    content += `<dt>Version Info</dt><dd class="mono">${JSON.stringify(py.version_info)}</dd>`;
  }
  content +=
    `<dt>Implementation</dt><dd>${escapeHtml(py.implementation)}</dd>` +
    `<dt>Executable</dt><dd class="mono">${escapeHtml(py.executable)}</dd>` +
    `<dt>Prefix</dt><dd class="mono">${escapeHtml(py.prefix)}</dd>` +
    `<dt>Base Prefix</dt><dd class="mono">${escapeHtml(py.base_prefix)}</dd>` +
    `<dt>Virtualenv</dt><dd>${py.is_virtualenv ? "Yes" : "No"}</dd>`;
  return renderEnvSection("Python", content);
}

function renderGitSection(git) {
  if (!git) {
    return "";
  }
  let content =
    `<dt>Repo Root</dt><dd class="mono">${escapeHtml(git.repo_root)}</dd>` +
    `<dt>Branch</dt><dd class="mono">${escapeHtml(git.branch)}</dd>` +
    `<dt>Commit</dt><dd class="mono">${escapeHtml(git.commit_short)}</dd>` +
    `<dt>Full SHA</dt><dd class="mono small">${escapeHtml(git.commit_sha)}</dd>` +
    `<dt>Dirty</dt><dd>${git.is_dirty ? "Yes" : "No"}</dd>`;
  if (git.remotes && Object.keys(git.remotes).length > 0) {
    const remoteItems = Object.entries(git.remotes)
      .map(
        ([name, url]) =>
          `<div><span class="mono">${escapeHtml(name)}:</span> ${escapeHtml(url)}</div>`
      )
      .join("");
    content += `<dt>Remotes</dt><dd><div class="nested-list">${remoteItems}</div></dd>`;
  }
  if (git.tags && git.tags.length > 0) {
    content += `<dt>Tags</dt><dd class="mono">${git.tags.map((t) => escapeHtml(t)).join(", ")}</dd>`;
  }
  return renderEnvSection("Git Repository", content);
}

function renderContainerSection(container) {
  if (!container) {
    return "";
  }
  let content =
    `<dt>Runtime</dt><dd>${escapeHtml(container.runtime)}</dd>` +
    `<dt>Container ID</dt><dd class="mono">${escapeHtml(container.container_id)}</dd>` +
    `<dt>Image</dt><dd class="mono">${escapeHtml(container.image)}</dd>`;
  if (container.image_digest) {
    content += `<dt>Image Digest</dt><dd class="mono small">${escapeHtml(container.image_digest)}</dd>`;
  }
  if (container.cgroup_path) {
    content += `<dt>Cgroup Path</dt><dd class="mono">${escapeHtml(container.cgroup_path)}</dd>`;
  }
  return renderEnvSection("Container", content);
}

function renderEnvVarsSection(envVars) {
  if (!envVars || Object.keys(envVars).length === 0) {
    return "";
  }
  const content = Object.entries(envVars)
    .map(
      ([key, value]) =>
        `<dt class="mono">${escapeHtml(key)}</dt><dd class="mono small">${escapeHtml(value)}</dd>`
    )
    .join("");
  return renderEnvSection("Environment Variables", content);
}

// ============================================================================
// Environment View initialization
// ============================================================================

/**
 * Initializes the environment view.
 *
 * @param {object} deps - Dependencies
 * @param {object} deps.state - The shared application state
 * @param {function} deps.fetchJSON - API fetch helper
 * @param {function} deps.showToast - Toast notification helper
 * @returns {{ loadEnvironment }}
 */
export function initEnvironmentView({ state, fetchJSON, showToast }) {
  const els = {
    emptyState: document.getElementById("environment-empty-state"),
    content: document.getElementById("environment-content"),
    data: document.getElementById("environment-data"),
    copy: document.getElementById("environment-copy"),
  };

  // -- Data loading --

  async function loadEnvironment() {
    try {
      const data = await fetchJSON("/api/environment");
      state.environmentData = data;
      state.hasEnvironmentData =
        data.system !== null ||
        data.python !== null ||
        data.git !== null ||
        data.container !== null;
      render();
    } catch (error) {
      els.data.innerHTML = `<p class="muted">Failed to load environment data: ${error.message}</p>`;
    }
  }

  // -- Rendering --

  function render() {
    if (!state.hasEnvironmentData) {
      els.emptyState.classList.remove("hidden");
      els.content.classList.add("hidden");
      return;
    }
    els.emptyState.classList.add("hidden");
    els.content.classList.remove("hidden");

    const data = state.environmentData;
    els.data.innerHTML =
      renderSystemSection(data.system) +
      renderPythonSection(data.python) +
      renderGitSection(data.git) +
      renderContainerSection(data.container) +
      renderEnvVarsSection(data.env_vars);
  }

  // -- Wire up DOM events --

  els.copy.addEventListener("click", async () => {
    if (!state.environmentData) {
      showToast("No environment data to copy", "error");
      return;
    }
    try {
      await navigator.clipboard.writeText(JSON.stringify(state.environmentData, null, 2));
      showToast("Copied environment data to clipboard", "success");
    } catch {
      showToast("Failed to copy", "error");
    }
  });

  return {
    loadEnvironment,
  };
}
