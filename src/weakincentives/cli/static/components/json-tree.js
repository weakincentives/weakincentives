// ============================================================================
// JSON Tree - Expandable tree view for sessions view
// ============================================================================

import {
  escapeHtml,
  getMarkdownPayload,
  isObject,
  isSimpleArray,
  pathKey,
  valueType,
} from "../lib.js";

/**
 * Creates a JSON tree renderer with state management callbacks.
 * @param {Object} options
 * @param {function(string[], number): boolean} options.shouldOpen - Check if path should be expanded
 * @param {function(string[], boolean): void} options.setOpen - Set path expansion state
 * @param {function(string[]): string} options.getMarkdownView - Get markdown view mode for path
 * @param {function(string[], string): void} options.setMarkdownView - Set markdown view mode
 * @param {function(): void} options.onRender - Callback to trigger re-render
 * @returns {Object} Tree renderer with renderTree and helper methods
 */
export function createTreeRenderer(options) {
  const { shouldOpen, setOpen, getMarkdownView, setMarkdownView, onRender } = options;

  /**
   * Creates the type badge text for tree nodes.
   * @param {*} value
   * @param {string} type
   * @returns {string}
   */
  function getTreeTypeBadge(value, type) {
    if (type === "array") {
      return `array (${value.length})`;
    }
    if (type === "object" && value !== null) {
      return `object (${Object.keys(value).length})`;
    }
    return type;
  }

  /**
   * Renders a markdown leaf node with toggle between rendered and raw views.
   * @param {HTMLElement} wrapper
   * @param {{text: string, html: string}} markdown
   * @param {string[]} path
   */
  function renderMarkdownLeaf(wrapper, markdown, path) {
    const view = getMarkdownView(path);
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
    attachMarkdownToggleHandlers(wrapper, path);
  }

  /**
   * Attaches click handlers for markdown view toggle buttons.
   * @param {HTMLElement} wrapper
   * @param {string[]} path
   */
  function attachMarkdownToggleHandlers(wrapper, path) {
    const buttons = wrapper.querySelectorAll(".markdown-toggle button");
    const sections = wrapper.querySelectorAll(".markdown-section");
    buttons[0].addEventListener("click", () => {
      setMarkdownView(path, "html");
      buttons[0].classList.add("active");
      buttons[1].classList.remove("active");
      sections[0].style.display = "flex";
      sections[1].style.display = "none";
    });
    buttons[1].addEventListener("click", () => {
      setMarkdownView(path, "raw");
      buttons[1].classList.add("active");
      buttons[0].classList.remove("active");
      sections[1].style.display = "flex";
      sections[0].style.display = "none";
    });
  }

  /**
   * Renders a non-expandable leaf node (markdown or simple value).
   * @param {HTMLElement} node
   * @param {HTMLElement} header
   * @param {HTMLElement} body
   * @param {*} value
   * @param {{text: string, html: string}|null} markdown
   * @param {string[]} path
   * @returns {HTMLElement}
   */
  function renderTreeLeaf(node, header, body, value, markdown, path) {
    const wrapper = document.createElement("div");
    wrapper.className = "leaf-wrapper";

    if (markdown) {
      renderMarkdownLeaf(wrapper, markdown, path);
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

  /**
   * Creates the expand/collapse controls for tree nodes.
   * @param {string[]} path
   * @param {number} depth
   * @returns {HTMLElement}
   */
  function createTreeControls(path, depth) {
    const controls = document.createElement("div");
    controls.className = "tree-controls";

    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "tree-toggle";
    toggle.textContent = shouldOpen(path, depth) ? "Collapse" : "Expand";
    controls.appendChild(toggle);

    toggle.addEventListener("click", () => {
      setOpen(path, !shouldOpen(path, depth));
      onRender();
    });

    return controls;
  }

  /**
   * Renders simple array as compact chips.
   * @param {HTMLElement} container
   * @param {Array} value
   */
  function renderSimpleArrayChips(container, value) {
    container.classList.add("compact-array");
    value.forEach((child) => {
      const chip = document.createElement("span");
      chip.className = "array-chip";
      chip.textContent = String(child);
      container.appendChild(chip);
    });
  }

  /**
   * Renders array children in the tree.
   * @param {HTMLElement} container
   * @param {Array} value
   * @param {string[]} path
   * @param {number} depth
   */
  function renderArrayChildren(container, value, path, depth) {
    if (isSimpleArray(value)) {
      renderSimpleArrayChips(container, value);
      return;
    }
    value.forEach((child, i) => {
      container.appendChild(renderTree(child, path.concat(String(i)), depth + 1, `[${i}]`));
    });
  }

  /**
   * Renders object children in the tree.
   * @param {HTMLElement} container
   * @param {Object} value
   * @param {string[]} path
   * @param {number} depth
   */
  function renderObjectChildren(container, value, path, depth) {
    Object.entries(value).forEach(([key, child]) => {
      container.appendChild(renderTree(child, path.concat(key), depth + 1, key));
    });
  }

  /**
   * Gets the child count for an array or object.
   * @param {*} value
   * @returns {number}
   */
  function getChildCount(value) {
    return Array.isArray(value) ? value.length : Object.keys(value).length;
  }

  /**
   * Populates tree children container based on value type and expansion state.
   * @param {HTMLElement} container
   * @param {*} value
   * @param {string[]} path
   * @param {number} depth
   */
  function populateTreeChildren(container, value, path, depth) {
    if (getChildCount(value) === 0) {
      // biome-ignore lint/nursery/noSecrets: HTML string, not a secret
      container.innerHTML = '<span class="muted">(empty)</span>';
      return;
    }
    if (!shouldOpen(path, depth)) {
      container.style.display = "none";
      return;
    }
    if (Array.isArray(value)) {
      renderArrayChildren(container, value, path, depth);
    } else {
      renderObjectChildren(container, value, path, depth);
    }
  }

  /**
   * Renders a tree node for the given value.
   * @param {*} value - The value to render
   * @param {string[]} path - The path to this node
   * @param {number} depth - Current depth
   * @param {string} label - Display label for this node
   * @returns {HTMLElement}
   */
  function renderTree(value, path, depth, label) {
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
      return renderTreeLeaf(node, header, body, value, markdown, path);
    }

    const childCount = Array.isArray(value) ? value.length : Object.keys(value).length;
    if (childCount > 0) {
      header.appendChild(createTreeControls(path, depth));
    }

    const childrenContainer = document.createElement("div");
    childrenContainer.className = "tree-children";
    populateTreeChildren(childrenContainer, value, path, depth);

    body.appendChild(childrenContainer);
    node.appendChild(header);
    node.appendChild(body);
    return node;
  }

  return { renderTree };
}

/**
 * Applies depth-based expansion to items.
 * @param {Array} items - Items to process
 * @param {number} depth - Max depth to expand
 * @param {function(string[], boolean): void} setOpen - Callback to set open state
 */
export function applyDepth(items, depth, setOpen) {
  const walk = (value, path, d) => {
    if (!(Array.isArray(value) || isObject(value))) {
      return;
    }
    if (d < depth) {
      setOpen(path, true);
    }
    if (Array.isArray(value)) {
      value.forEach((c, i) => walk(c, path.concat(String(i)), d + 1));
    } else {
      Object.entries(value).forEach(([k, v]) => walk(v, path.concat(k), d + 1));
    }
  };
  items.forEach((item, i) => walk(item, [`item-${i}`], 0));
}

/**
 * Sets open state for all nodes in items.
 * @param {Array} items - Items to process
 * @param {boolean} open - Whether to open or close
 * @param {function(string[], boolean): void} setOpen - Callback to set open state
 */
export function setOpenForAll(items, open, setOpen) {
  const update = (value, path) => {
    if (!(Array.isArray(value) || isObject(value))) {
      return;
    }
    setOpen(path, open);
    if (Array.isArray(value)) {
      value.forEach((c, i) => update(c, path.concat(String(i))));
    } else {
      Object.entries(value).forEach(([k, v]) => update(v, path.concat(k)));
    }
  };
  items.forEach((item, i) => update(item, [`item-${i}`]));
}

/**
 * Creates path-based state management helpers.
 * @returns {Object} State management object
 */
export function createPathState() {
  let openPaths = new Set();
  let closedPaths = new Set();
  let expandDepth = 2;

  return {
    shouldOpen(path, depth) {
      const key = pathKey(path);
      if (closedPaths.has(key)) {
        return false;
      }
      if (openPaths.has(key)) {
        return true;
      }
      return depth < expandDepth;
    },

    setOpen(path, open) {
      const key = pathKey(path);
      if (open) {
        openPaths.add(key);
        closedPaths.delete(key);
      } else {
        openPaths.delete(key);
        closedPaths.add(key);
      }
    },

    reset() {
      openPaths = new Set();
      closedPaths = new Set();
    },

    setExpandDepth(depth) {
      expandDepth = depth;
    },

    getExpandDepth() {
      return expandDepth;
    },
  };
}

/**
 * Creates markdown view state management helpers.
 * @returns {Object} Markdown view state object
 */
export function createMarkdownViewState() {
  const markdownViews = new Map();

  return {
    getMarkdownView(path) {
      return markdownViews.get(pathKey(path)) || "html";
    },

    setMarkdownView(path, view) {
      markdownViews.set(pathKey(path), view);
    },

    reset() {
      markdownViews.clear();
    },
  };
}
