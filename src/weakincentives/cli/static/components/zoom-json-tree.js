// ============================================================================
// Zoom JSON Tree - Collapsible JSON tree for zoom modal
// ============================================================================

import { escapeHtml } from "../lib.js";

/**
 * Renders the key portion of a JSON node.
 * @param {string} key
 * @returns {string}
 */
function zoomJsonKeyHtml(key) {
  return key ? `<span class="zoom-json-key">${escapeHtml(key)}</span>: ` : "";
}

/**
 * Renders a primitive JSON value (null, boolean, number).
 * @param {HTMLElement} node
 * @param {string} key
 * @param {*} value
 * @param {string} cssClass
 */
function renderZoomJsonPrimitive(node, key, value, cssClass) {
  const valueHtml = value === null ? "null" : String(value);
  node.innerHTML = `${zoomJsonKeyHtml(key)}<span class="${cssClass}">${valueHtml}</span>`;
}

/**
 * Attempts to parse a string as JSON and render it if successful.
 * Returns the wrapper node if parsed, null otherwise.
 * @param {string} value
 * @param {string} key
 * @param {number} depth
 * @returns {HTMLElement|null}
 */
function tryRenderParsedJsonString(value, key, depth) {
  if (value.length <= 1) {
    return null;
  }
  const looksLikeJson =
    (value.startsWith("{") && value.endsWith("}")) ||
    (value.startsWith("[") && value.endsWith("]"));
  if (!looksLikeJson) {
    return null;
  }
  try {
    const parsed = JSON.parse(value);
    const wrapper = document.createElement("div");
    wrapper.className = "zoom-json-node";
    if (key) {
      const keyLabel = document.createElement("span");
      keyLabel.innerHTML = `<span class="zoom-json-key">${escapeHtml(key)}</span>: <span class="zoom-json-parsed-hint">(parsed JSON string)</span>`;
      wrapper.appendChild(keyLabel);
    }
    wrapper.appendChild(renderZoomJsonTree(parsed, "", depth));
    return wrapper;
  } catch {
    return null;
  }
}

/**
 * Renders a string value in the JSON tree.
 * @param {HTMLElement} node
 * @param {string} key
 * @param {string} value
 * @param {number} depth
 * @returns {HTMLElement}
 */
function renderZoomJsonString(node, key, value, depth) {
  const parsedNode = tryRenderParsedJsonString(value, key, depth);
  if (parsedNode) {
    return parsedNode;
  }
  const escaped = escapeHtml(value);
  const formatted = escaped.replace(/\n/g, "<br>");
  node.innerHTML = `${zoomJsonKeyHtml(key)}<span class="zoom-json-string">"${formatted}"</span>`;
  return node;
}

/**
 * Attaches toggle click handler for collapsible JSON tree nodes.
 * @param {HTMLElement} header
 * @param {HTMLElement} children
 */
function attachZoomJsonToggle(header, children) {
  header.addEventListener("click", () => {
    const isOpen = children.style.display !== "none";
    children.style.display = isOpen ? "none" : "block";
    header.querySelector(".zoom-json-toggle").textContent = isOpen ? "▶" : "▼";
    header.querySelector(".zoom-json-close-bracket").style.display = isOpen ? "inline" : "none";
  });
}

/**
 * Creates the header element for collapsible JSON tree nodes.
 * @param {string} key
 * @param {string} openBracket
 * @param {number} count
 * @param {string} countLabel
 * @param {boolean} isExpanded
 * @returns {HTMLElement}
 */
function createZoomJsonHeader(key, openBracket, count, countLabel, isExpanded) {
  const header = document.createElement("div");
  header.className = "zoom-json-header";
  const toggle = isExpanded ? "▼" : "▶";
  const closeBracket = openBracket === "[" ? "]" : "}";
  header.innerHTML = `<span class="zoom-json-toggle">${toggle}</span>${zoomJsonKeyHtml(key)}<span class="zoom-json-bracket">${openBracket}</span><span class="zoom-json-count">${count} ${countLabel}</span><span class="zoom-json-bracket zoom-json-close-bracket" style="display: ${isExpanded ? "none" : "inline"}">${closeBracket}</span>`;
  return header;
}

/**
 * Creates children container and close bracket for collapsible nodes.
 * @param {string} closeBracket
 * @param {boolean} isExpanded
 * @returns {{children: HTMLElement, closeBracketEl: HTMLElement}}
 */
function createZoomJsonChildren(closeBracket, isExpanded) {
  const children = document.createElement("div");
  children.className = "zoom-json-children";
  children.style.display = isExpanded ? "block" : "none";

  const closeBracketEl = document.createElement("div");
  closeBracketEl.className = "zoom-json-close";
  closeBracketEl.innerHTML = `<span class="zoom-json-bracket">${closeBracket}</span>`;
  closeBracketEl.style.display = isExpanded ? "block" : "none";

  return { children, closeBracketEl };
}

/**
 * Renders an array value in the JSON tree.
 * @param {HTMLElement} node
 * @param {string} key
 * @param {Array} value
 * @param {number} depth
 * @returns {HTMLElement}
 */
function renderZoomJsonArray(node, key, value, depth) {
  const isExpanded = depth < 2;
  const header = createZoomJsonHeader(key, "[", value.length, "items", isExpanded);
  node.appendChild(header);

  const { children, closeBracketEl } = createZoomJsonChildren("]", isExpanded);
  for (let i = 0; i < value.length; i++) {
    children.appendChild(renderZoomJsonTree(value[i], String(i), depth + 1));
  }
  children.appendChild(closeBracketEl);
  node.appendChild(children);
  attachZoomJsonToggle(header, children);
  return node;
}

/**
 * Renders an object value in the JSON tree.
 * @param {HTMLElement} node
 * @param {string} key
 * @param {Object} value
 * @param {number} depth
 * @returns {HTMLElement}
 */
function renderZoomJsonObject(node, key, value, depth) {
  const keys = Object.keys(value);
  const isExpanded = depth < 2;
  const header = createZoomJsonHeader(key, "{", keys.length, "keys", isExpanded);
  node.appendChild(header);

  const { children, closeBracketEl } = createZoomJsonChildren("}", isExpanded);
  for (const k of keys) {
    children.appendChild(renderZoomJsonTree(value[k], k, depth + 1));
  }
  children.appendChild(closeBracketEl);
  node.appendChild(children);
  attachZoomJsonToggle(header, children);
  return node;
}

/**
 * Renders a collapsible JSON tree for the zoom modal details panel.
 * @param {*} value - The value to render
 * @param {string} key - The key label (empty string for root)
 * @param {number} depth - Current depth for expansion control
 * @returns {HTMLElement}
 */
export function renderZoomJsonTree(value, key, depth) {
  const node = document.createElement("div");
  node.className = "zoom-json-node";

  if (value === null) {
    renderZoomJsonPrimitive(node, key, null, "zoom-json-null");
    return node;
  }
  if (typeof value === "boolean") {
    renderZoomJsonPrimitive(node, key, value, "zoom-json-bool");
    return node;
  }
  if (typeof value === "number") {
    renderZoomJsonPrimitive(node, key, value, "zoom-json-number");
    return node;
  }
  if (typeof value === "string") {
    return renderZoomJsonString(node, key, value, depth);
  }
  if (Array.isArray(value)) {
    return renderZoomJsonArray(node, key, value, depth);
  }
  if (typeof value === "object") {
    return renderZoomJsonObject(node, key, value, depth);
  }

  node.textContent = String(value);
  return node;
}
