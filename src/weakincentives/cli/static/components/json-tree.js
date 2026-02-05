// ============================================================================
// JSON Tree State - Pure state management utilities for tree views
// ============================================================================
//
// This module provides testable, pure functions for managing tree expansion
// state. The actual DOM rendering remains in app.js which closes over
// application state.
// ============================================================================

import { isObject, pathKey } from "../lib.js";

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
