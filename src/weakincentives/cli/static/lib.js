// ============================================================================
// Pure utility functions for the debug app
// Extracted for testability with Bun
// ============================================================================

/**
 * Format bytes as human-readable string.
 * @param {number} bytes
 * @returns {string}
 */
export function formatBytes(bytes) {
  if (bytes === 0) {
    return "0 B";
  }
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${Math.round((bytes / k ** i) * 100) / 100} ${sizes[i]}`;
}

/**
 * Escape HTML special characters.
 * @param {string} text
 * @returns {string}
 */
export function escapeHtml(text) {
  const map = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  };
  return String(text).replace(/[&<>"']/g, (c) => map[c]);
}

/**
 * Check if value is a non-null object.
 * @param {unknown} v
 * @returns {boolean}
 */
export function isObject(v) {
  return typeof v === "object" && v !== null;
}

/**
 * Check if value is a primitive (null, string, number, boolean, etc.).
 * @param {unknown} v
 * @returns {boolean}
 */
export function isPrimitive(v) {
  return v === null || (typeof v !== "object" && !Array.isArray(v));
}

/**
 * Check if value is an array containing only primitives.
 * @param {unknown} v
 * @returns {boolean}
 */
export function isSimpleArray(v) {
  return Array.isArray(v) && v.every(isPrimitive);
}

/**
 * Determine the display type of a value.
 * @param {unknown} value
 * @param {string} markdownKey - The key used to identify markdown payloads
 * @returns {string}
 */
export function valueType(value, markdownKey = "__markdown__") {
  if (getMarkdownPayload(value, markdownKey)) {
    return "markdown";
  }
  if (Array.isArray(value)) {
    return "array";
  }
  if (value === null) {
    return "null";
  }
  return typeof value;
}

/**
 * Extract markdown payload from an object if it exists.
 * @param {unknown} value
 * @param {string} markdownKey
 * @returns {{text: string, html: string} | null}
 */
export function getMarkdownPayload(value, markdownKey = "__markdown__") {
  if (
    !value ||
    typeof value !== "object" ||
    Array.isArray(value) ||
    !Object.prototype.hasOwnProperty.call(value, markdownKey)
  ) {
    return null;
  }
  const payload = value[markdownKey];
  return payload && typeof payload.text === "string" && typeof payload.html === "string"
    ? payload
    : null;
}

/**
 * Convert a path array to a dot-separated key string.
 * @param {string[]} path
 * @returns {string}
 */
export function pathKey(path) {
  return path.join(".");
}

/**
 * Build query string from filter state.
 * @param {Object} filters - Filter configuration
 * @param {string[]} filters.include - Values to include
 * @param {string[]} filters.exclude - Values to exclude
 * @param {string} paramName - Query parameter name
 * @returns {string} Query string fragment (without leading &)
 */
export function buildFilterParams(filters, paramName) {
  const parts = [];
  if (filters.include.length > 0) {
    parts.push(`${paramName}=${filters.include.map(encodeURIComponent).join(",")}`);
  }
  if (filters.exclude.length > 0) {
    parts.push(`${paramName}_exclude=${filters.exclude.map(encodeURIComponent).join(",")}`);
  }
  return parts.join("&");
}

/**
 * Toggle a filter value between include/exclude/neither states.
 * Shift+click adds to exclude, regular click adds to include.
 * Clicking an already-active filter removes it.
 *
 * @param {string} name - The filter value to toggle
 * @param {string[]} include - Current include list
 * @param {string[]} exclude - Current exclude list
 * @returns {{include: string[], exclude: string[]}} New filter state
 */
export function toggleFilter(name, include, exclude) {
  const inInclude = include.includes(name);
  const inExclude = exclude.includes(name);

  if (inInclude) {
    // Remove from include
    return {
      include: include.filter((n) => n !== name),
      exclude,
    };
  }
  if (inExclude) {
    // Remove from exclude
    return {
      include,
      exclude: exclude.filter((n) => n !== name),
    };
  }
  // Add to include by default
  return {
    include: [...include, name],
    exclude,
  };
}

/**
 * Toggle a filter value with explicit include/exclude control.
 *
 * @param {string} name - The filter value to toggle
 * @param {string[]} currentInclude - Current include list
 * @param {string[]} currentExclude - Current exclude list
 * @param {boolean} toInclude - Whether to add to include list
 * @param {boolean} toExclude - Whether to add to exclude list
 * @returns {{include: string[], exclude: string[]}} New filter state
 */
export function toggleFilterExplicit(name, currentInclude, currentExclude, toInclude, toExclude) {
  const inInclude = currentInclude.includes(name);
  const inExclude = currentExclude.includes(name);

  let include = [...currentInclude];
  let exclude = [...currentExclude];

  // Remove from current list if present
  if (inInclude) {
    include = include.filter((n) => n !== name);
  }
  if (inExclude) {
    exclude = exclude.filter((n) => n !== name);
  }

  // If it was already in the target list, just remove it (toggle off)
  if ((toInclude && inInclude) || (toExclude && inExclude)) {
    return { include, exclude };
  }

  // Add to target list
  if (toInclude) {
    include.push(name);
  } else if (toExclude) {
    exclude.push(name);
  }

  return { include, exclude };
}

/**
 * Filter items by a search query against their JSON representation.
 * @param {unknown[]} items - Items to filter
 * @param {string} query - Search query
 * @returns {{item: unknown, index: number}[]} Filtered items with original indices
 */
export function filterItemsBySearch(items, query) {
  const q = query.toLowerCase().trim();
  if (!q) {
    return items.map((item, index) => ({ item, index }));
  }
  return items
    .map((item, index) => ({ item, index, text: JSON.stringify(item).toLowerCase() }))
    .filter((e) => e.text.includes(q))
    .map(({ item, index }) => ({ item, index }));
}

/**
 * Calculate visible range for virtual scrolling.
 * @param {Object} params
 * @param {number} params.scrollTop - Current scroll position
 * @param {number} params.viewportHeight - Height of viewport
 * @param {number} params.itemCount - Total number of items
 * @param {number} params.bufferSize - Items to keep above/below viewport
 * @param {function(number): number} params.getItemHeight - Function to get item height by index
 * @returns {{startIndex: number, endIndex: number}}
 */
export function calculateVisibleRange({
  scrollTop,
  viewportHeight,
  itemCount,
  bufferSize,
  getItemHeight,
}) {
  if (itemCount === 0) {
    return { startIndex: 0, endIndex: 0 };
  }

  // Find start index (first item visible in viewport)
  // Invariant: current = sum of heights of items 0 to (firstVisibleIndex - 1)
  let current = 0;
  let firstVisibleIndex = 0;
  let foundVisible = false;

  for (let i = 0; i < itemCount; i++) {
    const height = getItemHeight(i);
    if (current + height > scrollTop) {
      firstVisibleIndex = i;
      foundVisible = true;
      break;
    }
    current += height;
  }

  // Edge case: scrolled past all content
  // Loop completed without breaking, so current = total height of all items
  // Adjust to maintain invariant: current should exclude firstVisibleIndex's height
  if (!foundVisible) {
    firstVisibleIndex = itemCount - 1;
    current -= getItemHeight(firstVisibleIndex);
  }

  // Find end index by continuing from where we left off
  const viewportBottom = scrollTop + viewportHeight;
  let endIndex = itemCount;
  for (let i = firstVisibleIndex; i < itemCount; i++) {
    current += getItemHeight(i);
    if (current > viewportBottom) {
      endIndex = i + 1;
      break;
    }
  }

  // Apply buffer and clamp to valid range
  const startIndex = Math.max(0, firstVisibleIndex - bufferSize);
  endIndex = Math.min(itemCount, endIndex + bufferSize);

  return { startIndex, endIndex };
}

/**
 * Get cumulative offset for an item at given index.
 * @param {number} index - Target index
 * @param {number} itemCount - Total items
 * @param {function(number): number} getItemHeight - Function to get item height
 * @returns {number}
 */
export function getOffsetForIndex(index, itemCount, getItemHeight) {
  let offset = 0;
  for (let i = 0; i < index && i < itemCount; i++) {
    offset += getItemHeight(i);
  }
  return offset;
}

/**
 * Get total height of all items.
 * @param {number} itemCount - Total items
 * @param {function(number): number} getItemHeight - Function to get item height
 * @returns {number}
 */
export function getTotalHeight(itemCount, getItemHeight) {
  let total = 0;
  for (let i = 0; i < itemCount; i++) {
    total += getItemHeight(i);
  }
  return total;
}
