// ============================================================================
// Filter Chips - Reusable filter chip components
// ============================================================================

import { escapeHtml } from "../lib.js";

/**
 * Creates a filter chip for include/exclude filtering.
 * @param {string} name - The filter value name
 * @param {number} count - The count to display
 * @param {boolean} isIncluded - Whether the filter is in include state
 * @param {boolean} isExcluded - Whether the filter is in exclude state
 * @param {function(string, boolean, boolean): void} onToggle - Callback (name, toInclude, toExclude)
 * @returns {HTMLElement}
 */
export function createFilterChip(name, count, isIncluded, isExcluded, onToggle) {
  const chip = document.createElement("span");
  chip.className = "filter-chip";
  if (isIncluded) {
    chip.classList.add("included");
  }
  if (isExcluded) {
    chip.classList.add("excluded");
  }

  const displayName = name.split(".").pop() || name;
  let prefix = "";
  if (isIncluded) {
    prefix = "+ ";
  }
  if (isExcluded) {
    prefix = "− ";
  }
  chip.innerHTML = `${prefix}${escapeHtml(displayName)} <span class="chip-count">${count}</span>`;
  chip.title = `${name}\nClick: show only | Shift+click: hide`;

  chip.addEventListener("click", (e) => {
    e.preventDefault();
    if (e.shiftKey) {
      // Shift+click to exclude
      onToggle(name, false, !isExcluded);
    } else {
      // Regular click to include
      onToggle(name, !isIncluded, false);
    }
  });

  return chip;
}

/**
 * Creates an active filter pill showing current filter state.
 * @param {string} type - The filter type label (e.g., "source", "logger")
 * @param {string} name - The filter value name
 * @param {boolean} isExclude - Whether this is an exclude filter
 * @param {function(): void} onRemove - Callback when remove button is clicked
 * @returns {HTMLElement}
 */
export function createActiveFilter(type, name, isExclude, onRemove) {
  const filter = document.createElement("span");
  filter.className = `active-filter${isExclude ? " exclude" : ""}`;

  const displayName = name.split(".").pop() || name;
  const prefix = isExclude ? "−" : "+";
  filter.innerHTML = `${prefix}${escapeHtml(displayName)} <span class="remove-filter">×</span>`;
  filter.title = `${type}: ${name}`;

  filter.querySelector(".remove-filter").addEventListener("click", onRemove);
  return filter;
}
