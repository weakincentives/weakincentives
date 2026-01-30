// @ts-check
import { describe, expect, test } from "bun:test";

import {
  formatBytes,
  escapeHtml,
  isObject,
  isPrimitive,
  isSimpleArray,
  valueType,
  getMarkdownPayload,
  pathKey,
  buildFilterParams,
  toggleFilter,
  toggleFilterExplicit,
  filterItemsBySearch,
  calculateVisibleRange,
  getOffsetForIndex,
  getTotalHeight,
} from "../../src/weakincentives/cli/static/lib.js";

// ============================================================================
// formatBytes
// ============================================================================

describe("formatBytes", () => {
  test("returns '0 B' for zero bytes", () => {
    expect(formatBytes(0)).toBe("0 B");
  });

  test("formats bytes correctly", () => {
    expect(formatBytes(500)).toBe("500 B");
    expect(formatBytes(1024)).toBe("1 KB");
    expect(formatBytes(1536)).toBe("1.5 KB");
    expect(formatBytes(1048576)).toBe("1 MB");
    expect(formatBytes(1073741824)).toBe("1 GB");
    expect(formatBytes(1099511627776)).toBe("1 TB");
  });

  test("rounds to 2 decimal places", () => {
    expect(formatBytes(1234)).toBe("1.21 KB");
    expect(formatBytes(1234567)).toBe("1.18 MB");
  });
});

// ============================================================================
// escapeHtml
// ============================================================================

describe("escapeHtml", () => {
  test("escapes ampersands", () => {
    expect(escapeHtml("foo & bar")).toBe("foo &amp; bar");
  });

  test("escapes angle brackets", () => {
    expect(escapeHtml("<script>")).toBe("&lt;script&gt;");
  });

  test("escapes quotes", () => {
    expect(escapeHtml('say "hello"')).toBe("say &quot;hello&quot;");
    expect(escapeHtml("it's")).toBe("it&#39;s");
  });

  test("handles multiple special characters", () => {
    expect(escapeHtml('<a href="test">x & y</a>')).toBe(
      "&lt;a href=&quot;test&quot;&gt;x &amp; y&lt;/a&gt;"
    );
  });

  test("handles empty string", () => {
    expect(escapeHtml("")).toBe("");
  });

  test("passes through plain text", () => {
    expect(escapeHtml("hello world")).toBe("hello world");
  });

  test("coerces non-string values", () => {
    expect(escapeHtml(123)).toBe("123");
    expect(escapeHtml(null)).toBe("null");
  });
});

// ============================================================================
// isObject
// ============================================================================

describe("isObject", () => {
  test("returns true for plain objects", () => {
    expect(isObject({})).toBe(true);
    expect(isObject({ a: 1 })).toBe(true);
  });

  test("returns true for arrays", () => {
    expect(isObject([])).toBe(true);
    expect(isObject([1, 2, 3])).toBe(true);
  });

  test("returns false for null", () => {
    expect(isObject(null)).toBe(false);
  });

  test("returns false for primitives", () => {
    expect(isObject(undefined)).toBe(false);
    expect(isObject(42)).toBe(false);
    expect(isObject("string")).toBe(false);
    expect(isObject(true)).toBe(false);
  });
});

// ============================================================================
// isPrimitive
// ============================================================================

describe("isPrimitive", () => {
  test("returns true for null", () => {
    expect(isPrimitive(null)).toBe(true);
  });

  test("returns true for strings", () => {
    expect(isPrimitive("hello")).toBe(true);
    expect(isPrimitive("")).toBe(true);
  });

  test("returns true for numbers", () => {
    expect(isPrimitive(42)).toBe(true);
    expect(isPrimitive(0)).toBe(true);
    expect(isPrimitive(3.14)).toBe(true);
  });

  test("returns true for booleans", () => {
    expect(isPrimitive(true)).toBe(true);
    expect(isPrimitive(false)).toBe(true);
  });

  test("returns true for undefined", () => {
    expect(isPrimitive(undefined)).toBe(true);
  });

  test("returns false for objects", () => {
    expect(isPrimitive({})).toBe(false);
    expect(isPrimitive({ a: 1 })).toBe(false);
  });

  test("returns false for arrays", () => {
    expect(isPrimitive([])).toBe(false);
    expect(isPrimitive([1, 2])).toBe(false);
  });
});

// ============================================================================
// isSimpleArray
// ============================================================================

describe("isSimpleArray", () => {
  test("returns true for empty array", () => {
    expect(isSimpleArray([])).toBe(true);
  });

  test("returns true for array of primitives", () => {
    expect(isSimpleArray([1, 2, 3])).toBe(true);
    expect(isSimpleArray(["a", "b", "c"])).toBe(true);
    expect(isSimpleArray([1, "two", true, null])).toBe(true);
  });

  test("returns false for array with objects", () => {
    expect(isSimpleArray([1, { a: 1 }])).toBe(false);
    expect(isSimpleArray([{}])).toBe(false);
  });

  test("returns false for array with nested arrays", () => {
    expect(isSimpleArray([[1, 2]])).toBe(false);
    expect(isSimpleArray([1, [2]])).toBe(false);
  });

  test("returns false for non-arrays", () => {
    expect(isSimpleArray("not an array")).toBe(false);
    expect(isSimpleArray(123)).toBe(false);
    expect(isSimpleArray({})).toBe(false);
    expect(isSimpleArray(null)).toBe(false);
  });
});

// ============================================================================
// valueType
// ============================================================================

describe("valueType", () => {
  test("returns 'array' for arrays", () => {
    expect(valueType([])).toBe("array");
    expect(valueType([1, 2, 3])).toBe("array");
  });

  test("returns 'null' for null", () => {
    expect(valueType(null)).toBe("null");
  });

  test("returns typeof for primitives", () => {
    expect(valueType("string")).toBe("string");
    expect(valueType(42)).toBe("number");
    expect(valueType(true)).toBe("boolean");
    expect(valueType(undefined)).toBe("undefined");
  });

  test("returns 'object' for plain objects", () => {
    expect(valueType({})).toBe("object");
    expect(valueType({ a: 1 })).toBe("object");
  });

  test("returns 'markdown' for markdown payload objects", () => {
    const markdown = {
      __markdown__: {
        text: "# Hello",
        html: "<h1>Hello</h1>",
      },
    };
    expect(valueType(markdown)).toBe("markdown");
  });

  test("uses custom markdown key", () => {
    const custom = {
      _md_: {
        text: "# Hello",
        html: "<h1>Hello</h1>",
      },
    };
    expect(valueType(custom, "_md_")).toBe("markdown");
    expect(valueType(custom, "__markdown__")).toBe("object");
  });
});

// ============================================================================
// getMarkdownPayload
// ============================================================================

describe("getMarkdownPayload", () => {
  test("returns null for non-objects", () => {
    expect(getMarkdownPayload(null)).toBe(null);
    expect(getMarkdownPayload("string")).toBe(null);
    expect(getMarkdownPayload(42)).toBe(null);
    expect(getMarkdownPayload(undefined)).toBe(null);
  });

  test("returns null for arrays", () => {
    expect(getMarkdownPayload([])).toBe(null);
    expect(getMarkdownPayload([1, 2])).toBe(null);
  });

  test("returns null for objects without markdown key", () => {
    expect(getMarkdownPayload({})).toBe(null);
    expect(getMarkdownPayload({ a: 1 })).toBe(null);
  });

  test("returns null for invalid markdown payloads", () => {
    expect(getMarkdownPayload({ __markdown__: null })).toBe(null);
    expect(getMarkdownPayload({ __markdown__: "string" })).toBe(null);
    expect(getMarkdownPayload({ __markdown__: { text: "only text" } })).toBe(null);
    expect(getMarkdownPayload({ __markdown__: { html: "only html" } })).toBe(null);
  });

  test("returns payload for valid markdown objects", () => {
    const payload = { text: "# Hello", html: "<h1>Hello</h1>" };
    const obj = { __markdown__: payload };
    expect(getMarkdownPayload(obj)).toEqual(payload);
  });

  test("uses custom markdown key", () => {
    const payload = { text: "test", html: "<p>test</p>" };
    const obj = { _custom_md_: payload };
    expect(getMarkdownPayload(obj, "_custom_md_")).toEqual(payload);
    expect(getMarkdownPayload(obj, "__markdown__")).toBe(null);
  });
});

// ============================================================================
// pathKey
// ============================================================================

describe("pathKey", () => {
  test("joins path segments with dots", () => {
    expect(pathKey(["a", "b", "c"])).toBe("a.b.c");
  });

  test("handles single segment", () => {
    expect(pathKey(["root"])).toBe("root");
  });

  test("handles empty array", () => {
    expect(pathKey([])).toBe("");
  });

  test("handles numeric segments", () => {
    expect(pathKey(["items", "0", "name"])).toBe("items.0.name");
  });
});

// ============================================================================
// buildFilterParams
// ============================================================================

describe("buildFilterParams", () => {
  test("returns empty string for empty filters", () => {
    expect(buildFilterParams({ include: [], exclude: [] }, "source")).toBe("");
  });

  test("builds include param", () => {
    expect(buildFilterParams({ include: ["user", "assistant"], exclude: [] }, "source")).toBe(
      "source=user,assistant"
    );
  });

  test("builds exclude param", () => {
    expect(buildFilterParams({ include: [], exclude: ["system"] }, "source")).toBe(
      "source_exclude=system"
    );
  });

  test("builds both include and exclude params", () => {
    expect(
      buildFilterParams({ include: ["user"], exclude: ["system"] }, "source")
    ).toBe("source=user&source_exclude=system");
  });

  test("URL-encodes special characters", () => {
    expect(buildFilterParams({ include: ["a&b", "c=d"], exclude: [] }, "type")).toBe(
      "type=a%26b,c%3Dd"
    );
  });
});

// ============================================================================
// toggleFilter
// ============================================================================

describe("toggleFilter", () => {
  test("adds to include when not present in either list", () => {
    const result = toggleFilter("user", [], []);
    expect(result.include).toEqual(["user"]);
    expect(result.exclude).toEqual([]);
  });

  test("removes from include when already included", () => {
    const result = toggleFilter("user", ["user", "assistant"], []);
    expect(result.include).toEqual(["assistant"]);
    expect(result.exclude).toEqual([]);
  });

  test("removes from exclude when already excluded", () => {
    const result = toggleFilter("system", [], ["system", "tool"]);
    expect(result.include).toEqual([]);
    expect(result.exclude).toEqual(["tool"]);
  });

  test("does not modify original arrays", () => {
    const include = ["user"];
    const exclude = ["system"];
    toggleFilter("assistant", include, exclude);
    expect(include).toEqual(["user"]);
    expect(exclude).toEqual(["system"]);
  });
});

// ============================================================================
// toggleFilterExplicit
// ============================================================================

describe("toggleFilterExplicit", () => {
  test("adds to include list when toInclude is true", () => {
    const result = toggleFilterExplicit("user", [], [], true, false);
    expect(result.include).toEqual(["user"]);
    expect(result.exclude).toEqual([]);
  });

  test("adds to exclude list when toExclude is true", () => {
    const result = toggleFilterExplicit("system", [], [], false, true);
    expect(result.include).toEqual([]);
    expect(result.exclude).toEqual(["system"]);
  });

  test("removes from include when already included and toInclude is true", () => {
    const result = toggleFilterExplicit("user", ["user"], [], true, false);
    expect(result.include).toEqual([]);
    expect(result.exclude).toEqual([]);
  });

  test("removes from exclude when already excluded and toExclude is true", () => {
    const result = toggleFilterExplicit("system", [], ["system"], false, true);
    expect(result.include).toEqual([]);
    expect(result.exclude).toEqual([]);
  });

  test("moves from include to exclude", () => {
    const result = toggleFilterExplicit("user", ["user"], [], false, true);
    expect(result.include).toEqual([]);
    expect(result.exclude).toEqual(["user"]);
  });

  test("moves from exclude to include", () => {
    const result = toggleFilterExplicit("system", [], ["system"], true, false);
    expect(result.include).toEqual(["system"]);
    expect(result.exclude).toEqual([]);
  });

  test("removes from both when neither flag is set", () => {
    const result = toggleFilterExplicit("user", ["user"], [], false, false);
    expect(result.include).toEqual([]);
    expect(result.exclude).toEqual([]);
  });
});

// ============================================================================
// filterItemsBySearch
// ============================================================================

describe("filterItemsBySearch", () => {
  test("returns all items with original indices for empty query", () => {
    const items = [{ name: "a" }, { name: "b" }];
    const result = filterItemsBySearch(items, "");
    expect(result).toEqual([
      { item: { name: "a" }, index: 0 },
      { item: { name: "b" }, index: 1 },
    ]);
  });

  test("returns all items for whitespace-only query", () => {
    const items = [{ name: "a" }];
    const result = filterItemsBySearch(items, "   ");
    expect(result).toEqual([{ item: { name: "a" }, index: 0 }]);
  });

  test("filters items by JSON string match", () => {
    const items = [{ name: "alice" }, { name: "bob" }, { name: "charlie" }];
    const result = filterItemsBySearch(items, "ali");
    expect(result).toEqual([{ item: { name: "alice" }, index: 0 }]);
  });

  test("search is case-insensitive", () => {
    const items = [{ name: "Alice" }, { name: "BOB" }];
    const result = filterItemsBySearch(items, "alice");
    expect(result).toEqual([{ item: { name: "Alice" }, index: 0 }]);
  });

  test("preserves original indices", () => {
    const items = [{ name: "a" }, { name: "match" }, { name: "b" }, { name: "match2" }];
    const result = filterItemsBySearch(items, "match");
    expect(result).toEqual([
      { item: { name: "match" }, index: 1 },
      { item: { name: "match2" }, index: 3 },
    ]);
  });

  test("searches nested properties", () => {
    const items = [{ user: { name: "test" } }, { user: { name: "other" } }];
    const result = filterItemsBySearch(items, "test");
    expect(result).toEqual([{ item: { user: { name: "test" } }, index: 0 }]);
  });
});

// ============================================================================
// calculateVisibleRange
// ============================================================================

describe("calculateVisibleRange", () => {
  const fixedHeight = () => 100;

  test("calculates range for scrolled to top", () => {
    const result = calculateVisibleRange({
      scrollTop: 0,
      viewportHeight: 500,
      itemCount: 100,
      bufferSize: 2,
      getItemHeight: fixedHeight,
    });
    // Viewport shows items 0-5 (6 items fit in 500px at 100px each, with partial)
    // Plus buffer of 2 on bottom = 8 total
    expect(result.startIndex).toBe(0);
    expect(result.endIndex).toBe(8);
  });

  test("calculates range for scrolled position", () => {
    const result = calculateVisibleRange({
      scrollTop: 1000,
      viewportHeight: 500,
      itemCount: 100,
      bufferSize: 2,
      getItemHeight: fixedHeight,
    });
    // At scrollTop 1000, first visible is item 10
    // endIndex: items 10-15 are in viewport (6 items), plus 2 buffer = 18
    expect(result.startIndex).toBe(8); // 10 - 2 buffer
    expect(result.endIndex).toBe(18);
  });

  test("clamps startIndex to 0", () => {
    const result = calculateVisibleRange({
      scrollTop: 50,
      viewportHeight: 500,
      itemCount: 100,
      bufferSize: 10,
      getItemHeight: fixedHeight,
    });
    expect(result.startIndex).toBe(0);
  });

  test("clamps endIndex to itemCount", () => {
    const result = calculateVisibleRange({
      scrollTop: 9500,
      viewportHeight: 500,
      itemCount: 100,
      bufferSize: 10,
      getItemHeight: fixedHeight,
    });
    expect(result.endIndex).toBe(100);
  });

  test("handles variable height items", () => {
    const heights = [50, 100, 150, 200, 50, 100];
    const getHeight = (i) => heights[i] || 100;
    const result = calculateVisibleRange({
      scrollTop: 100,
      viewportHeight: 300,
      itemCount: 6,
      bufferSize: 1,
      getItemHeight: getHeight,
    });
    // At scroll 100: item 0 is 50px, item 1 starts at 50, ends at 150
    // First visible is item 1 (starts at 50, visible at scroll 100)
    expect(result.startIndex).toBe(0); // 1 - 1 buffer, clamped to 0
  });

  test("handles empty item list", () => {
    const result = calculateVisibleRange({
      scrollTop: 0,
      viewportHeight: 500,
      itemCount: 0,
      bufferSize: 2,
      getItemHeight: fixedHeight,
    });
    expect(result.startIndex).toBe(0);
    expect(result.endIndex).toBe(0);
  });
});

// ============================================================================
// getOffsetForIndex
// ============================================================================

describe("getOffsetForIndex", () => {
  test("returns 0 for index 0", () => {
    expect(getOffsetForIndex(0, 10, () => 100)).toBe(0);
  });

  test("sums heights for items before index", () => {
    expect(getOffsetForIndex(3, 10, () => 100)).toBe(300);
  });

  test("handles variable heights", () => {
    const heights = [50, 100, 150, 200];
    expect(getOffsetForIndex(3, 4, (i) => heights[i])).toBe(300); // 50 + 100 + 150
  });

  test("clamps to itemCount", () => {
    expect(getOffsetForIndex(100, 5, () => 100)).toBe(500);
  });
});

// ============================================================================
// getTotalHeight
// ============================================================================

describe("getTotalHeight", () => {
  test("returns 0 for empty list", () => {
    expect(getTotalHeight(0, () => 100)).toBe(0);
  });

  test("sums all item heights", () => {
    expect(getTotalHeight(5, () => 100)).toBe(500);
  });

  test("handles variable heights", () => {
    const heights = [50, 100, 150];
    expect(getTotalHeight(3, (i) => heights[i])).toBe(300);
  });
});
