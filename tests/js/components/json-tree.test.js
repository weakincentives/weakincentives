// @ts-check
import { describe, expect, test } from "bun:test";

import {
  applyDepth,
  createMarkdownViewState,
  createPathState,
  setOpenForAll,
} from "../../../src/weakincentives/cli/static/components/json-tree.js";

// ============================================================================
// createPathState
// ============================================================================

describe("createPathState", () => {
  test("shouldOpen returns true for paths within expand depth", () => {
    const pathState = createPathState();
    // Default expand depth is 2
    expect(pathState.shouldOpen(["a"], 0)).toBe(true);
    expect(pathState.shouldOpen(["a", "b"], 1)).toBe(true);
  });

  test("shouldOpen returns false for paths beyond expand depth", () => {
    const pathState = createPathState();
    expect(pathState.shouldOpen(["a", "b", "c"], 2)).toBe(false);
    expect(pathState.shouldOpen(["a", "b", "c", "d"], 3)).toBe(false);
  });

  test("setOpen opens a path", () => {
    const pathState = createPathState();
    pathState.setOpen(["deep", "nested", "path"], true);
    // Should be open regardless of depth because we explicitly set it
    expect(pathState.shouldOpen(["deep", "nested", "path"], 10)).toBe(true);
  });

  test("setOpen closes a path", () => {
    const pathState = createPathState();
    pathState.setOpen(["a"], false);
    // Should be closed even though depth 0 would normally be open
    expect(pathState.shouldOpen(["a"], 0)).toBe(false);
  });

  test("setExpandDepth changes default expansion depth", () => {
    const pathState = createPathState();
    pathState.setExpandDepth(5);
    expect(pathState.getExpandDepth()).toBe(5);
    expect(pathState.shouldOpen(["a", "b", "c", "d"], 4)).toBe(true);
  });

  test("reset clears open and closed paths", () => {
    const pathState = createPathState();
    pathState.setOpen(["a"], true);
    pathState.setOpen(["b"], false);
    pathState.reset();
    // After reset, shouldOpen uses default depth behavior
    expect(pathState.shouldOpen(["a"], 0)).toBe(true);
    expect(pathState.shouldOpen(["b"], 0)).toBe(true);
  });
});

// ============================================================================
// createMarkdownViewState
// ============================================================================

describe("createMarkdownViewState", () => {
  test("getMarkdownView returns 'html' by default", () => {
    const viewState = createMarkdownViewState();
    expect(viewState.getMarkdownView(["some", "path"])).toBe("html");
  });

  test("setMarkdownView sets view mode", () => {
    const viewState = createMarkdownViewState();
    viewState.setMarkdownView(["path"], "raw");
    expect(viewState.getMarkdownView(["path"])).toBe("raw");
  });

  test("setMarkdownView can switch back to html", () => {
    const viewState = createMarkdownViewState();
    viewState.setMarkdownView(["path"], "raw");
    viewState.setMarkdownView(["path"], "html");
    expect(viewState.getMarkdownView(["path"])).toBe("html");
  });

  test("different paths have independent view states", () => {
    const viewState = createMarkdownViewState();
    viewState.setMarkdownView(["path1"], "raw");
    viewState.setMarkdownView(["path2"], "html");
    expect(viewState.getMarkdownView(["path1"])).toBe("raw");
    expect(viewState.getMarkdownView(["path2"])).toBe("html");
  });

  test("reset clears all view states", () => {
    const viewState = createMarkdownViewState();
    viewState.setMarkdownView(["path1"], "raw");
    viewState.setMarkdownView(["path2"], "raw");
    viewState.reset();
    expect(viewState.getMarkdownView(["path1"])).toBe("html");
    expect(viewState.getMarkdownView(["path2"])).toBe("html");
  });
});

// ============================================================================
// applyDepth
// ============================================================================

describe("applyDepth", () => {
  test("opens paths up to specified depth", () => {
    const calls = [];
    const setOpen = (path, open) => calls.push({ path: path.join("."), open });

    const items = [{ a: { b: { c: "value" } } }];
    applyDepth(items, 2, setOpen);

    // Should open item-0, item-0.a (depth 0 and 1)
    expect(calls).toContainEqual({ path: "item-0", open: true });
    expect(calls).toContainEqual({ path: "item-0.a", open: true });
  });

  test("does not open paths beyond specified depth", () => {
    const calls = [];
    const setOpen = (path, open) => calls.push({ path: path.join("."), open });

    const items = [{ a: { b: { c: "value" } } }];
    applyDepth(items, 1, setOpen);

    // Should only open item-0 (depth 0)
    const openPaths = calls.filter((c) => c.open).map((c) => c.path);
    expect(openPaths).toContain("item-0");
    expect(openPaths).not.toContain("item-0.a");
  });

  test("handles arrays in items", () => {
    const calls = [];
    const setOpen = (path, open) => calls.push({ path: path.join("."), open });

    const items = [[1, 2, 3]];
    applyDepth(items, 2, setOpen);

    expect(calls).toContainEqual({ path: "item-0", open: true });
  });

  test("handles multiple items", () => {
    const calls = [];
    const setOpen = (path, open) => calls.push({ path: path.join("."), open });

    const items = [{ a: 1 }, { b: 2 }];
    applyDepth(items, 1, setOpen);

    expect(calls).toContainEqual({ path: "item-0", open: true });
    expect(calls).toContainEqual({ path: "item-1", open: true });
  });

  test("handles empty items array", () => {
    const calls = [];
    const setOpen = (path, open) => calls.push({ path: path.join("."), open });

    applyDepth([], 2, setOpen);
    expect(calls).toEqual([]);
  });

  test("handles primitive items (skips them)", () => {
    const calls = [];
    const setOpen = (path, open) => calls.push({ path: path.join("."), open });

    const items = ["string", 123, null];
    applyDepth(items, 2, setOpen);
    expect(calls).toEqual([]);
  });
});

// ============================================================================
// setOpenForAll
// ============================================================================

describe("setOpenForAll", () => {
  test("opens all paths when open is true", () => {
    const calls = [];
    const setOpen = (path, open) => calls.push({ path: path.join("."), open });

    const items = [{ a: { b: 1 } }];
    setOpenForAll(items, true, setOpen);

    expect(calls.every((c) => c.open)).toBe(true);
    expect(calls.length).toBeGreaterThan(0);
  });

  test("closes all paths when open is false", () => {
    const calls = [];
    const setOpen = (path, open) => calls.push({ path: path.join("."), open });

    const items = [{ a: { b: 1 } }];
    setOpenForAll(items, false, setOpen);

    expect(calls.every((c) => !c.open)).toBe(true);
  });

  test("traverses nested objects", () => {
    const paths = [];
    const setOpen = (path) => paths.push(path.join("."));

    const items = [{ a: { b: { c: 1 } } }];
    setOpenForAll(items, true, setOpen);

    expect(paths).toContain("item-0");
    expect(paths).toContain("item-0.a");
    expect(paths).toContain("item-0.a.b");
  });

  test("traverses arrays", () => {
    const paths = [];
    const setOpen = (path) => paths.push(path.join("."));

    const items = [[[1]]];
    setOpenForAll(items, true, setOpen);

    expect(paths).toContain("item-0");
    expect(paths).toContain("item-0.0");
  });

  test("handles empty items", () => {
    const calls = [];
    const setOpen = (path, open) => calls.push({ path: path.join("."), open });

    setOpenForAll([], true, setOpen);
    expect(calls).toEqual([]);
  });
});
