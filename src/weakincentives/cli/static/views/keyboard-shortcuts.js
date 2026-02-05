// ============================================================================
// Keyboard Shortcuts - Global key handler and shortcuts overlay
// ============================================================================

/**
 * Initializes keyboard shortcuts and the help overlay.
 *
 * @param {object} deps - Dependencies
 * @param {object} deps.store - The application store
 * @param {object} deps.sessionsView - Sessions view module
 * @param {object} deps.transcriptView - Transcript view module
 * @param {object} deps.logsView - Logs view module
 * @param {object} deps.zoomModal - Zoom modal module
 * @param {function} deps.switchView - View switching function
 * @param {function} deps.reloadBundle - Bundle reload function
 * @param {function} deps.toggleTheme - Theme toggle function
 * @param {function} deps.toggleSidebar - Sidebar toggle function
 * @param {function} deps.navigateBundle - Bundle navigation function
 */
export function initKeyboardShortcuts({
  store,
  sessionsView,
  transcriptView,
  logsView,
  zoomModal,
  switchView,
  reloadBundle,
  toggleTheme,
  toggleSidebar,
  navigateBundle,
}) {
  const state = store.getState();

  const els = {
    shortcutsOverlay: document.getElementById("shortcuts-overlay"),
    shortcutsClose: document.getElementById("shortcuts-close"),
    helpButton: document.getElementById("help-button"),
    transcriptSearch: document.getElementById("transcript-search"),
    logsSearch: document.getElementById("logs-search"),
    filesystemFilter: document.getElementById("filesystem-filter"),
  };

  // -- Shortcuts overlay --

  function openShortcuts() {
    state.shortcutsOpen = true;
    els.shortcutsOverlay.classList.remove("hidden");
  }

  function closeShortcuts() {
    state.shortcutsOpen = false;
    els.shortcutsOverlay.classList.add("hidden");
  }

  els.shortcutsClose.addEventListener("click", closeShortcuts);
  els.shortcutsOverlay
    .querySelector(".shortcuts-backdrop")
    .addEventListener("click", closeShortcuts);
  els.helpButton.addEventListener("click", openShortcuts);

  // -- Focus search --

  function focusCurrentSearch() {
    if (state.activeView === "sessions") {
      sessionsView.focusSearch();
    } else if (state.activeView === "transcript") {
      els.transcriptSearch.focus();
    } else if (state.activeView === "logs") {
      els.logsSearch.focus();
    } else if (state.activeView === "filesystem") {
      els.filesystemFilter.focus();
    }
  }

  // -- Navigation helpers --

  function navigateNext() {
    if (state.activeView === "sessions") {
      sessionsView.navigateNext();
    } else if (state.activeView === "transcript") {
      transcriptView.scrollBy(1);
    } else if (state.activeView === "logs") {
      logsView.scrollBy(1);
    }
  }

  function navigatePrev() {
    if (state.activeView === "sessions") {
      sessionsView.navigatePrev();
    } else if (state.activeView === "transcript") {
      transcriptView.scrollBy(-1);
    } else if (state.activeView === "logs") {
      logsView.scrollBy(-1);
    }
  }

  // -- Key handler helpers --

  function handleEscapeKey(e) {
    if (zoomModal.isOpen()) {
      e.preventDefault();
      zoomModal.closeZoomModal();
      return true;
    }
    if (state.shortcutsOpen) {
      e.preventDefault();
      closeShortcuts();
      return true;
    }
    return false;
  }

  async function handleZoomModalKeys(e) {
    const nextKeys = ["j", "J", "ArrowDown", "ArrowRight"];
    const prevKeys = ["k", "K", "ArrowUp", "ArrowLeft"];
    if (nextKeys.includes(e.key)) {
      e.preventDefault();
      await zoomModal.zoomNext();
      return true;
    }
    if (prevKeys.includes(e.key)) {
      e.preventDefault();
      zoomModal.zoomPrev();
      return true;
    }
    return true;
  }

  function isInputElement(target) {
    const tag = target.tagName;
    return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
  }

  const TAB_VIEWS = ["sessions", "transcript", "logs", "filesystem", "environment"];

  const KEYBOARD_SHORTCUTS = {
    r: reloadBundle,
    R: reloadBundle,
    d: toggleTheme,
    D: toggleTheme,
    "[": toggleSidebar,
    "/": focusCurrentSearch,
    j: navigateNext,
    J: navigateNext,
    k: navigatePrev,
    K: navigatePrev,
  };

  function handleTabShortcut(e, key) {
    if (key >= "1" && key <= "5") {
      e.preventDefault();
      switchView(TAB_VIEWS[Number.parseInt(key, 10) - 1]);
      return true;
    }
    return false;
  }

  function handleArrowShortcut(e, key) {
    if (key === "ArrowLeft" || key === "ArrowRight") {
      e.preventDefault();
      navigateBundle(key === "ArrowRight" ? 1 : -1);
      return true;
    }
    return false;
  }

  function handleGlobalShortcuts(e) {
    const key = e.key;

    if (handleTabShortcut(e, key)) {
      return true;
    }

    if (KEYBOARD_SHORTCUTS[key]) {
      e.preventDefault();
      KEYBOARD_SHORTCUTS[key]();
      return true;
    }

    if (key === "?" || (e.shiftKey && key === "/")) {
      e.preventDefault();
      openShortcuts();
      return true;
    }

    return handleArrowShortcut(e, key);
  }

  // -- Global keydown handler --

  async function handleZoomKeydown(e) {
    if (zoomModal.isNextPending()) {
      e.preventDefault();
      return;
    }
    await handleZoomModalKeys(e);
  }

  async function handleKeydown(e) {
    if (e.key === "Escape") {
      handleEscapeKey(e);
      return;
    }

    if (zoomModal.isOpen()) {
      await handleZoomKeydown(e);
      return;
    }

    if (state.shortcutsOpen || isInputElement(e.target)) {
      return;
    }

    handleGlobalShortcuts(e);
  }

  document.addEventListener("keydown", handleKeydown);
}
