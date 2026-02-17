// ============================================================================
// Keyboard Shortcuts - Global key handler and shortcuts overlay
// ============================================================================

/**
 * Initializes keyboard shortcuts and the help overlay.
 *
 * @param {object} deps - Dependencies
 * @param {object} deps.state - The shared application state
 * @param {object} deps.sessionsView - Sessions view module
 * @param {object} deps.transcriptView - Transcript view module
 * @param {object} deps.logsView - Logs view module
 * @param {function} deps.switchView - View switching function
 * @param {function} deps.reloadBundle - Bundle reload function
 * @param {function} deps.toggleTheme - Theme toggle function
 * @param {function} deps.toggleSidebar - Sidebar toggle function
 * @param {function} deps.navigateBundle - Bundle navigation function
 * @param {function} deps.openCommandPalette - Command palette open function
 * @param {function} deps.closeCommandPalette - Command palette close function
 */
export function initKeyboardShortcuts({
  state,
  sessionsView,
  transcriptView,
  logsView,
  switchView,
  reloadBundle,
  toggleTheme,
  toggleSidebar,
  navigateBundle,
  openCommandPalette,
  closeCommandPalette,
}) {
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

  // Jump to start/end

  function scrollListToEnd(id) {
    const list = document.getElementById(id);
    if (list) {
      list.scrollTop = list.scrollHeight;
    }
  }

  function scrollListToStart(id) {
    const list = document.getElementById(id);
    if (list) {
      list.scrollTop = 0;
    }
  }

  const JUMP_END_HANDLERS = {
    sessions: () => {
      const items = document.querySelectorAll("#json-viewer .item-card");
      if (items.length > 0) {
        sessionsView.focusItem(items.length - 1);
      }
    },
    transcript: () => scrollListToEnd("transcript-list"),
    logs: () => scrollListToEnd("logs-list"),
  };

  const JUMP_START_HANDLERS = {
    sessions: () => sessionsView.focusItem(0),
    transcript: () => scrollListToStart("transcript-list"),
    logs: () => scrollListToStart("logs-list"),
  };

  function jumpToEnd() {
    const handler = JUMP_END_HANDLERS[state.activeView];
    if (handler) {
      handler();
    }
  }

  function jumpToStart() {
    const handler = JUMP_START_HANDLERS[state.activeView];
    if (handler) {
      handler();
    }
  }

  // -- gg (double-g) detection --
  let lastGTime = 0;

  // -- Key handler helpers --

  function handleEscapeKey(e) {
    if (state.commandPaletteOpen) {
      e.preventDefault();
      closeCommandPalette();
      return true;
    }
    if (state.shortcutsOpen) {
      e.preventDefault();
      closeShortcuts();
      return true;
    }
    return false;
  }

  function isInputElement(target) {
    const tag = target.tagName;
    return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
  }

  const ALL_TAB_VIEWS = ["sessions", "transcript", "logs", "filesystem", "environment"];

  function getVisibleTabs() {
    return ALL_TAB_VIEWS.filter((view) => {
      if (view === "transcript") {
        return state.hasTranscript;
      }
      return true;
    });
  }

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
    const visibleTabs = getVisibleTabs();
    const idx = Number.parseInt(key, 10) - 1;
    if (key >= "1" && idx < visibleTabs.length) {
      e.preventDefault();
      switchView(visibleTabs[idx]);
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

  function toggleCommandPalette(e) {
    e.preventDefault();
    if (state.commandPaletteOpen) {
      closeCommandPalette();
    } else {
      openCommandPalette();
    }
  }

  function isCommandPaletteKey(e) {
    return (e.metaKey || e.ctrlKey) && e.key === "k";
  }

  function handleDoubleG(e) {
    const now = Date.now();
    if (now - lastGTime < 400) {
      e.preventDefault();
      jumpToStart();
      lastGTime = 0;
      return true;
    }
    lastGTime = now;
    return false;
  }

  function handleJumpEnd(e) {
    e.preventDefault();
    jumpToEnd();
    return true;
  }

  function handleJumpShortcut(e, key) {
    if (key === "G") {
      return handleJumpEnd(e);
    }
    if (key === "g" && !e.shiftKey) {
      return handleDoubleG(e);
    }
    return false;
  }

  function handleHelpShortcut(e, key) {
    if (key === "?" || (e.shiftKey && key === "/")) {
      e.preventDefault();
      openShortcuts();
      return true;
    }
    return false;
  }

  function handleMappedShortcut(e, key) {
    if (KEYBOARD_SHORTCUTS[key]) {
      e.preventDefault();
      KEYBOARD_SHORTCUTS[key]();
      return true;
    }
    return false;
  }

  const SHORTCUT_CHAIN = [
    (e, key) => handleTabShortcut(e, key),
    (e, key) => handleJumpShortcut(e, key),
    (e, key) => handleMappedShortcut(e, key),
    (e, key) => handleHelpShortcut(e, key),
    (e, key) => handleArrowShortcut(e, key),
  ];

  function handleGlobalShortcuts(e) {
    if (isCommandPaletteKey(e)) {
      toggleCommandPalette(e);
      return true;
    }
    const key = e.key;
    return SHORTCUT_CHAIN.some((handler) => handler(e, key));
  }

  // -- Global keydown handler --

  function handleKeydown(e) {
    if (isCommandPaletteKey(e)) {
      toggleCommandPalette(e);
      return;
    }
    if (e.key === "Escape") {
      handleEscapeKey(e);
      return;
    }
    if (state.shortcutsOpen || state.commandPaletteOpen || isInputElement(e.target)) {
      return;
    }
    handleGlobalShortcuts(e);
  }

  document.addEventListener("keydown", handleKeydown);
}
