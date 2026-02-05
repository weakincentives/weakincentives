// ============================================================================
// VirtualScroller - Windowed rendering with garbage collection
// ============================================================================

/**
 * VirtualScroller - Implements windowed rendering with garbage collection
 * Only renders items visible in viewport plus a buffer zone.
 * Items scrolling out of the buffer are removed from DOM to save memory.
 */
export class VirtualScroller {
  constructor(options) {
    this.container = options.container;
    this.estimatedItemHeight = options.estimatedItemHeight || 100;
    this.bufferSize = options.bufferSize || 10; // Items to keep above/below viewport
    this.renderItem = options.renderItem;
    this.onLoadMore = options.onLoadMore || null;
    this.onLoadError = options.onLoadError || null; // Callback for load errors
    this.loadMoreThreshold = options.loadMoreThreshold || 3; // Items from bottom to trigger load

    this.items = [];
    this.totalCount = 0;
    this.hasMore = false;
    this.isLoading = false;

    // Track rendered items
    this.renderedItems = new Map(); // index -> DOM element
    this.itemHeights = new Map(); // index -> measured height

    // Spacer elements
    this.topSpacer = document.createElement("div");
    this.topSpacer.className = "virtual-spacer virtual-spacer-top";
    this.bottomSpacer = document.createElement("div");
    this.bottomSpacer.className = "virtual-spacer virtual-spacer-bottom";
    this.loadMoreSentinel = document.createElement("div");
    this.loadMoreSentinel.className = "virtual-load-sentinel";

    // Intersection observer for infinite scroll
    this.loadMoreObserver = null;
    this.setupLoadMoreObserver();

    // Scroll handler with debounce
    this.scrollHandler = this.debounce(() => this.updateVisibleRange(), 16);
    this.container.addEventListener("scroll", this.scrollHandler);

    // Resize observer for container size changes
    this.resizeObserver = new ResizeObserver(() => this.updateVisibleRange());
    this.resizeObserver.observe(this.container);

    // Initial state
    this.visibleStart = 0;
    this.visibleEnd = 0;
    this.scrollTimeout = null; // Track debounce timeout for cleanup
  }

  debounce(fn, delay) {
    return (...args) => {
      clearTimeout(this.scrollTimeout);
      this.scrollTimeout = setTimeout(() => fn(...args), delay);
    };
  }

  setupLoadMoreObserver() {
    this.loadMoreObserver = new IntersectionObserver(
      (entries) => {
        // Check if any entry is intersecting (avoid race condition with multiple entries)
        const anyIntersecting = entries.some((e) => e.isIntersecting);
        if (anyIntersecting && this.hasMore && !this.isLoading && this.onLoadMore) {
          this.isLoading = true;
          this.onLoadMore()
            .catch((error) => {
              if (this.onLoadError) {
                this.onLoadError(error);
              } else {
                console.error("Failed to load more items:", error);
              }
            })
            .finally(() => {
              this.isLoading = false;
            });
        }
      },
      { root: this.container, rootMargin: "200px" }
    );
  }

  setData(items, totalCount, hasMore) {
    this.items = items;
    this.totalCount = totalCount;
    this.hasMore = hasMore;
    // Clear stale height measurements when data changes
    this.itemHeights.clear();
    this.render();
  }

  appendData(newItems, totalCount, hasMore) {
    const wasObserving = this.hasMore;
    this.items = this.items.concat(newItems);
    this.totalCount = totalCount;
    this.hasMore = hasMore;
    this.updateVisibleRange();
    this.updateSpacers();

    // If hasMore changed from false to true, re-observe the sentinel
    if (!wasObserving && this.hasMore) {
      this.loadMoreObserver.observe(this.loadMoreSentinel);
    }
  }

  reset() {
    // Unobserve sentinel before removing it from DOM to prevent memory leak
    this.loadMoreObserver.unobserve(this.loadMoreSentinel);

    this.items = [];
    this.totalCount = 0;
    this.hasMore = false;
    this.renderedItems.clear();
    this.itemHeights.clear();
    this.container.innerHTML = "";
    this.visibleStart = 0;
    this.visibleEnd = 0;
  }

  getItemHeight(index) {
    return this.itemHeights.get(index) || this.estimatedItemHeight;
  }

  getTotalHeight() {
    let total = 0;
    for (let i = 0; i < this.items.length; i++) {
      total += this.getItemHeight(i);
    }
    return total;
  }

  getOffsetForIndex(index) {
    let offset = 0;
    for (let i = 0; i < index && i < this.items.length; i++) {
      offset += this.getItemHeight(i);
    }
    return offset;
  }

  getIndexAtOffset(offset) {
    let current = 0;
    for (let i = 0; i < this.items.length; i++) {
      const height = this.getItemHeight(i);
      if (current + height > offset) {
        return i;
      }
      current += height;
    }
    return Math.max(0, this.items.length - 1);
  }

  calculateVisibleRange() {
    const scrollTop = this.container.scrollTop;
    const viewportHeight = this.container.clientHeight;

    const startIndex = Math.max(0, this.getIndexAtOffset(scrollTop) - this.bufferSize);
    const endIndex = Math.min(
      this.items.length,
      this.getIndexAtOffset(scrollTop + viewportHeight) + this.bufferSize + 1
    );

    return { startIndex, endIndex };
  }

  updateSpacers() {
    const topHeight = this.getOffsetForIndex(this.visibleStart);
    const bottomHeight = this.getTotalHeight() - this.getOffsetForIndex(this.visibleEnd);

    this.topSpacer.style.height = `${topHeight}px`;
    this.bottomSpacer.style.height = `${Math.max(0, bottomHeight)}px`;
  }

  measureRenderedItems() {
    this.renderedItems.forEach((element, index) => {
      const rect = element.getBoundingClientRect();
      if (rect.height > 0) {
        this.itemHeights.set(index, rect.height);
      }
    });
  }

  updateVisibleRange() {
    if (this.items.length === 0) {
      return;
    }

    // Measure current items before updating
    this.measureRenderedItems();

    const { startIndex, endIndex } = this.calculateVisibleRange();

    // Only update if range changed
    if (startIndex === this.visibleStart && endIndex === this.visibleEnd) {
      return;
    }

    // Garbage collection: remove items outside new range
    this.renderedItems.forEach((element, index) => {
      if (index < startIndex || index >= endIndex) {
        element.remove();
        this.renderedItems.delete(index);
      }
    });

    // Add new items in range
    for (let i = startIndex; i < endIndex; i++) {
      if (!this.renderedItems.has(i) && i < this.items.length) {
        const element = this.renderItem(this.items[i], i);
        element.dataset.virtualIndex = i;
        this.renderedItems.set(i, element);
      }
    }

    // Update visible range
    this.visibleStart = startIndex;
    this.visibleEnd = endIndex;

    // Reorder elements in DOM
    this.reorderElements();

    // Update spacers
    this.updateSpacers();
  }

  reorderElements() {
    // Get sorted indices
    const indices = Array.from(this.renderedItems.keys()).sort((a, b) => a - b);

    // Remove sentinel observer temporarily
    this.loadMoreObserver.unobserve(this.loadMoreSentinel);

    try {
      // Clear and rebuild container content
      const fragment = document.createDocumentFragment();
      fragment.appendChild(this.topSpacer);

      indices.forEach((index) => {
        fragment.appendChild(this.renderedItems.get(index));
      });

      fragment.appendChild(this.bottomSpacer);
      fragment.appendChild(this.loadMoreSentinel);

      // Use replaceChildren for atomic replacement (avoids memory leak)
      this.container.replaceChildren(fragment);
    } finally {
      // Re-observe sentinel (always reconnect, even on error)
      if (this.hasMore) {
        this.loadMoreObserver.observe(this.loadMoreSentinel);
      }
    }
  }

  render() {
    // Unobserve sentinel before clearing (it may have been observed from previous render)
    this.loadMoreObserver.unobserve(this.loadMoreSentinel);

    this.container.innerHTML = "";
    this.renderedItems.clear();

    if (this.items.length === 0) {
      return;
    }

    // Calculate initial visible range
    const { startIndex, endIndex } = this.calculateVisibleRange();
    this.visibleStart = startIndex;
    this.visibleEnd = endIndex;

    // Build DOM
    const fragment = document.createDocumentFragment();
    fragment.appendChild(this.topSpacer);

    for (let i = startIndex; i < endIndex && i < this.items.length; i++) {
      const element = this.renderItem(this.items[i], i);
      element.dataset.virtualIndex = i;
      this.renderedItems.set(i, element);
      fragment.appendChild(element);
    }

    fragment.appendChild(this.bottomSpacer);
    fragment.appendChild(this.loadMoreSentinel);

    this.container.appendChild(fragment);

    // Measure items synchronously to avoid scroll jumps
    // (accessing offsetHeight forces a synchronous layout)
    this.measureRenderedItems();

    // Update spacers with accurate measurements
    this.updateSpacers();

    // Observe sentinel for infinite scroll
    if (this.hasMore) {
      this.loadMoreObserver.observe(this.loadMoreSentinel);
    }
  }

  scrollToBottom() {
    this.container.scrollTop = this.container.scrollHeight;
  }

  scrollToIndex(index) {
    const offset = this.getOffsetForIndex(index);
    this.container.scrollTop = offset;
  }

  destroy() {
    // Clear pending debounced scroll handler to prevent post-destroy execution
    clearTimeout(this.scrollTimeout);

    this.container.removeEventListener("scroll", this.scrollHandler);
    this.resizeObserver.disconnect();
    this.loadMoreObserver.disconnect();

    // Explicitly remove rendered elements from DOM to prevent memory leaks
    this.renderedItems.forEach((element) => element.remove());
    this.renderedItems.clear();
    this.itemHeights.clear();

    // Clear container content
    this.container.innerHTML = "";
  }
}
