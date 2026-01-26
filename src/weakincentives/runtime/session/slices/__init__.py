# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Slice storage backends for session state with pluggable persistence.

This package provides storage backends for session state slices. Slices
store typed collections of frozen dataclass instances with support for
both in-memory and file-backed persistence.

Storage Backends
----------------

**MemorySlice**
    In-memory tuple-backed storage with O(1) reads. All operations work
    on immutable tuples. This is the default backend suitable for most
    use cases::

        from weakincentives.runtime.session.slices import MemorySlice

        slice: MemorySlice[Event] = MemorySlice()
        slice.append(Event(...))
        all_items = slice.all()

**JsonlSlice**
    File-backed JSONL storage optimized for large datasets and durability.
    Provides efficient append-only writes and streaming reads::

        from weakincentives.runtime.session.slices import JsonlSlice

        slice = JsonlSlice[Event](
            slice_type=Event,
            path=Path("/tmp/events.jsonl"),
        )
        slice.append(Event(...))  # Appends line to file

    JSONL format stores one JSON object per line, enabling:

    - Efficient append operations (no rewrite needed)
    - Streaming reads for memory efficiency
    - Human-readable debugging and inspection

Slice Operations
----------------

Reducers return :class:`SliceOp` values describing mutations. The
operation type determines how efficiently the backend can process it:

**Append**
    Add a single item. Most efficient for both backends - memory just
    extends the tuple, JSONL appends a line to the file::

        return Append(new_event)

**Extend**
    Add multiple items at once. More efficient than multiple Appends::

        return Extend((event1, event2, event3))

**Replace**
    Replace all items. Required when the reducer transforms existing
    state. For JSONL, this rewrites the entire file::

        filtered = tuple(item for item in view if not item.expired)
        return Replace(filtered)

**Clear**
    Remove items, optionally filtered by predicate::

        # Clear all
        return Clear()

        # Clear matching items
        return Clear(predicate=lambda x: x.expired)

SliceView Protocol
------------------

Reducers receive a :class:`SliceView` providing lazy read-only access
to current state. This enables efficient append-only reducers that
never need to load existing data::

    def append_reducer(
        view: SliceView[Event],
        event: NewEvent,
        *,
        context: ReducerContextProtocol,
    ) -> Append[Event]:
        # This reducer never accesses view - O(1) for JSONL
        return Append(Event.from_new_event(event))

    def transform_reducer(
        view: SliceView[Event],
        event: Transform,
        *,
        context: ReducerContextProtocol,
    ) -> Replace[Event]:
        # This reducer loads all items - O(n) for JSONL
        items = view.all()
        transformed = tuple(transform(item) for item in items)
        return Replace(transformed)

SliceView methods:

- ``is_empty`` - Check emptiness without loading data
- ``__len__()`` - Item count (may require scanning for JSONL)
- ``__iter__()`` - Lazy iteration (streaming for JSONL)
- ``all()`` - Load all items as tuple (cached)
- ``latest()`` - Most recent item (optimized for JSONL)
- ``where(predicate)`` - Filtered iteration (streaming)

Factory Configuration
---------------------

Configure backends per slice policy with :class:`SliceFactoryConfig`::

    from pathlib import Path
    from weakincentives.runtime.session.slices import (
        MemorySliceFactory,
        JsonlSliceFactory,
        SliceFactoryConfig,
    )

    # STATE slices in memory, LOG slices on disk
    config = SliceFactoryConfig(
        state_factory=MemorySliceFactory(),
        log_factory=JsonlSliceFactory(base_dir=Path("/var/log/session")),
        transient_factory=MemorySliceFactory(),
    )

    session = Session(slice_config=config)

The default configuration uses :class:`MemorySliceFactory` for all
policies, which is appropriate for most single-process deployments.

Use :func:`default_slice_config` to get the default configuration::

    from weakincentives.runtime.session.slices import default_slice_config

    config = default_slice_config()  # All MemorySlice

Backend Selection Guidelines
----------------------------

**Use MemorySlice when:**

- Data fits comfortably in memory
- Fast random access is important
- Durability is not required
- Session lifetime is short

**Use JsonlSlice when:**

- Data may grow large (thousands of items)
- Durability across restarts is needed
- Append-only access pattern dominates
- Memory efficiency is critical

Exports
-------

**Protocols:**
    - :class:`Slice` - Storage backend protocol
    - :class:`SliceView` - Read-only lazy view protocol
    - :class:`SliceFactory` - Factory protocol for creating slices

**Operations:**
    - :class:`SliceOp` - Union of all operation types
    - :class:`Append` - Add single item
    - :class:`Extend` - Add multiple items
    - :class:`Replace` - Replace all items
    - :class:`Clear` - Remove items (optionally filtered)

**Memory Backend:**
    - :class:`MemorySlice` - In-memory tuple storage
    - :class:`MemorySliceFactory` - Factory for MemorySlice
    - :class:`MemorySliceView` - Immutable view snapshot

**JSONL Backend:**
    - :class:`JsonlSlice` - File-backed JSONL storage
    - :class:`JsonlSliceFactory` - Factory for JsonlSlice
    - :class:`JsonlSliceView` - Streaming file view

**Configuration:**
    - :class:`SliceFactoryConfig` - Per-policy factory configuration
    - :func:`default_slice_config` - Default configuration factory
"""

from ._config import SliceFactoryConfig, default_slice_config
from ._jsonl import JsonlSlice, JsonlSliceFactory, JsonlSliceView
from ._memory import MemorySlice, MemorySliceFactory, MemorySliceView
from ._ops import Append, Clear, Extend, Replace, SliceOp
from ._protocols import Slice, SliceFactory, SliceView

__all__ = [
    "Append",
    "Clear",
    "Extend",
    "JsonlSlice",
    "JsonlSliceFactory",
    "JsonlSliceView",
    "MemorySlice",
    "MemorySliceFactory",
    "MemorySliceView",
    "Replace",
    "Slice",
    "SliceFactory",
    "SliceFactoryConfig",
    "SliceOp",
    "SliceView",
    "default_slice_config",
]
