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

"""Tests for slice storage backends."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from weakincentives.runtime.session import (
    Clear,
    Extend,
    MemorySlice,
    MemorySliceView,
)
from weakincentives.runtime.session.slices._jsonl import (
    JsonlSlice,
    JsonlSliceFactory,
)

if TYPE_CHECKING:
    from tests.conftest import SessionFactory


# ──────────────────────────────────────────────────────────────────────
# Test data types
# ──────────────────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class SimpleItem:
    """Simple item for testing."""

    key: str
    value: int = 0


# ──────────────────────────────────────────────────────────────────────
# MemorySlice tests
# ──────────────────────────────────────────────────────────────────────


def test_memory_slice_view_is_frozen() -> None:
    """MemorySliceView is immutable (frozen dataclass)."""
    view = MemorySliceView((SimpleItem("a", 1),))

    # Frozen dataclass can't be modified
    assert view._data == (SimpleItem("a", 1),)
    assert view.is_empty is False
    assert len(view) == 1


def test_memory_slice_view_iteration() -> None:
    """MemorySliceView supports iteration."""
    items = (SimpleItem("a", 1), SimpleItem("b", 2))
    view = MemorySliceView(items)

    collected = list(view)
    assert collected == list(items)


def test_memory_slice_view_all() -> None:
    """MemorySliceView.all() returns tuple."""
    items = (SimpleItem("a", 1), SimpleItem("b", 2))
    view = MemorySliceView(items)

    assert view.all() == items


def test_memory_slice_view_latest() -> None:
    """MemorySliceView.latest() returns last item."""
    view = MemorySliceView((SimpleItem("a", 1), SimpleItem("b", 2)))
    assert view.latest() == SimpleItem("b", 2)

    empty_view = MemorySliceView[SimpleItem](())
    assert empty_view.latest() is None


def test_memory_slice_view_where() -> None:
    """MemorySliceView.where() filters items."""
    items = (SimpleItem("a", 1), SimpleItem("b", 2), SimpleItem("c", 1))
    view = MemorySliceView(items)

    filtered = list(view.where(lambda x: x.value == 1))
    assert filtered == [SimpleItem("a", 1), SimpleItem("c", 1)]


def test_memory_slice_extend() -> None:
    """MemorySlice.extend() adds multiple items."""
    slice_obj = MemorySlice[SimpleItem]()
    slice_obj.extend([SimpleItem("a", 1), SimpleItem("b", 2)])

    assert len(slice_obj) == 2
    assert slice_obj.all() == (SimpleItem("a", 1), SimpleItem("b", 2))


def test_memory_slice_is_empty() -> None:
    """MemorySlice.is_empty property works."""
    slice_obj = MemorySlice[SimpleItem]()
    assert slice_obj.is_empty is True

    slice_obj.append(SimpleItem("a", 1))
    assert slice_obj.is_empty is False


def test_memory_slice_len() -> None:
    """MemorySlice.__len__() works."""
    slice_obj = MemorySlice((SimpleItem("a", 1), SimpleItem("b", 2)))
    assert len(slice_obj) == 2


def test_memory_slice_iter() -> None:
    """MemorySlice.__iter__() works."""
    items = (SimpleItem("a", 1), SimpleItem("b", 2))
    slice_obj = MemorySlice(items)

    collected = list(slice_obj)
    assert collected == list(items)


def test_memory_slice_latest() -> None:
    """MemorySlice.latest() returns last item."""
    slice_obj = MemorySlice((SimpleItem("a", 1), SimpleItem("b", 2)))
    assert slice_obj.latest() == SimpleItem("b", 2)

    empty_slice = MemorySlice[SimpleItem]()
    assert empty_slice.latest() is None


def test_memory_slice_where() -> None:
    """MemorySlice.where() filters items."""
    items = (SimpleItem("a", 1), SimpleItem("b", 2), SimpleItem("c", 1))
    slice_obj = MemorySlice(items)

    filtered = list(slice_obj.where(lambda x: x.value == 1))
    assert filtered == [SimpleItem("a", 1), SimpleItem("c", 1)]


# ──────────────────────────────────────────────────────────────────────
# JsonlSlice tests
# ──────────────────────────────────────────────────────────────────────


def test_jsonl_slice_basic_operations() -> None:
    """JsonlSlice supports basic append/all/latest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        slice_obj = JsonlSlice(path=path, item_type=SimpleItem)

        # Initially empty
        assert slice_obj.all() == ()
        assert slice_obj.latest() is None
        assert len(slice_obj) == 0

        # Append items
        slice_obj.append(SimpleItem("a", 1))
        slice_obj.append(SimpleItem("b", 2))

        # Read back
        assert len(slice_obj) == 2
        assert slice_obj.latest() == SimpleItem("b", 2)
        all_items = slice_obj.all()
        assert all_items == (SimpleItem("a", 1), SimpleItem("b", 2))


def test_jsonl_slice_extend() -> None:
    """JsonlSlice.extend() adds multiple items."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        slice_obj = JsonlSlice(path=path, item_type=SimpleItem)

        slice_obj.extend([SimpleItem("a", 1), SimpleItem("b", 2)])

        assert len(slice_obj) == 2
        assert slice_obj.all() == (SimpleItem("a", 1), SimpleItem("b", 2))


def test_jsonl_slice_replace() -> None:
    """JsonlSlice.replace() replaces all items."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        slice_obj = JsonlSlice(path=path, item_type=SimpleItem)

        slice_obj.append(SimpleItem("a", 1))
        slice_obj.replace((SimpleItem("x", 99),))

        assert len(slice_obj) == 1
        assert slice_obj.latest() == SimpleItem("x", 99)


def test_jsonl_slice_clear() -> None:
    """JsonlSlice.clear() removes all items."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        slice_obj = JsonlSlice(path=path, item_type=SimpleItem)

        slice_obj.append(SimpleItem("a", 1))
        slice_obj.append(SimpleItem("b", 2))
        slice_obj.clear()

        assert len(slice_obj) == 0
        assert slice_obj.all() == ()


def test_jsonl_slice_clear_with_predicate() -> None:
    """JsonlSlice.clear() with predicate removes matching items."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        slice_obj = JsonlSlice(path=path, item_type=SimpleItem)

        slice_obj.extend([SimpleItem("a", 1), SimpleItem("b", 2), SimpleItem("c", 1)])
        slice_obj.clear(lambda x: x.value == 1)

        assert len(slice_obj) == 1
        assert slice_obj.all() == (SimpleItem("b", 2),)


def test_jsonl_slice_snapshot() -> None:
    """JsonlSlice.snapshot() returns tuple."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        slice_obj = JsonlSlice(path=path, item_type=SimpleItem)

        slice_obj.extend([SimpleItem("a", 1), SimpleItem("b", 2)])
        snapshot = slice_obj.snapshot()

        assert snapshot == (SimpleItem("a", 1), SimpleItem("b", 2))


def test_jsonl_slice_view() -> None:
    """JsonlSlice.view() returns JsonlSliceView."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        slice_obj = JsonlSlice(path=path, item_type=SimpleItem)

        slice_obj.extend([SimpleItem("a", 1), SimpleItem("b", 2)])
        view = slice_obj.view()

        assert view.is_empty is False
        assert len(view) == 2
        assert list(view) == [SimpleItem("a", 1), SimpleItem("b", 2)]
        assert view.all() == (SimpleItem("a", 1), SimpleItem("b", 2))
        assert view.latest() == SimpleItem("b", 2)


def test_jsonl_slice_view_where() -> None:
    """JsonlSliceView.where() filters items."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        slice_obj = JsonlSlice(path=path, item_type=SimpleItem)

        slice_obj.extend([SimpleItem("a", 1), SimpleItem("b", 2), SimpleItem("c", 1)])
        view = slice_obj.view()

        filtered = list(view.where(lambda x: x.value == 1))
        assert filtered == [SimpleItem("a", 1), SimpleItem("c", 1)]


def test_jsonl_slice_view_empty() -> None:
    """JsonlSliceView handles empty slice."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        slice_obj = JsonlSlice(path=path, item_type=SimpleItem)

        view = slice_obj.view()

        assert view.is_empty is True
        assert len(view) == 0
        assert list(view) == []
        assert view.latest() is None


def test_jsonl_slice_factory() -> None:
    """JsonlSliceFactory creates slices with correct path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        factory = JsonlSliceFactory(base_dir=Path(tmpdir))

        slice_obj = factory.create(SimpleItem)
        slice_obj.append(SimpleItem("a", 1))

        assert slice_obj.latest() == SimpleItem("a", 1)
        assert factory.directory == Path(tmpdir)


def test_jsonl_slice_factory_temp_dir() -> None:
    """JsonlSliceFactory creates temp directory if not specified."""
    factory = JsonlSliceFactory()

    assert factory.directory.exists()
    slice_obj = factory.create(SimpleItem)
    slice_obj.append(SimpleItem("a", 1))

    assert slice_obj.latest() == SimpleItem("a", 1)


def test_jsonl_slice_view_len_uses_cache() -> None:
    """JsonlSliceView.__len__() uses cache when available."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        slice_obj = JsonlSlice(path=path, item_type=SimpleItem)

        slice_obj.extend([SimpleItem("a", 1), SimpleItem("b", 2)])

        # Force cache population by calling all()
        _ = slice_obj.all()

        # Now view should use cache for __len__
        view = slice_obj.view()
        assert len(view) == 2


def test_jsonl_slice_view_latest_uses_cache() -> None:
    """JsonlSliceView.latest() uses cache when available."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        slice_obj = JsonlSlice(path=path, item_type=SimpleItem)

        slice_obj.extend([SimpleItem("a", 1), SimpleItem("b", 2)])

        # Force cache population by calling all()
        _ = slice_obj.all()

        # Now view should use cache for latest()
        view = slice_obj.view()
        assert view.latest() == SimpleItem("b", 2)


def test_jsonl_slice_view_latest_empty_file() -> None:
    """JsonlSliceView.latest() returns None for empty existing file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        # Create empty file
        path.touch()

        slice_obj = JsonlSlice(path=path, item_type=SimpleItem)
        view = slice_obj.view()

        assert view.latest() is None


def test_jsonl_slice_view_latest_no_cache() -> None:
    """JsonlSliceView.latest() loads file when no cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        slice_obj = JsonlSlice(path=path, item_type=SimpleItem)

        slice_obj.extend([SimpleItem("a", 1), SimpleItem("b", 2)])

        # Invalidate cache
        slice_obj._cache = None

        # View should load from file
        view = slice_obj.view()
        assert view.latest() == SimpleItem("b", 2)


def test_jsonl_slice_handles_empty_lines_in_file() -> None:
    """JsonlSlice handles files with empty lines gracefully."""
    import json

    from weakincentives.serde import dump

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        # Manually create file with empty lines
        with path.open("w") as f:
            # Write a valid item
            data = dump(SimpleItem("a", 1), include_dataclass_type=True)
            f.write(json.dumps(data) + "\n")
            # Empty line
            f.write("\n")
            # Another item
            data = dump(SimpleItem("b", 2), include_dataclass_type=True)
            f.write(json.dumps(data) + "\n")
            # Whitespace-only line
            f.write("   \n")

        slice_obj = JsonlSlice(path=path, item_type=SimpleItem)

        # Should skip empty/whitespace lines
        items = slice_obj.all()
        assert items == (SimpleItem("a", 1), SimpleItem("b", 2))


def test_jsonl_slice_view_iteration_skips_empty_lines() -> None:
    """JsonlSliceView iteration skips empty lines."""
    import json

    from weakincentives.serde import dump

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        # Manually create file with empty lines
        with path.open("w") as f:
            data = dump(SimpleItem("a", 1), include_dataclass_type=True)
            f.write(json.dumps(data) + "\n")
            f.write("\n")  # Empty line
            data = dump(SimpleItem("b", 2), include_dataclass_type=True)
            f.write(json.dumps(data) + "\n")

        slice_obj = JsonlSlice(path=path, item_type=SimpleItem)
        view = slice_obj.view()

        # Invalidate cache to force file read via view iteration
        slice_obj._cache = None

        # Iterate via view - should skip empty lines
        items = list(view)
        assert items == [SimpleItem("a", 1), SimpleItem("b", 2)]


# ──────────────────────────────────────────────────────────────────────
# Session SliceOp integration tests
# ──────────────────────────────────────────────────────────────────────


def test_session_extend_slice_op(session_factory: SessionFactory) -> None:
    """Session applies Extend SliceOp correctly."""
    from weakincentives.runtime.session._types import ReducerContextProtocol
    from weakincentives.runtime.session.slices import SliceView

    @dataclass(slots=True, frozen=True)
    class BatchEvent:
        items: tuple[SimpleItem, ...]

    session, _ = session_factory()

    def batch_reducer(
        view: SliceView[SimpleItem],
        event: BatchEvent,
        *,
        context: ReducerContextProtocol,
    ) -> Extend[SimpleItem]:
        del context, view
        return Extend(event.items)

    session[SimpleItem].register(BatchEvent, batch_reducer)

    session.dispatch(BatchEvent(items=(SimpleItem("a", 1), SimpleItem("b", 2))))

    assert session[SimpleItem].all() == (SimpleItem("a", 1), SimpleItem("b", 2))


def test_session_clear_slice_op(session_factory: SessionFactory) -> None:
    """Session applies Clear SliceOp correctly."""
    from weakincentives.runtime.session._types import ReducerContextProtocol
    from weakincentives.runtime.session.slices import SliceView

    @dataclass(slots=True, frozen=True)
    class ClearEvent:
        pass

    session, _ = session_factory()

    def clear_reducer(
        view: SliceView[SimpleItem],
        event: ClearEvent,
        *,
        context: ReducerContextProtocol,
    ) -> Clear[SimpleItem]:
        del context, view, event
        return Clear()

    session[SimpleItem].register(ClearEvent, clear_reducer)

    # Add some initial data
    session[SimpleItem].seed((SimpleItem("a", 1), SimpleItem("b", 2)))

    # Clear it
    session.dispatch(ClearEvent())

    assert session[SimpleItem].all() == ()


def test_session_clear_with_predicate_slice_op(
    session_factory: SessionFactory,
) -> None:
    """Session applies Clear SliceOp with predicate correctly."""
    from weakincentives.runtime.session._types import ReducerContextProtocol
    from weakincentives.runtime.session.slices import SliceView

    @dataclass(slots=True, frozen=True)
    class ClearByValueEvent:
        value: int

    session, _ = session_factory()

    def clear_by_value_reducer(
        view: SliceView[SimpleItem],
        event: ClearByValueEvent,
        *,
        context: ReducerContextProtocol,
    ) -> Clear[SimpleItem]:
        del context, view
        return Clear(lambda x: x.value == event.value)

    session[SimpleItem].register(ClearByValueEvent, clear_by_value_reducer)

    # Add some initial data
    session[SimpleItem].seed(
        (SimpleItem("a", 1), SimpleItem("b", 2), SimpleItem("c", 1))
    )

    # Clear items with value 1
    session.dispatch(ClearByValueEvent(value=1))

    assert session[SimpleItem].all() == (SimpleItem("b", 2),)
