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

from __future__ import annotations

import pytest

from weakincentives.runtime.clock import FakeClock
from weakincentives.runtime.session import Session, Snapshot, iter_sessions_bottom_up


def test_session_tracks_parent_and_children(clock: FakeClock) -> None:
    root = Session(clock=clock)
    left_child = Session(parent=root, clock=clock)
    right_child = Session(parent=root, clock=clock)
    grandchild = Session(parent=left_child, clock=clock)

    assert left_child.parent is root
    assert right_child.parent is root
    assert grandchild.parent is left_child

    assert root.children == (left_child, right_child)
    assert left_child.children == (grandchild,)
    assert right_child.children == ()
    assert grandchild.children == ()


def test_iter_sessions_bottom_up_orders_leaves_first(clock: FakeClock) -> None:
    root = Session(clock=clock)
    left_child = Session(parent=root, clock=clock)
    right_child = Session(parent=root, clock=clock)
    grandchild = Session(parent=left_child, clock=clock)

    traversal = tuple(iter_sessions_bottom_up(root))

    assert traversal == (grandchild, left_child, right_child, root)


def test_iter_sessions_bottom_up_skips_cycles(clock: FakeClock) -> None:
    root = Session(clock=clock)
    child = Session(parent=root, clock=clock)
    child._register_child(root)

    traversal = tuple(iter_sessions_bottom_up(root))

    assert traversal == (child, root)


def test_register_child_ignores_duplicates(clock: FakeClock) -> None:
    parent = Session(clock=clock)
    child = Session(parent=parent, clock=clock)

    parent._register_child(child)

    assert parent.children == (child,)


def test_session_rejects_self_parent(clock: FakeClock) -> None:
    session = Session.__new__(Session)

    with pytest.raises(ValueError):
        Session.__init__(session, parent=session, clock=clock)


def test_session_snapshot_carries_tags(clock: FakeClock) -> None:
    root = Session(tags={"scope": "root"}, clock=clock)
    child = Session(parent=root, tags={"scope": "child"}, clock=clock)

    snapshot = child.snapshot()

    assert snapshot.tags["scope"] == "child"
    assert snapshot.tags["session_id"] == str(child.session_id)
    assert snapshot.tags["parent_session_id"] == str(root.session_id)

    restored = Snapshot.from_json(snapshot.to_json())
    assert restored.tags == snapshot.tags


def test_session_rejects_non_string_tags(clock: FakeClock) -> None:
    with pytest.raises(TypeError):
        Session(tags={"scope": 123}, clock=clock)
