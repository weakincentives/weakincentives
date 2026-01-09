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

"""Tests for SessionView read-only session access."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.session import (
    ReadOnlySliceAccessor,
    Session,
    SessionView,
    SliceAccessor,
    as_view,
)

if TYPE_CHECKING:
    from tests.conftest import SessionFactory


@dataclass(slots=True, frozen=True)
class Plan:
    name: str
    active: bool = True


@dataclass(slots=True, frozen=True)
class Step:
    description: str


class TestSessionViewCreation:
    """Tests for SessionView construction."""

    def test_session_view_wraps_session(self, session_factory: SessionFactory) -> None:
        session, _ = session_factory()
        view = SessionView(session)

        assert view._session is session

    def test_as_view_creates_session_view(
        self, session_factory: SessionFactory
    ) -> None:
        session, _ = session_factory()
        view = as_view(session)

        assert isinstance(view, SessionView)
        assert view._session is session

    def test_session_view_has_protocol_methods(
        self, session_factory: SessionFactory
    ) -> None:
        session, _ = session_factory()
        view = SessionView(session)

        # SessionView should have all SessionViewProtocol methods
        assert hasattr(view, "__getitem__")
        assert hasattr(view, "dispatch")
        assert hasattr(view, "snapshot")
        assert hasattr(view, "dispatcher")
        assert hasattr(view, "parent")
        assert hasattr(view, "children")
        assert hasattr(view, "tags")


class TestSessionViewQueryOperations:
    """Tests for read-only query operations through SessionView."""

    def test_view_returns_read_only_slice_accessor(
        self, session_factory: SessionFactory
    ) -> None:
        session, _ = session_factory()
        session[Plan].seed(Plan(name="test"))

        view = SessionView(session)
        accessor = view[Plan]

        assert isinstance(accessor, ReadOnlySliceAccessor)
        # Ensure it's not a full SliceAccessor
        assert not isinstance(accessor, SliceAccessor)

    def test_view_all_returns_session_state(
        self, session_factory: SessionFactory
    ) -> None:
        session, _ = session_factory()
        plans = [Plan(name="first"), Plan(name="second")]
        session[Plan].seed(plans)

        view = SessionView(session)

        assert view[Plan].all() == tuple(plans)

    def test_view_latest_returns_last_item(
        self, session_factory: SessionFactory
    ) -> None:
        session, _ = session_factory()
        plans = [Plan(name="first"), Plan(name="second")]
        session[Plan].seed(plans)

        view = SessionView(session)

        assert view[Plan].latest() == Plan(name="second")

    def test_view_latest_returns_none_for_empty_slice(
        self, session_factory: SessionFactory
    ) -> None:
        session, _ = session_factory()
        view = SessionView(session)

        assert view[Plan].latest() is None

    def test_view_where_filters_items(self, session_factory: SessionFactory) -> None:
        session, _ = session_factory()
        plans = [
            Plan(name="active1", active=True),
            Plan(name="inactive", active=False),
            Plan(name="active2", active=True),
        ]
        session[Plan].seed(plans)

        view = SessionView(session)
        active_plans = view[Plan].where(lambda p: p.active)

        assert active_plans == (
            Plan(name="active1", active=True),
            Plan(name="active2", active=True),
        )


class TestSessionViewProperties:
    """Tests for SessionView property access."""

    def test_view_dispatcher_returns_session_dispatcher(
        self, session_factory: SessionFactory
    ) -> None:
        session, dispatcher = session_factory()
        view = SessionView(session)

        assert view.dispatcher is dispatcher

    def test_view_tags_returns_session_tags(
        self, session_factory: SessionFactory
    ) -> None:
        session, _ = session_factory()
        view = SessionView(session)

        assert view.tags == session.tags
        assert "session_id" in view.tags

    def test_view_parent_returns_none_for_root(
        self, session_factory: SessionFactory
    ) -> None:
        session, _ = session_factory()
        view = SessionView(session)

        assert view.parent is None

    def test_view_parent_returns_view_of_parent(self) -> None:
        dispatcher = InProcessDispatcher()
        parent = Session(dispatcher=dispatcher)
        child = Session(dispatcher=dispatcher, parent=parent)

        view = SessionView(child)
        parent_view = view.parent

        assert parent_view is not None
        assert isinstance(parent_view, SessionView)
        assert parent_view.tags["session_id"] == str(parent.session_id)

    def test_view_children_returns_views_of_children(self) -> None:
        dispatcher = InProcessDispatcher()
        parent = Session(dispatcher=dispatcher)
        child1 = Session(dispatcher=dispatcher, parent=parent)
        child2 = Session(dispatcher=dispatcher, parent=parent)

        view = SessionView(parent)
        children_views = view.children

        assert len(children_views) == 2
        assert all(isinstance(c, SessionView) for c in children_views)
        child_ids = {v.tags["session_id"] for v in children_views}
        assert child_ids == {str(child1.session_id), str(child2.session_id)}


class TestSessionViewDispatch:
    """Tests for event dispatch through SessionView."""

    def test_view_dispatch_delegates_to_session(
        self, session_factory: SessionFactory
    ) -> None:
        session, _ = session_factory()
        view = SessionView(session)

        result = view.dispatch(Plan(name="dispatched"))

        assert result.ok
        assert session[Plan].latest() == Plan(name="dispatched")

    def test_view_dispatch_returns_dispatch_result(
        self, session_factory: SessionFactory
    ) -> None:
        session, _ = session_factory()
        view = SessionView(session)

        result = view.dispatch(Step(description="step"))

        assert result.ok


class TestSessionViewSnapshot:
    """Tests for snapshot operations through SessionView."""

    def test_view_snapshot_captures_session_state(
        self, session_factory: SessionFactory
    ) -> None:
        session, _ = session_factory()
        session[Plan].seed(Plan(name="snapshotted"))

        view = SessionView(session)
        snapshot = view.snapshot(include_all=True)

        assert Plan in snapshot.slices
        assert snapshot.slices[Plan] == (Plan(name="snapshotted"),)

    def test_view_snapshot_respects_policies(
        self, session_factory: SessionFactory
    ) -> None:
        session, _ = session_factory()
        session[Plan].seed(Plan(name="test"))

        view = SessionView(session)
        # Default policies exclude LOG slices but include STATE
        snapshot = view.snapshot()

        # Plan should be included (STATE by default)
        assert Plan in snapshot.slices


class TestReadOnlySliceAccessorNoMutation:
    """Tests verifying ReadOnlySliceAccessor does not expose mutation methods."""

    def test_read_only_accessor_has_no_seed_method(
        self, session_factory: SessionFactory
    ) -> None:
        session, _ = session_factory()
        view = SessionView(session)
        accessor = view[Plan]

        assert not hasattr(accessor, "seed")

    def test_read_only_accessor_has_no_clear_method(
        self, session_factory: SessionFactory
    ) -> None:
        session, _ = session_factory()
        view = SessionView(session)
        accessor = view[Plan]

        assert not hasattr(accessor, "clear")

    def test_read_only_accessor_has_no_append_method(
        self, session_factory: SessionFactory
    ) -> None:
        session, _ = session_factory()
        view = SessionView(session)
        accessor = view[Plan]

        assert not hasattr(accessor, "append")

    def test_read_only_accessor_has_no_register_method(
        self, session_factory: SessionFactory
    ) -> None:
        session, _ = session_factory()
        view = SessionView(session)
        accessor = view[Plan]

        assert not hasattr(accessor, "register")


class TestSessionViewNoMutation:
    """Tests verifying SessionView does not expose global mutation methods."""

    def test_view_has_no_reset_method(self, session_factory: SessionFactory) -> None:
        session, _ = session_factory()
        view = SessionView(session)

        assert not hasattr(view, "reset")

    def test_view_has_no_restore_method(self, session_factory: SessionFactory) -> None:
        session, _ = session_factory()
        view = SessionView(session)

        assert not hasattr(view, "restore")

    def test_view_has_no_install_method(self, session_factory: SessionFactory) -> None:
        session, _ = session_factory()
        view = SessionView(session)

        assert not hasattr(view, "install")


class TestReducerContextReceivesSessionView:
    """Tests verifying reducers receive SessionView instead of full Session."""

    def test_reducer_context_contains_session_view(
        self, session_factory: SessionFactory
    ) -> None:
        from weakincentives.runtime.session import (
            ReducerContextProtocol,
            ReducerEvent,
        )

        session, _ = session_factory()
        captured_context: list[ReducerContextProtocol] = []

        def capturing_reducer(
            slice_values: tuple[Plan, ...],
            event: ReducerEvent,
            *,
            context: ReducerContextProtocol,
        ) -> tuple[Plan, ...]:
            captured_context.append(context)
            return (*slice_values, Plan(name="from_reducer"))

        session[Plan].register(Step, capturing_reducer)
        session.dispatch(Step(description="trigger"))

        assert len(captured_context) == 1
        ctx = captured_context[0]
        # Context.session should be a SessionView
        assert isinstance(ctx.session, SessionView)

    def test_reducer_can_read_other_slices_via_view(
        self, session_factory: SessionFactory
    ) -> None:
        from weakincentives.runtime.session import (
            Append,
            ReducerContextProtocol,
            ReducerEvent,
            SliceView,
        )

        session, _ = session_factory()
        session[Step].seed(Step(description="existing step"))

        def cross_slice_reducer(
            view: SliceView[Plan],
            event: ReducerEvent,
            *,
            context: ReducerContextProtocol,
        ) -> Append[Plan]:
            del view  # Unused in this test
            # Read from another slice via the view
            steps = context.session[Step].all()
            if steps:
                return Append(Plan(name=f"saw {len(steps)} steps"))
            return Append(Plan(name="no steps"))

        session[Plan].register(Plan, cross_slice_reducer)
        session.dispatch(Plan(name="trigger"))

        plans = session[Plan].all()
        # Should have both the dispatched plan and the one from reducer
        assert any("saw 1 steps" in p.name for p in plans)
