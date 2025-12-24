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

"""Tests for SessionState helper class.

These tests verify that the SessionState helper correctly manages state,
reducers, observers, and dispatch logic after the refactoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.session import (
    EMPTY_SLICE,
    Session,
    SessionState,
    SlicePolicy,
    append_all,
    replace_latest,
)

if TYPE_CHECKING:
    from tests.conftest import SessionFactory


@dataclass(slots=True, frozen=True)
class ExamplePayload:
    value: int


@dataclass(slots=True, frozen=True)
class ExampleOutput:
    text: str


# ──────────────────────────────────────────────────────────────────────
# SessionState unit tests
# ──────────────────────────────────────────────────────────────────────


class TestSessionStateInit:
    def test_initializes_with_empty_state(self) -> None:
        state = SessionState()

        assert state.get_state_snapshot() == {}
        assert state.get_reducers_snapshot() == []
        assert state.registered_slice_types() == set()

    def test_initializes_with_provided_policies(self) -> None:
        policies = {ExamplePayload: SlicePolicy.LOG}
        state = SessionState(initial_policies=policies)

        assert state.get_policy(ExamplePayload) == SlicePolicy.LOG


class TestSessionStateQueryOperations:
    def test_select_all_returns_empty_tuple_for_unknown_slice(self) -> None:
        state = SessionState()

        result = state.select_all(ExamplePayload)

        assert result == ()

    def test_select_all_returns_slice_values(self) -> None:
        state = SessionState()
        state.seed_slice(ExamplePayload, [ExamplePayload(1), ExamplePayload(2)])

        result = state.select_all(ExamplePayload)

        assert result == (ExamplePayload(1), ExamplePayload(2))

    def test_registered_slice_types_returns_state_keys(self) -> None:
        state = SessionState()
        state.seed_slice(ExamplePayload, [ExamplePayload(1)])

        types = state.registered_slice_types()

        assert ExamplePayload in types

    def test_registered_slice_types_includes_reducer_targets(self) -> None:
        state = SessionState()
        state.register_reducer(ExampleOutput, append_all, slice_type=ExamplePayload)

        types = state.registered_slice_types()

        assert ExamplePayload in types

    def test_get_policy_returns_default_for_unknown_slice(self) -> None:
        state = SessionState()

        policy = state.get_policy(ExamplePayload)

        assert policy == SlicePolicy.STATE

    def test_get_policy_returns_configured_policy(self) -> None:
        state = SessionState(initial_policies={ExamplePayload: SlicePolicy.LOG})

        policy = state.get_policy(ExamplePayload)

        assert policy == SlicePolicy.LOG

    def test_get_state_snapshot_returns_copy(self) -> None:
        state = SessionState()
        state.seed_slice(ExamplePayload, [ExamplePayload(1)])

        snapshot = state.get_state_snapshot()
        snapshot[ExamplePayload] = ()

        # Original unaffected
        assert state.select_all(ExamplePayload) == (ExamplePayload(1),)


class TestSessionStateMutations:
    def test_seed_slice_replaces_existing_values(self) -> None:
        state = SessionState()
        state.seed_slice(ExamplePayload, [ExamplePayload(1)])
        state.seed_slice(ExamplePayload, [ExamplePayload(2), ExamplePayload(3)])

        result = state.select_all(ExamplePayload)

        assert result == (ExamplePayload(2), ExamplePayload(3))

    def test_clear_slice_removes_all_values(self) -> None:
        state = SessionState()
        state.seed_slice(ExamplePayload, [ExamplePayload(1), ExamplePayload(2)])

        state.clear_slice(ExamplePayload)

        assert state.select_all(ExamplePayload) == ()

    def test_clear_slice_with_predicate_filters_values(self) -> None:
        state = SessionState()
        state.seed_slice(
            ExamplePayload, [ExamplePayload(1), ExamplePayload(2), ExamplePayload(3)]
        )

        state.clear_slice(ExamplePayload, lambda p: p.value % 2 == 1)

        assert state.select_all(ExamplePayload) == (ExamplePayload(2),)

    def test_clear_slice_noop_for_empty_slice(self) -> None:
        state = SessionState()

        state.clear_slice(ExamplePayload)

        assert state.select_all(ExamplePayload) == ()


class TestSessionStateReducerRegistration:
    def test_register_reducer_creates_empty_slice(self) -> None:
        state = SessionState()

        state.register_reducer(ExamplePayload, append_all)

        assert ExamplePayload in state.registered_slice_types()
        assert state.select_all(ExamplePayload) == EMPTY_SLICE

    def test_register_reducer_sets_policy(self) -> None:
        state = SessionState()

        state.register_reducer(ExamplePayload, append_all, policy=SlicePolicy.LOG)

        assert state.get_policy(ExamplePayload) == SlicePolicy.LOG

    def test_register_reducer_allows_different_slice_type(self) -> None:
        state = SessionState()

        state.register_reducer(ExampleOutput, append_all, slice_type=ExamplePayload)

        # Verify via get_reducers_snapshot
        reducer_snapshot = state.get_reducers_snapshot()
        output_reducers = [r for t, r in reducer_snapshot if t is ExampleOutput]
        assert len(output_reducers) == 1
        assert output_reducers[0][0].slice_type is ExamplePayload

    def test_has_reducer_returns_true_when_registered(self) -> None:
        state = SessionState()
        state.register_reducer(ExamplePayload, append_all)

        assert state.has_reducer(ExamplePayload) is True

    def test_has_reducer_returns_false_when_not_registered(self) -> None:
        state = SessionState()

        assert state.has_reducer(ExamplePayload) is False


class TestSessionStateObservers:
    def test_register_observer_returns_subscription(self) -> None:
        state = SessionState()

        def observer(
            old: tuple[ExamplePayload, ...], new: tuple[ExamplePayload, ...]
        ) -> None:
            pass

        subscription = state.register_observer(ExamplePayload, observer)

        assert subscription is not None
        assert subscription.subscription_id is not None

    def test_subscription_unsubscribe_removes_observer(self) -> None:
        state = SessionState()
        calls: list[tuple[ExamplePayload, ...]] = []

        def observer(
            old: tuple[ExamplePayload, ...], new: tuple[ExamplePayload, ...]
        ) -> None:
            calls.append(new)

        subscription = state.register_observer(ExamplePayload, observer)
        # Verify observer is registered by calling notify
        state.notify_observers({ExamplePayload: ((), (ExamplePayload(1),))})
        assert len(calls) == 1

        result = subscription.unsubscribe()
        assert result is True

        # Verify observer is removed by calling notify again
        state.notify_observers({ExamplePayload: ((), (ExamplePayload(2),))})
        assert len(calls) == 1  # Still only 1 call

    def test_notify_observers_calls_registered_observers(self) -> None:
        state = SessionState()
        calls: list[tuple[tuple[ExamplePayload, ...], tuple[ExamplePayload, ...]]] = []

        def observer(
            old: tuple[ExamplePayload, ...], new: tuple[ExamplePayload, ...]
        ) -> None:
            calls.append((old, new))

        state.register_observer(ExamplePayload, observer)

        state.notify_observers({ExamplePayload: ((), (ExamplePayload(1),))})

        assert len(calls) == 1
        assert calls[0] == ((), (ExamplePayload(1),))

    def test_notify_observers_handles_exceptions(self) -> None:
        state = SessionState()
        calls: list[str] = []

        def failing_observer(
            old: tuple[ExamplePayload, ...], new: tuple[ExamplePayload, ...]
        ) -> None:
            raise RuntimeError("Observer failed")

        def working_observer(
            old: tuple[ExamplePayload, ...], new: tuple[ExamplePayload, ...]
        ) -> None:
            calls.append("called")

        state.register_observer(ExamplePayload, failing_observer)
        state.register_observer(ExamplePayload, working_observer)

        state.notify_observers({ExamplePayload: ((), (ExamplePayload(1),))})

        assert calls == ["called"]


class TestSessionStateReset:
    def test_reset_clears_all_slices(self) -> None:
        state = SessionState()
        state.seed_slice(ExamplePayload, [ExamplePayload(1)])
        state.seed_slice(ExampleOutput, [ExampleOutput("hello")])

        state.reset()

        assert state.select_all(ExamplePayload) == ()
        assert state.select_all(ExampleOutput) == ()

    def test_reset_preserves_reducer_slice_types(self) -> None:
        state = SessionState()
        state.register_reducer(ExamplePayload, append_all)
        state.seed_slice(ExamplePayload, [ExamplePayload(1)])

        state.reset()

        # Slice type still registered but is empty
        assert ExamplePayload in state.registered_slice_types()
        assert state.select_all(ExamplePayload) == ()


class TestSessionStateCloneSupport:
    def test_copy_reducers_from_copies_registrations(self) -> None:
        source = SessionState()
        source.register_reducer(ExamplePayload, append_all)
        target = SessionState()

        reducer_snapshot = source.get_reducers_snapshot()
        target.copy_reducers_from(reducer_snapshot)

        assert target.has_reducer(ExamplePayload)

    def test_copy_reducers_skips_existing_registrations(self) -> None:
        source = SessionState()
        source.register_reducer(ExamplePayload, append_all)
        target = SessionState()
        target.register_reducer(ExamplePayload, replace_latest)

        reducer_snapshot = source.get_reducers_snapshot()
        target.copy_reducers_from(reducer_snapshot)

        # Only one registration (the original replace_latest, not append_all)
        target_snapshot = target.get_reducers_snapshot()
        payload_registrations = [r for t, r in target_snapshot if t is ExamplePayload]
        assert len(payload_registrations) == 1
        assert len(payload_registrations[0]) == 1


# ──────────────────────────────────────────────────────────────────────
# Session delegation tests (verify Session correctly delegates to SessionState)
# ──────────────────────────────────────────────────────────────────────


class TestSessionDelegation:
    def test_session_exposes_state_manager(self) -> None:
        session = Session(bus=InProcessDispatcher())

        assert isinstance(session.state_manager, SessionState)

    def test_session_locked_uses_state_manager_lock(self) -> None:
        session = Session(bus=InProcessDispatcher())

        with session.locked():
            # Should be able to acquire state manager lock too (reentrant)
            with session.state_manager.locked():
                pass

    def test_clone_preserves_state_via_state_manager(
        self, session_factory: SessionFactory
    ) -> None:
        session, _ = session_factory()
        session[ExamplePayload].seed([ExamplePayload(1), ExamplePayload(2)])

        clone_bus = InProcessDispatcher()
        clone = session.clone(bus=clone_bus)

        assert clone[ExamplePayload].all() == (ExamplePayload(1), ExamplePayload(2))

    def test_clone_preserves_reducers_via_state_manager(
        self, session_factory: SessionFactory
    ) -> None:
        session, _ = session_factory()
        session[ExamplePayload].register(ExamplePayload, replace_latest)
        session[ExamplePayload].append(ExamplePayload(1))

        clone_bus = InProcessDispatcher()
        clone = session.clone(bus=clone_bus)
        clone[ExamplePayload].append(ExamplePayload(2))

        # replace_latest should be active in clone
        assert clone[ExamplePayload].all() == (ExamplePayload(2),)

    def test_clone_preserves_policies_via_state_manager(
        self, session_factory: SessionFactory
    ) -> None:
        session, _ = session_factory()
        session[ExamplePayload].register(
            ExamplePayload, append_all, policy=SlicePolicy.LOG
        )
        session[ExamplePayload].seed([ExamplePayload(1)])

        clone_bus = InProcessDispatcher()
        clone = session.clone(bus=clone_bus)

        # LOG policy should be preserved
        snapshot = clone.snapshot()
        assert ExamplePayload not in snapshot.slices

    def test_reset_delegates_to_state_manager(
        self, session_factory: SessionFactory
    ) -> None:
        session, _ = session_factory()
        session[ExamplePayload].register(ExamplePayload, append_all)
        session[ExamplePayload].append(ExamplePayload(1))

        session.reset()

        assert session[ExamplePayload].all() == ()

    def test_restore_delegates_to_state_manager(
        self, session_factory: SessionFactory
    ) -> None:
        session, _ = session_factory()
        session[ExamplePayload].register(ExamplePayload, append_all)
        session[ExamplePayload].append(ExamplePayload(1))
        snapshot = session.snapshot()

        session[ExamplePayload].append(ExamplePayload(2))
        session.restore(snapshot)

        assert session[ExamplePayload].all() == (ExamplePayload(1),)


class TestObserverCallbacksAfterRefactor:
    def test_observer_called_on_dispatch(self, session_factory: SessionFactory) -> None:
        session, _ = session_factory()
        calls: list[tuple[tuple[ExamplePayload, ...], tuple[ExamplePayload, ...]]] = []

        def observer(
            old: tuple[ExamplePayload, ...], new: tuple[ExamplePayload, ...]
        ) -> None:
            calls.append((old, new))

        session.observe(ExamplePayload, observer)
        session[ExamplePayload].append(ExamplePayload(1))

        assert len(calls) == 1
        assert calls[0] == ((), (ExamplePayload(1),))

    def test_observer_receives_old_and_new_values(
        self, session_factory: SessionFactory
    ) -> None:
        session, _ = session_factory()
        calls: list[tuple[tuple[ExamplePayload, ...], tuple[ExamplePayload, ...]]] = []

        def observer(
            old: tuple[ExamplePayload, ...], new: tuple[ExamplePayload, ...]
        ) -> None:
            calls.append((old, new))

        session.observe(ExamplePayload, observer)
        session[ExamplePayload].append(ExamplePayload(1))
        session[ExamplePayload].append(ExamplePayload(2))

        assert len(calls) == 2
        assert calls[1] == (
            (ExamplePayload(1),),
            (ExamplePayload(1), ExamplePayload(2)),
        )

    def test_multiple_observers_all_called(
        self, session_factory: SessionFactory
    ) -> None:
        session, _ = session_factory()
        first_calls: list[tuple[ExamplePayload, ...]] = []
        second_calls: list[tuple[ExamplePayload, ...]] = []

        def first_observer(
            old: tuple[ExamplePayload, ...], new: tuple[ExamplePayload, ...]
        ) -> None:
            first_calls.append(new)

        def second_observer(
            old: tuple[ExamplePayload, ...], new: tuple[ExamplePayload, ...]
        ) -> None:
            second_calls.append(new)

        session.observe(ExamplePayload, first_observer)
        session.observe(ExamplePayload, second_observer)
        session[ExamplePayload].append(ExamplePayload(1))

        assert first_calls == [(ExamplePayload(1),)]
        assert second_calls == [(ExamplePayload(1),)]

    def test_unsubscribe_stops_notifications(
        self, session_factory: SessionFactory
    ) -> None:
        session, _ = session_factory()
        calls: list[tuple[ExamplePayload, ...]] = []

        def observer(
            old: tuple[ExamplePayload, ...], new: tuple[ExamplePayload, ...]
        ) -> None:
            calls.append(new)

        subscription = session.observe(ExamplePayload, observer)
        session[ExamplePayload].append(ExamplePayload(1))
        assert len(calls) == 1

        subscription.unsubscribe()
        session[ExamplePayload].append(ExamplePayload(2))

        # No new calls after unsubscribe
        assert len(calls) == 1


class TestSnapshotPoliciesAfterRefactor:
    def test_snapshot_filters_log_slices_by_default(
        self, session_factory: SessionFactory
    ) -> None:
        session, _ = session_factory()
        session[ExamplePayload].register(
            ExamplePayload, append_all, policy=SlicePolicy.LOG
        )
        session[ExamplePayload].append(ExamplePayload(1))
        session[ExampleOutput].register(
            ExampleOutput, append_all, policy=SlicePolicy.STATE
        )
        session[ExampleOutput].seed([ExampleOutput("hello")])

        snapshot = session.snapshot()

        assert ExamplePayload not in snapshot.slices
        assert ExampleOutput in snapshot.slices

    def test_snapshot_includes_logs_when_requested(
        self, session_factory: SessionFactory
    ) -> None:
        session, _ = session_factory()
        session[ExamplePayload].register(
            ExamplePayload, append_all, policy=SlicePolicy.LOG
        )
        session[ExamplePayload].append(ExamplePayload(1))

        snapshot = session.snapshot(
            policies=frozenset({SlicePolicy.STATE, SlicePolicy.LOG})
        )

        assert ExamplePayload in snapshot.slices

    def test_snapshot_include_all_ignores_policies(
        self, session_factory: SessionFactory
    ) -> None:
        session, _ = session_factory()
        session[ExamplePayload].register(
            ExamplePayload, append_all, policy=SlicePolicy.LOG
        )
        session[ExamplePayload].append(ExamplePayload(1))

        snapshot = session.snapshot(include_all=True)

        assert ExamplePayload in snapshot.slices

    def test_restore_preserves_log_slices_by_default(
        self, session_factory: SessionFactory
    ) -> None:
        session, _ = session_factory()
        session[ExamplePayload].register(
            ExamplePayload, append_all, policy=SlicePolicy.LOG
        )
        session[ExamplePayload].append(ExamplePayload(1))
        snapshot = session.snapshot(include_all=True)

        session[ExamplePayload].append(ExamplePayload(2))
        session.restore(snapshot)

        # LOG slices preserved (not restored)
        assert session[ExamplePayload].all() == (ExamplePayload(1), ExamplePayload(2))

    def test_restore_can_overwrite_log_slices(
        self, session_factory: SessionFactory
    ) -> None:
        session, _ = session_factory()
        session[ExamplePayload].register(
            ExamplePayload, append_all, policy=SlicePolicy.LOG
        )
        session[ExamplePayload].append(ExamplePayload(1))
        snapshot = session.snapshot(include_all=True)

        session[ExamplePayload].append(ExamplePayload(2))
        session.restore(snapshot, preserve_logs=False)

        # LOG slices restored when preserve_logs=False
        assert session[ExamplePayload].all() == (ExamplePayload(1),)
