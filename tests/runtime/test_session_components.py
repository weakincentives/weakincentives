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

"""Tests for session internal components (SliceStore, ReducerRegistry)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from weakincentives.runtime.session import (
    Append,
    ReducerContextProtocol,
    ReducerEvent,
    SlicePolicy,
    SliceView,
    append_all,
    default_slice_config,
)
from weakincentives.runtime.session.reducer_registry import ReducerRegistry
from weakincentives.runtime.session.slice_store import SliceStore

if TYPE_CHECKING:
    from tests.conftest import SessionFactory

pytestmark = pytest.mark.core


@dataclass(slots=True, frozen=True)
class SampleEvent:
    value: str


@dataclass(slots=True, frozen=True)
class SampleSlice:
    data: str


def sample_reducer(
    view: SliceView[SampleSlice],
    event: ReducerEvent,
    *,
    context: ReducerContextProtocol,
) -> Append[SampleSlice]:
    del context, view, event
    return Append(SampleSlice("test"))


class TestReducerRegistry:
    """Tests for ReducerRegistry component."""

    def test_all_event_types_returns_registered_types(self) -> None:
        registry = ReducerRegistry()
        registry.register(SampleEvent, append_all, target_slice=SampleEvent)
        registry.register(SampleSlice, sample_reducer, target_slice=SampleSlice)

        event_types = registry.all_event_types()

        assert SampleEvent in event_types
        assert SampleSlice in event_types

    def test_iter_registrations_yields_all_registrations(self) -> None:
        registry = ReducerRegistry()
        registry.register(SampleEvent, append_all, target_slice=SampleEvent)
        registry.register(SampleSlice, sample_reducer, target_slice=SampleSlice)

        registrations = list(registry.iter_registrations())

        assert len(registrations) == 2
        event_types = {r[0] for r in registrations}
        assert event_types == {SampleEvent, SampleSlice}

    def test_snapshot_creates_copy_of_registrations(self) -> None:
        registry = ReducerRegistry()
        registry.register(SampleEvent, append_all, target_slice=SampleEvent)

        snapshot = registry.snapshot()

        assert len(snapshot) == 1
        assert snapshot[0][0] == SampleEvent

    def test_copy_from_copies_registrations(self) -> None:
        source = ReducerRegistry()
        source.register(SampleEvent, append_all, target_slice=SampleEvent)
        snapshot = source.snapshot()

        target = ReducerRegistry()
        target.copy_from(snapshot)

        assert target.has_registrations(SampleEvent)
        assert len(target.get_registrations(SampleEvent)) == 1

    def test_copy_from_skips_existing_by_default(self) -> None:
        source = ReducerRegistry()
        source.register(SampleEvent, append_all, target_slice=SampleEvent)
        snapshot = source.snapshot()

        target = ReducerRegistry()
        target.register(SampleEvent, sample_reducer, target_slice=SampleSlice)
        target.copy_from(snapshot, skip_existing=True)

        # Should still have only the original registration
        regs = target.get_registrations(SampleEvent)
        assert len(regs) == 1
        assert regs[0].slice_type == SampleSlice


class TestSliceStore:
    """Tests for SliceStore component."""

    def test_get_policy_returns_policy_for_type(self) -> None:
        store = SliceStore()
        store.set_policy(SampleSlice, SlicePolicy.LOG)

        policy = store.get_policy(SampleSlice)

        assert policy == SlicePolicy.LOG

    def test_get_policy_returns_state_as_default(self) -> None:
        store = SliceStore()

        policy = store.get_policy(SampleSlice)

        assert policy == SlicePolicy.STATE

    def test_iter_slices_yields_all_slices(self) -> None:
        store = SliceStore()
        _ = store.get_or_create(SampleSlice)
        _ = store.get_or_create(SampleEvent)

        slices = list(store.iter_slices())

        assert len(slices) == 2
        slice_types = {s[0] for s in slices}
        assert slice_types == {SampleSlice, SampleEvent}

    def test_apply_policies_sets_all_policies(self) -> None:
        store = SliceStore()
        policies = {SampleSlice: SlicePolicy.LOG, SampleEvent: SlicePolicy.STATE}

        store.apply_policies(policies)

        assert store.get_policy(SampleSlice) == SlicePolicy.LOG
        assert store.get_policy(SampleEvent) == SlicePolicy.STATE


class TestSessionBackwardCompatibility:
    """Tests for backward compatibility properties on Session."""

    def test_slice_config_returns_store_config(
        self, session_factory: SessionFactory
    ) -> None:
        session, _ = session_factory()

        config = session._slice_config

        assert config == default_slice_config()
