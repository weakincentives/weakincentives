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

"""Tests for Session, SliceAccessor, reducers, and observability events."""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from typing import Any

import pytest

from weakincentives.core.errors import SessionError
from weakincentives.core.session import (
    Append,
    PromptExecuted,
    PromptRendered,
    Replace,
    Session,
    ToolInvoked,
    reducer,
)


@dataclass(frozen=True)
class AddStep:
    step: str


@dataclass(frozen=True)
class ClearAll:
    pass


@dataclass(frozen=True)
class Plan:
    steps: tuple[str, ...] = ()

    @reducer(on=AddStep)
    def add(self, event: AddStep) -> Replace[Plan]:
        return Replace((replace(self, steps=(*self.steps, event.step)),))

    @reducer(on=ClearAll)
    def clear(self, event: ClearAll) -> Replace[Plan]:
        del event
        return Replace(())


@dataclass(frozen=True)
class LogEntry:
    text: str


def _log_reducer(current: tuple[Any, ...], event: AddStep) -> Append[LogEntry]:
    del current
    return Append((LogEntry(text=event.step),))


def test_reducer_decorator_marks_method() -> None:
    assert getattr(Plan.add, "__wink_reducer_event__", None) is AddStep


def test_slice_seed_query_clear() -> None:
    session = Session()
    plan = Plan(steps=("a",))
    session[Plan].seed(plan)
    assert session[Plan].latest() == plan
    assert session[Plan].all() == (plan,)
    session[Plan].clear()
    assert session[Plan].latest() is None
    assert session[Plan].all() == ()


def test_slice_where() -> None:
    session = Session()
    session[Plan].seed(Plan(steps=("a",)))
    session[Plan].append(Plan(steps=("a", "b")))
    long_plans = session[Plan].where(lambda p: len(p.steps) > 1)
    assert len(long_plans) == 1
    assert long_plans[0].steps == ("a", "b")


def test_install_rejects_non_dataclass() -> None:
    class NotADataclass:
        pass

    session = Session()
    with pytest.raises(SessionError, match="frozen dataclass"):
        session.install(NotADataclass)


def test_install_rejects_unfrozen_dataclass() -> None:
    @dataclass
    class Mutable:
        x: int = 0

    session = Session()
    with pytest.raises(SessionError, match="frozen dataclass"):
        session.install(Mutable)


def test_install_rejects_class_without_reducers() -> None:
    @dataclass(frozen=True)
    class Empty:
        pass

    session = Session()
    with pytest.raises(SessionError, match="declares no @reducer"):
        session.install(Empty)


def test_install_rejects_duplicate_reducers() -> None:
    @dataclass(frozen=True)
    class Dup:
        @reducer(on=AddStep)
        def first(self, event: AddStep) -> Replace[Dup]:
            del event
            return Replace((self,))

        @reducer(on=AddStep)
        def second(self, event: AddStep) -> Replace[Dup]:
            del event
            return Replace((self,))

    session = Session()
    with pytest.raises(SessionError, match="duplicate reducers"):
        session.install(Dup)


def test_dispatch_replace_via_method_reducer() -> None:
    session = Session()
    session.install(Plan, initial=Plan)
    session.dispatch(AddStep(step="a"))
    session.dispatch(AddStep(step="b"))
    latest = session[Plan].latest()
    assert latest is not None
    assert latest.steps == ("a", "b")


def test_initial_factory_used_when_empty() -> None:
    session = Session()
    session.install(Plan, initial=lambda: Plan(steps=("seed",)))
    session.dispatch(AddStep(step="a"))
    latest = session[Plan].latest()
    assert latest is not None
    assert latest.steps == ("seed", "a")


def test_no_initial_raises_on_empty() -> None:
    session = Session()
    session.install(Plan)
    with pytest.raises(SessionError, match="no state for Plan"):
        session.dispatch(AddStep(step="a"))


def test_append_op_via_free_reducer() -> None:
    session = Session()
    session.register(AddStep, _log_reducer, slice_type=LogEntry)
    session.dispatch(AddStep(step="hello"))
    session.dispatch(AddStep(step="world"))
    items = session[LogEntry].all()
    assert [entry.text for entry in items] == ["hello", "world"]


def test_event_with_no_reducer_is_noop() -> None:
    session = Session()
    session.dispatch(AddStep(step="ignored"))
    assert session[Plan].all() == ()


def test_clear_via_replace_empty() -> None:
    session = Session()
    session.install(Plan, initial=Plan)
    session.dispatch(AddStep(step="a"))
    session.dispatch(ClearAll())
    assert session[Plan].latest() is None


def test_reset_clears_slices() -> None:
    session = Session()
    session[Plan].seed(Plan(steps=("a",)))
    session.reset()
    assert session[Plan].all() == ()


def test_subscribe_and_publish() -> None:
    captured: list[object] = []

    class Listener:
        def on_event(self, event: object) -> None:
            captured.append(event)

    session = Session()
    session.subscribe(Listener())
    event = PromptRendered(ns="n", key="k", tool_count=1)
    session.publish(event)
    session.publish(ToolInvoked(name="t", success=True, message=""))
    session.publish(PromptExecuted(ns="n", key="k", tool_calls=1))
    assert len(captured) == 3
    assert captured[0] is event


def test_thread_safety_under_concurrent_dispatch() -> None:
    session = Session()
    session.install(Plan, initial=Plan)
    barrier = threading.Barrier(8)

    def worker(index: int) -> None:
        barrier.wait()
        for value in range(50):
            session.dispatch(AddStep(step=f"{index}-{value}"))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    latest = session[Plan].latest()
    assert latest is not None
    assert len(latest.steps) == 8 * 50
