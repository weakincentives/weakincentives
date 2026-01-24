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

"""Tests for the internal design-by-contract helpers."""

from __future__ import annotations

import builtins
import logging
import pathlib
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import pytest

from weakincentives.dbc import (
    dbc_active,
    dbc_suspended,
    ensure,
    invariant,
    pure,
    require,
    skip_invariant,
)
from weakincentives.runtime.session.session import (
    SESSION_ID_BYTE_LENGTH,
    Session,
    _created_at_has_tz,
    _created_at_is_utc,
    _session_id_is_well_formed,
)

pytestmark = pytest.mark.core


@pytest.fixture(autouse=True)
def reset_dbc_state() -> Iterator[None]:
    """Ensure DbC is active for each test (which is the default)."""
    # DbC is always enabled by default via ContextVar, no setup needed.
    # Each test runs in a fresh context.
    yield


def test_require_allows_valid_inputs() -> None:
    @require(lambda value: value > 0)
    def square(value: int) -> int:
        return value * value

    assert square(4) == 16


def test_require_rejects_invalid_inputs() -> None:
    @require(lambda value: (value > 0, "value must be positive"))
    def cube(value: int) -> int:
        return value**3

    with pytest.raises(AssertionError) as exc:
        cube(-1)

    assert "value must be positive" in str(exc.value)


def test_ensure_validates_return_values() -> None:
    @ensure(lambda value, result: result >= value)
    def increment(value: int) -> int:
        return value + 1

    assert increment(2) == 3


def test_ensure_validates_exceptions() -> None:
    @ensure(lambda _, exception: isinstance(exception, ValueError))
    def fail(value: int) -> int:
        raise ValueError(f"invalid: {value}")

    with pytest.raises(ValueError):
        fail(5)


def test_invariant_enforces_state_between_calls() -> None:
    @invariant(lambda self: self.balance >= 0)
    class Counter:
        def __init__(self) -> None:
            self.balance = 0

        def deposit(self, amount: int) -> None:
            self.balance += amount

        def withdraw(self, amount: int) -> None:
            self.balance -= amount

    counter = Counter()
    counter.deposit(3)
    with pytest.raises(AssertionError):
        counter.withdraw(4)


def test_invariant_can_be_skipped_for_helpers() -> None:
    @invariant(lambda self: not self._buffer)
    class Buffer:
        def __init__(self) -> None:
            self._buffer: list[int] = []

        @skip_invariant
        def reset(self) -> None:
            self._buffer.clear()

        def append(self, value: int) -> None:
            self._buffer.append(value)

    buf = Buffer()
    buf.reset()
    with pytest.raises(AssertionError):
        buf.append(1)


def test_invariant_is_inert_when_suspended() -> None:
    @invariant(lambda self: self.balance >= 0)
    class Counter:
        def __init__(self) -> None:
            self.balance = 0

        def withdraw(self, amount: int) -> None:
            self.balance -= amount

    counter = Counter()
    with dbc_suspended():
        counter.withdraw(5)
    assert counter.balance == -5


def test_invariant_init_skipped_when_suspended() -> None:
    @invariant(lambda self: self.balance >= 0)
    class Counter:
        def __init__(self, initial: int) -> None:
            self.balance = initial

    # Instantiate when dbc is suspended - should skip invariant check
    with dbc_suspended():
        counter = Counter(-10)
    assert counter.balance == -10


def test_pure_detects_mutation() -> None:
    @pure
    def mutate(values: list[int]) -> list[int]:
        values.append(1)
        return values

    with pytest.raises(AssertionError):
        mutate([])


def test_pure_detects_io_calls(tmp_path: pathlib.Path) -> None:
    @pure
    def touch_file(path: pathlib.Path) -> None:
        path.write_text("hello")

    with pytest.raises(AssertionError):
        touch_file(tmp_path / "sample.txt")


def test_pure_detects_open_calls(tmp_path: pathlib.Path) -> None:
    @pure
    def touch_file(path: pathlib.Path) -> None:
        with builtins.open(path, "w", encoding="utf-8") as handle:  # noqa: PTH123
            handle.write("hello")

    with pytest.raises(AssertionError):
        touch_file(tmp_path / "sample.txt")


def test_pure_detects_write_bytes_calls(tmp_path: pathlib.Path) -> None:
    @pure
    def touch_file(path: pathlib.Path) -> None:
        path.write_bytes(b"hello")

    with pytest.raises(AssertionError):
        touch_file(tmp_path / "sample.bin")


def test_pure_patching_is_thread_safe(tmp_path: pathlib.Path) -> None:
    started = threading.Event()
    done = threading.Event()

    @pure
    def wait_for_done_signal() -> None:
        started.set()
        assert done.wait(timeout=1.0)

    thread = threading.Thread(target=wait_for_done_signal)
    thread.start()
    assert started.wait(timeout=1.0)

    with builtins.open(tmp_path / "allowed.txt", "w", encoding="utf-8") as handle:  # noqa: PTH123
        handle.write("ok")
    (tmp_path / "allowed-pathlib.txt").write_text("ok")
    (tmp_path / "allowed.bin").write_bytes(b"ok")

    logger = logging.getLogger("weakincentives.dbc.thread-safety")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    logger.warning("message while pure patches are active")

    done.set()
    thread.join(timeout=1.0)
    assert not thread.is_alive()


def test_pure_patch_depth_is_thread_safe() -> None:
    started_one = threading.Event()
    started_two = threading.Event()
    done = threading.Event()

    @pure
    def wait_one() -> None:
        started_one.set()
        assert done.wait(timeout=1.0)

    @pure
    def wait_two() -> None:
        started_two.set()
        assert done.wait(timeout=1.0)

    thread_one = threading.Thread(target=wait_one)
    thread_two = threading.Thread(target=wait_two)
    thread_one.start()
    thread_two.start()
    assert started_one.wait(timeout=1.0)
    assert started_two.wait(timeout=1.0)

    done.set()
    thread_one.join(timeout=1.0)
    thread_two.join(timeout=1.0)
    assert not thread_one.is_alive()
    assert not thread_two.is_alive()


def test_pure_is_inert_when_suspended() -> None:
    assert dbc_active() is True

    @pure
    def mutate(values: list[int]) -> list[int]:
        values.append(1)
        return values

    with dbc_suspended():
        assert mutate([]) == [1]

    with pytest.raises(AssertionError):
        mutate([])


def test_dbc_is_always_active_by_default() -> None:
    """DbC is always enabled and cannot be globally disabled."""
    import weakincentives.dbc as dbc_module

    # DbC is always active by default
    assert dbc_module.dbc_active() is True

    # Can be temporarily suspended via context manager
    with dbc_module.dbc_suspended():
        assert dbc_module.dbc_active() is False

    # Automatically restored after context exits
    assert dbc_module.dbc_active() is True


def test_dbc_suspended_is_nestable() -> None:
    """dbc_suspended() can be nested and restores correctly."""
    import weakincentives.dbc as dbc_module

    assert dbc_module.dbc_active() is True

    with dbc_module.dbc_suspended():
        assert dbc_module.dbc_active() is False
        with dbc_module.dbc_suspended():
            assert dbc_module.dbc_active() is False
        assert dbc_module.dbc_active() is False

    assert dbc_module.dbc_active() is True


def test_require_raises_without_predicates() -> None:
    with pytest.raises(ValueError):
        require()


def test_require_predicate_exception() -> None:
    def boom(_: int) -> bool:
        raise RuntimeError("boom")

    @require(boom)
    def identity(value: int) -> int:
        return value

    with pytest.raises(AssertionError) as exc:
        identity(1)

    assert "RuntimeError" in str(exc.value)


def test_require_predicate_empty_tuple() -> None:
    @require(lambda _: ())
    def identity(value: int) -> int:
        return value

    with pytest.raises(TypeError):
        identity(10)


def test_require_coerces_non_boolean_predicate_results() -> None:
    @require(lambda _: "truthy")
    def identity(value: int) -> int:
        return value

    assert identity(4) == 4


def test_ensure_raises_without_predicates() -> None:
    with pytest.raises(ValueError):
        ensure()


def test_ensure_skips_when_suspended() -> None:
    @ensure(lambda value, result: result > value)
    def bump(value: int) -> int:
        return value + 1

    with dbc_suspended():
        assert bump(4) == 5


def test_invariant_requires_predicates() -> None:
    with pytest.raises(ValueError):
        invariant()


def test_invariant_skips_static_and_class_methods() -> None:
    tracker: list[str] = []

    @invariant(lambda self: True)
    class Example:
        def __init__(self) -> None:
            tracker.append("init")

        @staticmethod
        def helper() -> str:
            return "static"

        @classmethod
        def build(cls) -> Example:
            tracker.append("build")
            return cls()

        @property
        def constant(self) -> int:
            return 7

        def ping(self) -> str:
            tracker.append("ping")
            _ = self.constant
            return "pong"

    example = Example()
    assert example.helper() == "static"
    assert Example.build().constant == 7
    assert example.ping() == "pong"
    assert tracker.count("init") >= 1
    assert "ping" in tracker


def test_pure_detects_keyword_mutation() -> None:
    @pure
    def mutate_keyword(*, data: dict[str, int]) -> None:
        data["value"] += 1

    with pytest.raises(AssertionError):
        mutate_keyword(data={"value": 1})


def test_pure_handles_uncopyable_arguments() -> None:
    class NoCopy:
        def __deepcopy__(self, memo: object) -> NoCopy:
            raise TypeError("no-copy")

    @pure
    def use(value: NoCopy) -> int:
        return id(value)

    instance = NoCopy()
    assert isinstance(use(instance), int)


def test_pure_handles_uncopyable_keyword_arguments() -> None:
    class NoCopy:
        def __deepcopy__(self, memo: object) -> NoCopy:
            raise TypeError("no-copy")

    @pure
    def use(*, value: NoCopy) -> int:
        return id(value)

    instance = NoCopy()
    assert isinstance(use(value=instance), int)


def test_require_predicate_returning_none() -> None:
    @require(lambda _: None)
    def identity(value: int) -> int:
        return value

    with pytest.raises(AssertionError):
        identity(1)


def test_require_preserves_assertion_errors() -> None:
    def predicate(_: int) -> bool:
        raise AssertionError("boom")

    @require(predicate)
    def identity(value: int) -> int:
        return value

    with pytest.raises(AssertionError) as exc:
        identity(1)

    assert "boom" in str(exc.value)


def test_session_invariant_helpers_cover_basics() -> None:
    session = SimpleNamespace(
        session_id=uuid4(),
        created_at=datetime.now(UTC),
    )

    typed_session = cast(Session, session)

    assert _session_id_is_well_formed(typed_session)
    assert len(typed_session.session_id.bytes) == SESSION_ID_BYTE_LENGTH
    assert _created_at_has_tz(typed_session)
    assert _created_at_is_utc(typed_session)


def test_require_skips_when_suspended() -> None:
    @require(lambda value: value > 0)
    def square(value: int) -> int:
        return value * value

    with dbc_suspended():
        assert square(-1) == 1


def test_pure_handles_equal_keyword_arguments() -> None:
    @pure
    def read_dict(*, data: dict[str, int]) -> int:
        return data["value"]

    assert read_dict(data={"value": 42}) == 42
