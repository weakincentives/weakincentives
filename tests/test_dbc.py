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

"""Tests for ``weakincentives.dbc``."""

from __future__ import annotations

import pytest

from weakincentives.dbc import (
    ContractError,
    PostconditionError,
    PreconditionError,
    ensure,
    invariant,
    require,
)

# ---------------------------------------------------------------------------
# require
# ---------------------------------------------------------------------------


def test_require_accepts_when_predicate_true() -> None:
    @require(lambda x: x > 0)
    def double(x: int) -> int:
        return x * 2

    assert double(3) == 6


def test_require_rejects_when_predicate_false() -> None:
    @require(lambda x: x > 0, message="x must be positive")
    def double(x: int) -> int:
        return x * 2

    with pytest.raises(PreconditionError, match="x must be positive"):
        double(-1)


def test_require_predicate_can_return_message_tuple() -> None:
    def positive(x: int) -> tuple[bool, str]:
        return x > 0, f"got {x}"

    @require(positive)
    def double(x: int) -> int:
        return x * 2

    with pytest.raises(PreconditionError, match="got -2"):
        double(-2)


def test_require_handles_default_arguments() -> None:
    @require(lambda x, y: x + y < 100)
    def add(x: int, y: int = 5) -> int:
        return x + y

    assert add(10) == 15


# ---------------------------------------------------------------------------
# ensure
# ---------------------------------------------------------------------------


def test_ensure_accepts_when_postcondition_true() -> None:
    @ensure(lambda x, result: result == x * 2)
    def double(x: int) -> int:
        return x * 2

    assert double(3) == 6


def test_ensure_rejects_when_postcondition_false() -> None:
    @ensure(lambda x, result: result > x, message="result must grow")
    def identity(x: int) -> int:
        return x

    with pytest.raises(PostconditionError, match="result must grow"):
        identity(5)


def test_ensure_predicate_can_return_tuple() -> None:
    @ensure(lambda result, **_: (result > 0, f"non-positive: {result}"))
    def make() -> int:
        return -1

    with pytest.raises(PostconditionError, match="non-positive: -1"):
        make()


# ---------------------------------------------------------------------------
# invariant
# ---------------------------------------------------------------------------


def test_invariant_holds_after_public_call() -> None:
    @invariant(lambda self: self.value >= 0, message="must be non-negative")
    class Counter:
        def __init__(self) -> None:
            self.value = 0

        def increment(self) -> None:
            self.value += 1

    counter = Counter()
    counter.increment()
    assert counter.value == 1


def test_invariant_fires_when_violated() -> None:
    @invariant(lambda self: self.value >= 0, message="must be non-negative")
    class Counter:
        def __init__(self) -> None:
            self.value = 0

        def decrement(self) -> None:
            self.value -= 1

    counter = Counter()
    with pytest.raises(ContractError, match="non-negative"):
        counter.decrement()


def test_invariant_does_not_wrap_private_methods() -> None:
    @invariant(lambda self: self.value >= 0)
    class Box:
        def __init__(self) -> None:
            self.value = 1

        def _internal(self) -> None:
            self.value = -1

    box = Box()
    box._internal()
    assert box.value == -1


def test_invariant_skips_non_callable_class_members() -> None:
    @invariant(lambda self: self.x >= 0)
    class WithConstant:
        TAG = "ok"

        def __init__(self) -> None:
            self.x = 0

        def bump(self) -> None:
            self.x += 1

    obj = WithConstant()
    assert obj.TAG == "ok"
    obj.bump()
    assert obj.x == 1


def test_invariant_predicate_can_return_tuple() -> None:
    @invariant(lambda self: (self.value <= 1, f"too big: {self.value}"))
    class Box:
        def __init__(self) -> None:
            self.value = 0

        def grow(self) -> None:
            self.value = 5

    box = Box()
    with pytest.raises(ContractError, match="too big: 5"):
        box.grow()
