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

"""Property-based tests for design-by-contract decorators."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from hypothesis import given, settings, strategies as st

from weakincentives.dbc import (
    dbc_enabled,
    ensure,
    invariant,
    pure,
    require,
)


@pytest.fixture(autouse=True)
def reset_dbc_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Ensure DbC toggles reset between tests."""
    import weakincentives.dbc as dbc_module

    monkeypatch.delenv("WEAKINCENTIVES_DBC", raising=False)
    dbc_module._forced_state = None
    dbc_module.enable_dbc()
    yield
    dbc_module._forced_state = None
    monkeypatch.delenv("WEAKINCENTIVES_DBC", raising=False)


# ============================================================================
# Property Tests: @require Decorator
# ============================================================================


@given(st.integers(min_value=1, max_value=1000))
@settings(max_examples=100)
def test_require_passes_for_valid_positive_ints(value: int) -> None:
    """@require allows all positive integers when predicate checks positive."""

    @require(lambda x: x > 0)
    def square(x: int) -> int:
        return x * x

    result = square(value)
    assert result == value * value


@given(st.integers(max_value=0))
@settings(max_examples=100)
def test_require_rejects_non_positive_ints(value: int) -> None:
    """@require rejects all non-positive integers when predicate checks positive."""

    @require(lambda x: x > 0)
    def square(x: int) -> int:
        return x * x

    with pytest.raises(AssertionError):
        square(value)


@given(st.integers(min_value=0, max_value=100), st.integers(min_value=0, max_value=100))
@settings(max_examples=100)
def test_require_with_multiple_args(a: int, b: int) -> None:
    """@require works with multiple arguments."""

    @require(lambda x, y: x >= 0 and y >= 0)
    def add(x: int, y: int) -> int:
        return x + y

    result = add(a, b)
    assert result == a + b


@given(st.text(min_size=1, max_size=50))
@settings(max_examples=100)
def test_require_with_string_predicate(text: str) -> None:
    """@require works with string length predicates."""

    @require(lambda s: len(s) > 0)
    def upper(s: str) -> str:
        return s.upper()

    result = upper(text)
    assert result == text.upper()


@given(st.lists(st.integers(), min_size=1, max_size=20))
@settings(max_examples=100)
def test_require_with_list_predicate(values: list[int]) -> None:
    """@require works with list non-empty predicates."""

    @require(lambda lst: len(lst) > 0)
    def sum_list(lst: list[int]) -> int:
        return sum(lst)

    result = sum_list(values)
    assert result == sum(values)


@given(st.integers())
@settings(max_examples=50)
def test_require_inactive_allows_invalid(value: int) -> None:
    """@require is bypassed when dbc is disabled."""

    @require(lambda x: x > 1000)  # Predicate that likely fails
    def identity(x: int) -> int:
        return x

    with dbc_enabled(False):
        result = identity(value)
    assert result == value


# ============================================================================
# Property Tests: @ensure Decorator
# ============================================================================


@given(st.integers(min_value=-1000, max_value=1000))
@settings(max_examples=100)
def test_ensure_validates_increment(value: int) -> None:
    """@ensure validates that result is greater than input."""

    @ensure(lambda x, result: result > x)
    def increment(x: int) -> int:
        return x + 1

    result = increment(value)
    assert result == value + 1


@given(st.integers(min_value=0, max_value=100))
@settings(max_examples=100)
def test_ensure_validates_squared_is_gte(value: int) -> None:
    """@ensure validates that squared value is >= original for non-negative."""

    @ensure(lambda x, result: result >= x)
    def square(x: int) -> int:
        return x * x

    result = square(value)
    assert result == value * value


@given(st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_ensure_validates_float_results(value: float) -> None:
    """@ensure works with float return values."""

    @ensure(lambda x, result: result >= x)
    def double(x: float) -> float:
        return x * 2

    result = double(value)
    assert result == value * 2


@given(st.text(max_size=50))
@settings(max_examples=100)
def test_ensure_validates_string_length(text: str) -> None:
    """@ensure can validate string length invariants."""

    @ensure(lambda s, result: len(result) >= len(s))
    def exclaim(s: str) -> str:
        return s + "!"

    result = exclaim(text)
    assert result == text + "!"


@given(st.integers())
@settings(max_examples=50)
def test_ensure_inactive_skips_validation(value: int) -> None:
    """@ensure is bypassed when dbc is disabled."""

    @ensure(lambda x, result: result > 1000)  # Predicate that likely fails
    def identity(x: int) -> int:
        return x

    with dbc_enabled(False):
        result = identity(value)
    assert result == value


# ============================================================================
# Property Tests: @pure Decorator
# ============================================================================


@given(st.lists(st.integers(), max_size=20))
@settings(max_examples=100)
def test_pure_allows_read_only_list_operations(values: list[int]) -> None:
    """@pure allows reading from lists without mutation."""

    @pure
    def sum_list(lst: list[int]) -> int:
        return sum(lst)

    original = list(values)
    result = sum_list(values)
    assert result == sum(original)
    assert values == original  # Unchanged


@given(st.dictionaries(st.text(min_size=1, max_size=10), st.integers(), max_size=10))
@settings(max_examples=100)
def test_pure_allows_read_only_dict_operations(data: dict[str, int]) -> None:
    """@pure allows reading from dicts without mutation."""

    @pure
    def sum_values(d: dict[str, int]) -> int:
        return sum(d.values())

    original = dict(data)
    result = sum_values(data)
    assert result == sum(original.values())
    assert data == original  # Unchanged


@given(st.integers())
@settings(max_examples=100)
def test_pure_allows_pure_computations(value: int) -> None:
    """@pure allows pure mathematical computations."""

    @pure
    def compute(x: int) -> int:
        return x * x + 2 * x + 1

    result = compute(value)
    assert result == value * value + 2 * value + 1


@given(st.text(max_size=50))
@settings(max_examples=100)
def test_pure_allows_string_operations(text: str) -> None:
    """@pure allows string operations that create new strings."""

    @pure
    def process(s: str) -> str:
        return s.strip().upper()

    result = process(text)
    assert result == text.strip().upper()


@given(st.lists(st.integers(), max_size=10))
@settings(max_examples=50)
def test_pure_inactive_allows_mutation(values: list[int]) -> None:
    """@pure is bypassed when dbc is disabled."""

    @pure
    def append_one(lst: list[int]) -> list[int]:
        lst.append(1)
        return lst

    with dbc_enabled(False):
        result = append_one(values)
    assert result[-1] == 1


@given(st.tuples(st.integers(), st.integers(), st.integers()))
@settings(max_examples=100)
def test_pure_with_tuple_args(args: tuple[int, int, int]) -> None:
    """@pure works with tuple arguments (immutable)."""

    @pure
    def sum_tuple(t: tuple[int, int, int]) -> int:
        return t[0] + t[1] + t[2]

    result = sum_tuple(args)
    assert result == sum(args)


# ============================================================================
# Property Tests: @invariant Decorator
# ============================================================================


@given(st.integers(min_value=0, max_value=100))
@settings(max_examples=100)
def test_invariant_holds_for_valid_operations(initial: int) -> None:
    """@invariant holds when operations maintain the invariant."""

    @invariant(lambda self: self.value >= 0)
    class NonNegativeCounter:
        def __init__(self, value: int) -> None:
            self.value = value

        def add(self, amount: int) -> None:
            self.value += amount

    counter = NonNegativeCounter(initial)
    counter.add(10)
    assert counter.value == initial + 10


@given(st.integers(min_value=0, max_value=100), st.integers(min_value=0, max_value=100))
@settings(max_examples=100)
def test_invariant_maintained_across_multiple_ops(init: int, add: int) -> None:
    """@invariant is checked after each method call."""

    @invariant(lambda self: self.value >= 0)
    class Counter:
        def __init__(self, value: int) -> None:
            self.value = value

        def increment(self, amount: int) -> None:
            self.value += amount

        def get(self) -> int:
            return self.value

    counter = Counter(init)
    counter.increment(add)
    result = counter.get()
    assert result == init + add


@given(st.integers(min_value=0, max_value=50))
@settings(max_examples=50)
def test_invariant_inactive_allows_violation(initial: int) -> None:
    """@invariant is bypassed when dbc is disabled."""

    @invariant(lambda self: self.value >= 0)
    class Counter:
        def __init__(self, value: int) -> None:
            self.value = value

        def subtract(self, amount: int) -> None:
            self.value -= amount

    with dbc_enabled(False):
        counter = Counter(initial)
        counter.subtract(initial + 100)  # Would violate invariant
    assert counter.value < 0


@given(st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=10))
@settings(max_examples=50)
def test_invariant_with_list_property(values: list[int]) -> None:
    """@invariant works with list-based invariants."""

    @invariant(lambda self: len(self.items) <= self.max_size)
    class BoundedList:
        def __init__(self, max_size: int) -> None:
            self.items: list[int] = []
            self.max_size = max_size

        def add(self, item: int) -> None:
            if len(self.items) < self.max_size:
                self.items.append(item)

    bounded = BoundedList(max_size=len(values))
    for v in values:
        bounded.add(v)
    assert len(bounded.items) <= len(values)


# ============================================================================
# Property Tests: Decorator Composition
# ============================================================================


@given(st.integers(min_value=1, max_value=100))
@settings(max_examples=50)
def test_require_and_ensure_together(value: int) -> None:
    """@require and @ensure can be composed on the same function."""

    @require(lambda x: x > 0)
    @ensure(lambda x, result: result == x * x)
    def square(x: int) -> int:
        return x * x

    result = square(value)
    assert result == value * value


@given(st.integers(min_value=0, max_value=50))
@settings(max_examples=50)
def test_pure_with_require(value: int) -> None:
    """@pure and @require can be composed."""

    @require(lambda x: x >= 0)
    @pure
    def sqrt_approx(x: int) -> float:
        return x**0.5

    result = sqrt_approx(value)
    assert result >= 0


# ============================================================================
# Property Tests: Edge Cases
# ============================================================================


@given(st.just(0))
@settings(max_examples=5)
def test_require_with_zero(value: int) -> None:
    """@require handles zero correctly."""

    @require(lambda x: x >= 0)
    def identity(x: int) -> int:
        return x

    assert identity(value) == 0


@given(st.just(""))
@settings(max_examples=5)
def test_require_with_empty_string(text: str) -> None:
    """@require handles empty strings correctly when allowed."""

    @require(lambda s: isinstance(s, str))
    def length(s: str) -> int:
        return len(s)

    assert length(text) == 0


@given(st.just([]))
@settings(max_examples=5)
def test_pure_with_empty_list(values: list[int]) -> None:
    """@pure handles empty lists correctly."""

    @pure
    def sum_list(lst: list[int]) -> int:
        return sum(lst)

    assert sum_list(values) == 0


@given(st.just({}))
@settings(max_examples=5)
def test_pure_with_empty_dict(data: dict[str, int]) -> None:
    """@pure handles empty dicts correctly."""

    @pure
    def sum_values(d: dict[str, int]) -> int:
        return sum(d.values())

    assert sum_values(data) == 0


# ============================================================================
# Property Tests: Multiple Predicates
# ============================================================================


@given(st.integers(min_value=1, max_value=99))
@settings(max_examples=100)
def test_require_multiple_predicates(value: int) -> None:
    """@require with multiple predicates validates all."""

    @require(
        lambda x: x > 0,
        lambda x: x < 100,
    )
    def bounded_square(x: int) -> int:
        return x * x

    result = bounded_square(value)
    assert result == value * value


@given(st.integers(min_value=2, max_value=50))
@settings(max_examples=100)
def test_ensure_multiple_predicates(value: int) -> None:
    """@ensure with multiple predicates validates all."""

    @ensure(
        lambda x, result: result >= x,
        lambda x, result: result <= x * x,
    )
    def double_bounded(x: int) -> int:
        return x * 2

    # For x >= 2: x*2 <= x*x (since 2 <= x, so x*2 <= x*x)
    result = double_bounded(value)
    assert result == value * 2


# ============================================================================
# Property Tests: Predicate with Messages
# ============================================================================


@given(st.integers(max_value=0))
@settings(max_examples=50)
def test_require_with_message_on_failure(value: int) -> None:
    """@require includes message in assertion error."""

    @require(lambda x: (x > 0, "value must be positive"))
    def square(x: int) -> int:
        return x * x

    with pytest.raises(AssertionError) as exc:
        square(value)
    assert "value must be positive" in str(exc.value)


@given(st.integers(min_value=0, max_value=100))
@settings(max_examples=50)
def test_ensure_with_message_on_failure(value: int) -> None:
    """@ensure includes message in assertion error."""

    @ensure(lambda x, result: (result > 1000, "result must be > 1000"))
    def identity(x: int) -> int:
        return x

    with pytest.raises(AssertionError) as exc:
        identity(value)
    assert "result must be > 1000" in str(exc.value)
