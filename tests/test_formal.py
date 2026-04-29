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

"""Tests for ``weakincentives.formal``."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from weakincentives.formal import (
    FormalError,
    FormalSpec,
    all_formal_specs,
    formal_spec,
    get_formal_spec,
    reset_registry,
)


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    reset_registry()
    yield
    reset_registry()


def test_formal_spec_attaches_metadata_to_function() -> None:
    @formal_spec(name="add", properties=("commutative",))
    def add(a: int, b: int) -> int:
        """Sum two integers."""
        return a + b

    spec = get_formal_spec(add)
    assert spec is not None
    assert spec.name == "add"
    assert spec.properties == ("commutative",)
    assert spec.description == "Sum two integers."


def test_formal_spec_uses_explicit_description() -> None:
    @formal_spec(name="x", description="explicit doc")
    def fn() -> None:
        """ignored docstring"""

    spec = get_formal_spec(fn)
    assert spec is not None
    assert spec.description == "explicit doc"


def test_formal_spec_attaches_to_class() -> None:
    @formal_spec(name="counter")
    class Counter:
        """A simple counter."""

    assert isinstance(get_formal_spec(Counter), FormalSpec)


def test_formal_spec_rejects_non_callable() -> None:
    decorator = formal_spec(name="bad")
    with pytest.raises(FormalError, match="callable or class"):
        decorator(42)


def test_formal_spec_rejects_duplicate_names() -> None:
    @formal_spec(name="dup")
    def first() -> None: ...

    with pytest.raises(FormalError, match="duplicate"):

        @formal_spec(name="dup")
        def second() -> None: ...


def test_get_formal_spec_returns_none_for_unannotated() -> None:
    def fn() -> None: ...

    assert get_formal_spec(fn) is None


def test_get_formal_spec_ignores_non_formal_attribute() -> None:
    class Carrier:
        __wink_formal_spec__ = "not a FormalSpec"

    assert get_formal_spec(Carrier()) is None


def test_all_formal_specs_returns_registry() -> None:
    @formal_spec(name="alpha")
    def alpha() -> None: ...

    @formal_spec(name="beta")
    def beta() -> None: ...

    names = sorted(spec.name for spec in all_formal_specs())
    assert names == ["alpha", "beta"]
