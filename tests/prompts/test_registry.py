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

"""Tests for the prompt registry helpers."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from weakincentives.prompt import Prompt, registry
from weakincentives.session import Session


@pytest.fixture(autouse=True)
def _clear_registry() -> Iterator[None]:
    registry.clear()
    yield
    registry.clear()


def _factory(*, session: Session) -> Prompt[Any]:
    return Prompt(ns="tests", key="sample")


def test_register_prevents_duplicates() -> None:
    registry.register("tests", "sample", _factory)
    with pytest.raises(ValueError):
        registry.register("tests", "sample", _factory)


def test_unregister_removes_registered_prompt() -> None:
    registry.register("tests", "sample", _factory)
    assert registry.resolve("tests", "sample") is _factory

    registry.unregister("tests", "sample")
    assert registry.resolve("tests", "sample") is None
