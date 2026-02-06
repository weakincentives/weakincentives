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

"""Pytest plugin auditing dataclass round-trips through ``weakincentives.serde``.

The discovery logic mirrors the runtime components that feed session snapshots.
When new packages introduce dataclasses that participate in reducer pipelines or
emit tool results, extend :data:`_DISCOVERY_BUILDERS` with a new callable that
returns factories for each type. Prefer composing the new factories from the
existing helpers rather than hard-coding inline payloads so nested dataclasses
continue to exercise realistic structures.

* Add a new ``_discover_<area>`` helper next to the existing builders.
* Import the production module, assemble per-type factories, and append the
  helper to :data:`_DISCOVERY_BUILDERS`.
* Provide a small factory function for each dataclass so tests can construct
  standalone instances without bespoke fixtures. Skip coverage for complex cases
  by decorating the dataclass with ``@pytest.mark.skip_serialization``.

Keeping the list current ensures the audit continues to reflect the dataclasses
that session snapshots persist.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable
from dataclasses import dataclass, is_dataclass
from typing import Any

import pytest

from weakincentives.contrib.tools import digests as digests_tools
from weakincentives.serde import dump, parse

SupportsFactory = Callable[[], object]


@dataclass(slots=True, frozen=True)
class DataclassSerializationCase:
    """Container describing a discovered dataclass and its factory."""

    cls: type[Any]
    factory: SupportsFactory
    origin: str

    def id(self) -> str:
        return f"{self.origin}:{self.cls.__qualname__}"


def _make_workspace_digest() -> digests_tools.WorkspaceDigest:
    return digests_tools.WorkspaceDigest(
        section_key="workspace-digest",
        summary="Python web application with FastAPI.",
        body="Full workspace analysis including build commands and dependencies.",
    )


def _discover_digests() -> dict[type[Any], SupportsFactory]:
    return {
        digests_tools.WorkspaceDigest: _make_workspace_digest,
    }


_DISCOVERY_BUILDERS: tuple[Callable[[], dict[type[Any], SupportsFactory]], ...] = (
    _discover_digests,
)


@functools.lru_cache(maxsize=1)
def _collect_cases() -> tuple[DataclassSerializationCase, ...]:
    discovered: dict[type[Any], DataclassSerializationCase] = {}
    for builder in _DISCOVERY_BUILDERS:
        factories = builder()
        for cls, factory in factories.items():
            if not isinstance(cls, type) or not is_dataclass(cls):
                continue
            case = DataclassSerializationCase(
                cls=cls,
                factory=factory,
                origin=cls.__module__,
            )
            discovered.setdefault(cls, case)
    ordered = sorted(discovered.values(), key=lambda case: case.id())
    return tuple(ordered)


def _has_skip_marker(cls: type[Any]) -> bool:
    marks = getattr(cls, "pytestmark", ())
    if not isinstance(marks, Iterable):
        marks = (marks,)
    return any(getattr(mark, "name", None) == "skip_serialization" for mark in marks)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "dataclass_case" not in metafunc.fixturenames:
        return
    cases = _collect_cases()
    metafunc.parametrize(
        "dataclass_case",
        cases,
        ids=[case.id() for case in cases],
    )


@pytest.hookimpl
def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "skip_serialization: Skip automatic dataclass serialization coverage.",
    )


@pytest.fixture
def dataclass_case(request: pytest.FixtureRequest) -> DataclassSerializationCase:
    return request.param


def assert_serialization_round_trip(
    dataclass_case: DataclassSerializationCase,
) -> None:
    cls = dataclass_case.cls
    if _has_skip_marker(cls):
        pytest.skip(f"{cls.__qualname__} opts out of serialization audit")

    instance = dataclass_case.factory()
    assert isinstance(instance, cls)

    payload = dump(instance)
    restored = parse(cls, payload)
    assert restored == instance


__all__ = [
    "DataclassSerializationCase",
    "assert_serialization_round_trip",
]
