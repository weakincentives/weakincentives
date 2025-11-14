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

"""Pytest plugin auditing dataclass contracts for built-in tools."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Iterable
from dataclasses import MISSING, dataclass, fields, is_dataclass
from importlib import import_module
from typing import Annotated, Any, get_args, get_origin, get_type_hints

import pytest


@dataclass(slots=True, frozen=True)
class ToolDataclassContractCase:
    """Describe a dataclass that should satisfy the tool contract."""

    cls: type[object]
    origin: str

    def id(self) -> str:
        return f"{self.origin}:{self.cls.__qualname__}"


_MODULES_TO_AUDIT: tuple[str, ...] = (
    "weakincentives.tools.planning",
    "weakincentives.tools.vfs",
    "weakincentives.tools.asteval",
    "weakincentives.tools.subagents",
)


def _iter_dataclasses(module_name: str) -> Iterable[type[object]]:
    module = import_module(module_name)
    for _, candidate in inspect.getmembers(module, inspect.isclass):
        if candidate.__module__ != module_name:
            continue
        if not is_dataclass(candidate):
            continue
        yield candidate


@functools.lru_cache(maxsize=1)
def _collect_tool_dataclasses() -> tuple[ToolDataclassContractCase, ...]:
    cases: list[ToolDataclassContractCase] = []
    for module_name in _MODULES_TO_AUDIT:
        for cls in _iter_dataclasses(module_name):
            case = ToolDataclassContractCase(cls=cls, origin=module_name)
            cases.append(case)
    cases.sort(key=lambda case: case.id())
    return tuple(cases)


def _has_skip_marker(cls: type[object]) -> bool:
    marks = getattr(cls, "pytestmark", ())
    if not isinstance(marks, Iterable):
        marks = (marks,)
    return any(getattr(mark, "name", None) == "skip_tool_contracts" for mark in marks)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parameterise fixtures that request tool dataclass cases."""

    if "tool_dataclass_case" not in metafunc.fixturenames:
        return
    cases = _collect_tool_dataclasses()
    metafunc.parametrize(
        "tool_dataclass_case",
        cases,
        ids=[case.id() for case in cases],
    )


@pytest.hookimpl
def pytest_configure(config: pytest.Config) -> None:
    """Register tool contract markers."""

    config.addinivalue_line(
        "markers",
        "skip_tool_contracts: opt out of automatic tool dataclass contract checks.",
    )


@pytest.fixture
def tool_dataclass_case(request: pytest.FixtureRequest) -> ToolDataclassContractCase:
    """Fixture returning the current tool dataclass contract case."""

    return request.param


def _unwrap_annotated(annotation: object) -> object:
    origin = get_origin(annotation)
    if origin is None:
        return annotation
    if origin is Annotated:
        base, *_ = get_args(annotation)
        return _unwrap_annotated(base)
    return annotation


def _contains_any(annotation: object) -> bool:
    annotation = _unwrap_annotated(annotation)
    if annotation is Any:
        return True
    origin = get_origin(annotation)
    if origin is None:
        return False
    return any(_contains_any(arg) for arg in get_args(annotation))


def _is_tuple_annotation(annotation: object) -> bool:
    annotation = _unwrap_annotated(annotation)
    origin = get_origin(annotation)
    if origin is None:
        return annotation is tuple
    return origin is tuple


def assert_tool_dataclass_contract(case: ToolDataclassContractCase) -> None:
    """Validate invariants that tool-facing dataclasses must satisfy."""

    cls = case.cls
    if _has_skip_marker(cls):
        pytest.skip(f"{cls.__qualname__} opts out of tool dataclass contracts")

    params = getattr(cls, "__dataclass_params__", None)
    assert params is not None, f"{cls.__qualname__} is not a dataclass"
    assert params.slots, f"{cls.__qualname__} must enable dataclass slots"

    type_hints = get_type_hints(cls, include_extras=True)

    for field in fields(cls):
        metadata = field.metadata
        description = metadata.get("description")
        assert isinstance(description, str), (
            f"{cls.__qualname__}.{field.name} must document its description"
        )
        assert description.strip(), (
            f"{cls.__qualname__}.{field.name} description must not be empty"
        )

        annotation = type_hints.get(field.name, field.type)
        assert not _contains_any(annotation), (
            f"{cls.__qualname__}.{field.name} must not use typing.Any"
        )

        if _is_tuple_annotation(annotation):
            if field.default is not MISSING:
                assert isinstance(field.default, tuple), (
                    f"{cls.__qualname__}.{field.name} default must be a tuple"
                )
            if field.default_factory is not MISSING:
                produced = field.default_factory()
                assert isinstance(produced, tuple), (
                    f"{cls.__qualname__}.{field.name} default_factory must produce a tuple"
                )


__all__ = [
    "ToolDataclassContractCase",
    "assert_tool_dataclass_contract",
]
