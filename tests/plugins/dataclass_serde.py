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
from datetime import UTC, datetime
from typing import Any

import pytest

from weakincentives.serde import dump, parse
from weakincentives.tools import asteval as asteval_tools
from weakincentives.tools import planning as planning_tools
from weakincentives.tools import vfs as vfs_tools

SupportsFactory = Callable[[], object]


@dataclass(slots=True, frozen=True)
class DataclassSerializationCase:
    """Container describing a discovered dataclass and its factory."""

    cls: type[Any]
    factory: SupportsFactory
    origin: str

    def id(self) -> str:
        return f"{self.origin}:{self.cls.__qualname__}"


def _make_plan_step() -> planning_tools.PlanStep:
    return planning_tools.PlanStep(
        step_id="S001",
        title="Draft plan",
        details="Sketch the first milestone",
        status="pending",
        notes=("Initial note",),
    )


def _make_plan() -> planning_tools.Plan:
    return planning_tools.Plan(
        objective="Ship minimal feature",
        status="active",
        steps=(_make_plan_step(),),
    )


def _make_new_plan_step() -> planning_tools.NewPlanStep:
    return planning_tools.NewPlanStep(
        title="Write docs",
        details="Summarise behaviour",
    )


def _make_setup_plan() -> planning_tools.SetupPlan:
    return planning_tools.SetupPlan(
        objective="Prepare launch",
        initial_steps=(_make_new_plan_step(),),
    )


def _make_add_step() -> planning_tools.AddStep:
    return planning_tools.AddStep(steps=(_make_new_plan_step(),))


def _make_update_step() -> planning_tools.UpdateStep:
    return planning_tools.UpdateStep(
        step_id="S001",
        title="Updated title",
        details="Clarify intent",
    )


def _make_mark_step() -> planning_tools.MarkStep:
    return planning_tools.MarkStep(
        step_id="S001",
        status="done",
        note="Marked complete",
    )


def _make_clear_plan() -> planning_tools.ClearPlan:
    return planning_tools.ClearPlan()


def _make_read_plan() -> planning_tools.ReadPlan:
    return planning_tools.ReadPlan()


def _make_vfs_path() -> vfs_tools.VfsPath:
    return vfs_tools.VfsPath(("workspace", "notes.txt"))


def _make_vfs_file() -> vfs_tools.VfsFile:
    now = datetime.now(UTC)
    return vfs_tools.VfsFile(
        path=_make_vfs_path(),
        content=b"Hello, world!",
        encoding="utf-8",
        size_bytes=13,
        version=1,
        created_at=now,
        updated_at=now,
    )


def _make_virtual_file_system() -> vfs_tools.VirtualFileSystem:
    return vfs_tools.VirtualFileSystem(
        root_path="/tmp/weakincentives-vfs", files=(_make_vfs_file(),)
    )


def _make_list_directory() -> vfs_tools.ListDirectory:
    return vfs_tools.ListDirectory(path=_make_vfs_path())


def _make_list_directory_result() -> vfs_tools.ListDirectoryResult:
    return vfs_tools.ListDirectoryResult(
        path=_make_vfs_path(),
        directories=("docs",),
        files=("notes.txt",),
    )


def _make_read_file() -> vfs_tools.ReadFile:
    return vfs_tools.ReadFile(path=_make_vfs_path())


def _make_write_file() -> vfs_tools.WriteFile:
    return vfs_tools.WriteFile(
        path=_make_vfs_path(),
        content=b"import this",
        mode="overwrite",
        encoding="utf-8",
    )


def _make_delete_entry() -> vfs_tools.DeleteEntry:
    return vfs_tools.DeleteEntry(path=_make_vfs_path())


def _make_eval_file_read() -> asteval_tools.EvalFileRead:
    return asteval_tools.EvalFileRead(path=_make_vfs_path())


def _make_eval_file_write() -> asteval_tools.EvalFileWrite:
    return asteval_tools.EvalFileWrite(
        path=_make_vfs_path(),
        content="print('ok')",
        mode="overwrite",
    )


def _make_eval_params() -> asteval_tools.EvalParams:
    return asteval_tools.EvalParams(
        code="1 + 1",
        mode="expr",
        reads=(_make_eval_file_read(),),
        writes=(_make_eval_file_write(),),
    )


def _make_eval_result() -> asteval_tools.EvalResult:
    return asteval_tools.EvalResult(
        value_repr="2",
        stdout="2\n",
        stderr="",
        globals={"result": "2"},
        reads=(_make_eval_file_read(),),
        writes=(_make_eval_file_write(),),
    )


def _discover_planning() -> dict[type[Any], SupportsFactory]:
    return {
        planning_tools.PlanStep: _make_plan_step,
        planning_tools.Plan: _make_plan,
        planning_tools.NewPlanStep: _make_new_plan_step,
        planning_tools.SetupPlan: _make_setup_plan,
        planning_tools.AddStep: _make_add_step,
        planning_tools.UpdateStep: _make_update_step,
        planning_tools.MarkStep: _make_mark_step,
        planning_tools.ClearPlan: _make_clear_plan,
        planning_tools.ReadPlan: _make_read_plan,
    }


def _discover_vfs() -> dict[type[Any], SupportsFactory]:
    return {
        vfs_tools.VfsPath: _make_vfs_path,
        vfs_tools.VfsFile: _make_vfs_file,
        vfs_tools.VirtualFileSystem: _make_virtual_file_system,
        vfs_tools.ListDirectory: _make_list_directory,
        vfs_tools.ListDirectoryResult: _make_list_directory_result,
        vfs_tools.ReadFile: _make_read_file,
        vfs_tools.WriteFile: _make_write_file,
        vfs_tools.DeleteEntry: _make_delete_entry,
    }


def _discover_asteval() -> dict[type[Any], SupportsFactory]:
    return {
        asteval_tools.EvalFileRead: _make_eval_file_read,
        asteval_tools.EvalFileWrite: _make_eval_file_write,
        asteval_tools.EvalParams: _make_eval_params,
        asteval_tools.EvalResult: _make_eval_result,
    }


_DISCOVERY_BUILDERS: tuple[Callable[[], dict[type[Any], SupportsFactory]], ...] = (
    _discover_planning,
    _discover_vfs,
    _discover_asteval,
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
