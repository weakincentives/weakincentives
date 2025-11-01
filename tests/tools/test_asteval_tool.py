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

"""Tests for the asteval-backed evaluation tool."""

from __future__ import annotations

from typing import Protocol, TypeVar, cast

import pytest

from weakincentives.events import InProcessEventBus, ToolInvoked
from weakincentives.prompt import SupportsDataclass
from weakincentives.prompt.tool import Tool, ToolResult
from weakincentives.session import Session, select_latest
from weakincentives.tools import (
    AstevalSection,
    EvalFileRead,
    EvalFileWrite,
    EvalParams,
    EvalResult,
    ToolValidationError,
    VfsPath,
    VfsToolsSection,
    VirtualFileSystem,
    WriteFile,
)

ParamsT = TypeVar("ParamsT", bound=SupportsDataclass)
ResultT = TypeVar("ResultT", bound=SupportsDataclass)


class _ToolSection(Protocol):
    def tools(self) -> tuple[Tool[SupportsDataclass, SupportsDataclass], ...]: ...


def _find_tool(
    section: _ToolSection, name: str
) -> Tool[SupportsDataclass, SupportsDataclass]:
    for tool in section.tools():
        if tool.name == name:
            assert tool.handler is not None
            return tool
    raise AssertionError(f"Tool {name} not found")


def _invoke_tool(
    bus: InProcessEventBus,
    tool: Tool[ParamsT, ResultT],
    params: ParamsT,
) -> ToolResult[ResultT]:
    handler = tool.handler
    assert handler is not None
    result = handler(params)
    event = ToolInvoked(
        prompt_name="test",
        adapter="adapter",
        name=tool.name,
        params=params,
        result=cast(ToolResult[object], result),
    )
    bus.publish(event)
    return result


def _setup_sections() -> tuple[
    Session,
    InProcessEventBus,
    VfsToolsSection,
    Tool[EvalParams, EvalResult],
]:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    vfs_section = VfsToolsSection(session=session)
    section = AstevalSection(session=session)
    tool = cast(Tool[EvalParams, EvalResult], _find_tool(section, "evaluate_python"))
    return session, bus, vfs_section, tool


def test_expression_mode_success() -> None:
    session, bus, _vfs_section, tool = _setup_sections()

    result = _invoke_tool(bus, tool, EvalParams(code="1 + 2"))

    assert result.message == "Evaluation completed successfully."
    payload = result.value
    assert payload.value_repr == "3"
    assert payload.stdout == ""
    assert payload.stderr == ""
    assert payload.writes == ()
    assert payload.reads == ()
    assert payload.globals == {}

    snapshot = select_latest(session, VirtualFileSystem)
    assert snapshot is None or not snapshot.files


def test_statements_mode_reads_and_writes() -> None:
    session, bus, vfs_section, tool = _setup_sections()

    write_tool = cast(
        Tool[WriteFile, WriteFile], _find_tool(vfs_section, "vfs_write_file")
    )
    _invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("docs", "info.txt")), content="sample text"),
    )

    params = EvalParams(
        code=(
            "content = read_text('docs/info.txt')\n"
            "print(content.upper())\n"
            "summary = greeting + ':' + str(len(content))\n"
            "summary"
        ),
        mode="statements",
        globals={"greeting": '"hello"'},
        reads=(EvalFileRead(path=VfsPath(("docs", "info.txt"))),),
        writes=(
            EvalFileWrite(
                path=VfsPath(("output", "summary.txt")),
                content="Report: {greeting}",
                mode="create",
            ),
        ),
    )

    result = _invoke_tool(bus, tool, params)

    payload = result.value
    assert payload.value_repr == "'hello:11'"
    assert payload.stdout == "SAMPLE TEXT\n"
    assert payload.stderr == ""
    assert payload.globals["greeting"] == "hello"
    assert payload.globals["vfs:docs/info.txt"] == "sample text"
    assert payload.writes == (
        EvalFileWrite(
            path=VfsPath(("output", "summary.txt")),
            content="Report: hello",
            mode="create",
        ),
    )

    snapshot = select_latest(session, VirtualFileSystem)
    assert snapshot is not None
    written = {file.path.segments: file.content for file in snapshot.files}
    assert written[("output", "summary.txt")] == "Report: hello"


def test_helper_write_appends() -> None:
    session, bus, _vfs_section, tool = _setup_sections()

    params = EvalParams(
        code="write_text('logs/activity.log', 'started')",
        mode="statements",
    )

    _invoke_tool(bus, tool, params)

    params_append = EvalParams(
        code=("write_text('logs/activity.log', '-continued', mode='append')\n'done'"),
        mode="statements",
    )

    result = _invoke_tool(bus, tool, params_append)

    payload = result.value
    assert payload.value_repr == "'done'"
    assert payload.stderr == ""

    snapshot = select_latest(session, VirtualFileSystem)
    assert snapshot is not None
    files = {file.path.segments: file.content for file in snapshot.files}
    assert files[("logs", "activity.log")] == "started-continued"


def test_invalid_globals_raise() -> None:
    _session, _bus, _vfs_section, tool = _setup_sections()

    with pytest.raises(ToolValidationError):
        tool.handler(  # type: ignore[reportOptionalCall]
            EvalParams(code="0", globals={"bad": "not json"})
        )


def test_timeout_discards_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    session, bus, _vfs_section, tool = _setup_sections()

    def fake_timeout(func: object) -> tuple[bool, object | None, str]:  # noqa: ANN001
        return True, None, "Execution timed out."

    monkeypatch.setattr(
        "weakincentives.tools.asteval._execute_with_timeout", fake_timeout
    )

    params = EvalParams(
        code="1 + 1",
        mode="statements",
        writes=(
            EvalFileWrite(
                path=VfsPath(("tmp", "file.txt")),
                content="should not write",
            ),
        ),
    )

    result = _invoke_tool(bus, tool, params)

    payload = result.value
    assert payload.value_repr is None
    assert payload.stderr == "Execution timed out."
    assert payload.writes == ()

    snapshot = select_latest(session, VirtualFileSystem)
    assert snapshot is None or not snapshot.files
