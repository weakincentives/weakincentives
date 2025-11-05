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

import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

import weakincentives.tools.asteval as asteval_module
from tests.tools.helpers import find_tool, invoke_tool
from weakincentives.adapters.core import SessionProtocol
from weakincentives.events import InProcessEventBus
from weakincentives.prompt.tool import Tool, ToolContext, ToolResult
from weakincentives.session import Session, select_latest
from weakincentives.tools import (
    AstevalSection,
    EvalFileRead,
    EvalFileWrite,
    EvalParams,
    EvalResult,
    HostMount,
    ListDirectory,
    ListDirectoryResult,
    ToolValidationError,
    VfsPath,
    VfsToolsSection,
    VirtualFileSystem,
    WriteFile,
)


def _assert_success_message(result: ToolResult[EvalResult]) -> None:
    assert result.value is not None
    write_count = len(result.value.writes)
    if write_count == 0:
        expected = "Evaluation succeeded without pending file writes."
    else:
        suffix = "s" if write_count != 1 else ""
        expected = (
            f"Evaluation succeeded with {write_count} pending file write{suffix}."
        )
    assert result.message.startswith(expected)


def _assert_failure_message(
    result: ToolResult[EvalResult], *, had_pending_writes: bool
) -> None:
    if had_pending_writes:
        expected = (
            "Evaluation failed; pending file writes were discarded. "
            "Review stderr details in the payload."
        )
    else:
        expected = "Evaluation failed; review stderr details in the payload."
    assert result.message.startswith(expected)


def _setup_sections() -> tuple[
    Session,
    InProcessEventBus,
    VfsToolsSection,
    Tool[EvalParams, EvalResult],
]:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    vfs_section = VfsToolsSection()
    section = AstevalSection()
    tool = cast(Tool[EvalParams, EvalResult], find_tool(section, "evaluate_python"))
    return session, bus, vfs_section, tool


def test_context_requires_session() -> None:
    _session, bus, _vfs_section, tool = _setup_sections()

    handler = tool.handler
    assert handler is not None
    context = ToolContext(
        prompt=None,
        rendered_prompt=None,
        adapter=None,
        session=cast(SessionProtocol, object()),
        event_bus=bus,
    )

    with pytest.raises(ToolValidationError) as captured:
        handler(EvalParams(code="0"), context=context)

    assert "requires ToolContext.session" in str(captured.value)


def test_missing_dependency_instructs_extra_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_import(_module: str) -> object:
        raise ModuleNotFoundError

    monkeypatch.setattr(asteval_module, "import_module", fail_import)

    with pytest.raises(RuntimeError) as captured:
        asteval_module._load_asteval_module()

    assert "weakincentives[asteval]" in str(captured.value)


def test_asteval_section_disables_tool_overrides_by_default() -> None:
    section = AstevalSection()
    tool = cast(Tool[EvalParams, EvalResult], find_tool(section, "evaluate_python"))

    assert section.accepts_overrides is False
    assert tool.accepts_overrides is False


def test_asteval_section_override_flags_opt_in() -> None:
    section = AstevalSection(
        accepts_overrides=True,
    )
    tool = cast(Tool[EvalParams, EvalResult], find_tool(section, "evaluate_python"))

    assert section.accepts_overrides is True
    assert tool.accepts_overrides is True


def test_expression_mode_success() -> None:
    session, bus, _vfs_section, tool = _setup_sections()

    result = invoke_tool(bus, tool, EvalParams(code="1 + 2"), session=session)

    _assert_success_message(result)
    assert result.value is not None
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
        Tool[WriteFile, WriteFile], find_tool(vfs_section, "vfs_write_file")
    )
    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("docs", "info.txt")), content="sample text"),
        session=session,
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

    result = invoke_tool(bus, tool, params, session=session)

    assert result.value is not None
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
    assert written["output", "summary.txt"] == "Report: hello"


def test_helper_write_appends() -> None:
    session, bus, _vfs_section, tool = _setup_sections()

    params = EvalParams(
        code="write_text('logs/activity.log', 'started')",
        mode="statements",
    )

    invoke_tool(bus, tool, params, session=session)

    params_append = EvalParams(
        code=("write_text('logs/activity.log', '-continued', mode='append')\n'done'"),
        mode="statements",
    )

    result = invoke_tool(bus, tool, params_append, session=session)

    assert result.value is not None
    payload = result.value
    assert payload.value_repr == "'done'"
    assert payload.stderr == ""

    snapshot = select_latest(session, VirtualFileSystem)
    assert snapshot is not None
    files = {file.path.segments: file.content for file in snapshot.files}
    assert files["logs", "activity.log"] == "started-continued"


def test_invalid_globals_raise() -> None:
    session, bus, _vfs_section, tool = _setup_sections()

    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            tool,
            EvalParams(code="0", globals={"bad": "not json"}),
            session=session,
        )


def test_timeout_discards_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    session, bus, _vfs_section, tool = _setup_sections()

    def fake_timeout(func: Callable[[], object]) -> tuple[bool, object | None, str]:
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

    result = invoke_tool(bus, tool, params, session=session)

    assert result.value is not None
    payload = result.value
    assert payload.value_repr is None
    assert payload.stderr == "Execution timed out."
    assert payload.writes == ()

    snapshot = select_latest(session, VirtualFileSystem)
    assert snapshot is None or not snapshot.files


def test_stdout_truncation_and_flush() -> None:
    session, bus, _vfs_section, tool = _setup_sections()

    extra = asteval_module._MAX_STREAM_LENGTH + 100
    params = EvalParams(
        code=f"print('A' * {extra}, flush=True)",
        mode="statements",
    )

    result = invoke_tool(bus, tool, params, session=session)
    assert result.value is not None
    payload = result.value
    assert payload.stdout.endswith("...")
    assert len(payload.stdout) == asteval_module._MAX_STREAM_LENGTH


def test_print_invalid_sep_reports_error() -> None:
    session, bus, _vfs_section, tool = _setup_sections()

    params = EvalParams(code="print('value', sep=0)", mode="statements")

    result = invoke_tool(bus, tool, params, session=session)

    assert result.message == "Evaluation failed; review stderr details in the payload."
    assert result.value is not None
    payload = result.value
    assert "sep must be None or a string." in payload.stderr


def test_print_invalid_end_reports_error() -> None:
    session, bus, _vfs_section, tool = _setup_sections()

    params = EvalParams(code="print('value', end=0)", mode="statements")

    result = invoke_tool(bus, tool, params, session=session)

    assert result.message == "Evaluation failed; review stderr details in the payload."
    assert result.value is not None
    payload = result.value
    assert "end must be None or a string." in payload.stderr


def test_expression_mode_requires_expression() -> None:
    session, bus, _vfs_section, tool = _setup_sections()

    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            tool,
            EvalParams(code="value = 1", mode="expr"),
            session=session,
        )


def test_invalid_mode_rejected() -> None:
    session, bus, _vfs_section, tool = _setup_sections()

    params = EvalParams(code="0", mode="invalid")  # type: ignore[arg-type]

    with pytest.raises(ToolValidationError):
        invoke_tool(bus, tool, params, session=session)


def test_invalid_global_names_raise() -> None:
    session, bus, _vfs_section, tool = _setup_sections()

    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            tool,
            EvalParams(code="0", globals={"": "0"}),
            session=session,
        )

    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            tool,
            EvalParams(code="0", globals={"not valid": "0"}),
            session=session,
        )


def test_code_length_validation() -> None:
    session, bus, _vfs_section, tool = _setup_sections()

    long_code = "x" * (asteval_module._MAX_CODE_LENGTH + 1)
    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            tool,
            EvalParams(code=long_code),
            session=session,
        )


def test_control_character_validation() -> None:
    session, bus, _vfs_section, tool = _setup_sections()

    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            tool,
            EvalParams(code="print('ok')\x01"),
            session=session,
        )


def test_normalize_path_skips_blank_segments() -> None:
    normalized = asteval_module._normalize_vfs_path(VfsPath(("docs", " ", "file.txt")))

    assert normalized.segments == ("docs", "file.txt")

    nested = asteval_module._normalize_vfs_path(VfsPath(("logs//", "item.txt")))
    assert nested.segments == ("logs", "item.txt")


@pytest.mark.parametrize(
    "path",
    [
        VfsPath(("/absolute",)),
        VfsPath((".", "file.txt")),
        VfsPath(("..", "file.txt")),
        VfsPath(("a" * (asteval_module._MAX_SEGMENT_LENGTH + 1),)),
        VfsPath(tuple(f"seg{i}" for i in range(asteval_module._MAX_PATH_DEPTH + 1))),
    ],
)
def test_normalize_path_invalid_segments(path: VfsPath) -> None:
    with pytest.raises(ToolValidationError):
        asteval_module._normalize_vfs_path(path)


def test_read_requires_existing_file() -> None:
    session, bus, _vfs_section, tool = _setup_sections()

    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            tool,
            EvalParams(
                code="0",
                reads=(EvalFileRead(path=VfsPath(("docs", "missing.txt"))),),
            ),
            session=session,
        )


def test_duplicate_reads_rejected() -> None:
    session, bus, _vfs_section, tool = _setup_sections()

    read = EvalFileRead(path=VfsPath(("docs", "info.txt")))

    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            tool,
            EvalParams(code="0", reads=(read, read)),
            session=session,
        )


def test_duplicate_writes_rejected() -> None:
    session, bus, _vfs_section, tool = _setup_sections()

    write = EvalFileWrite(path=VfsPath(("output", "data.txt")), content="x")

    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            tool,
            EvalParams(code="0", writes=(write, write)),
            session=session,
        )


def test_read_write_conflict_rejected() -> None:
    session, bus, _vfs_section, tool = _setup_sections()

    path = VfsPath(("docs", "info.txt"))

    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            tool,
            EvalParams(
                code="0",
                reads=(EvalFileRead(path=path),),
                writes=(EvalFileWrite(path=path, content="x"),),
            ),
            session=session,
        )


def test_invalid_write_mode_rejected() -> None:
    session, bus, _vfs_section, tool = _setup_sections()

    write = EvalFileWrite(
        path=VfsPath(("output", "data.txt")),
        content="x",
        mode="invalid",  # type: ignore[arg-type]
    )

    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            tool,
            EvalParams(code="0", writes=(write,)),
            session=session,
        )


def test_write_content_length_validation() -> None:
    session, bus, _vfs_section, tool = _setup_sections()

    long_content = "x" * (asteval_module._MAX_WRITE_LENGTH + 1)
    write = EvalFileWrite(path=VfsPath(("output", "large.txt")), content=long_content)

    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            tool,
            EvalParams(code="0", writes=(write,)),
            session=session,
        )


def test_create_mode_rejects_existing_file() -> None:
    session, bus, vfs_section, tool = _setup_sections()

    write_tool = cast(
        Tool[WriteFile, WriteFile], find_tool(vfs_section, "vfs_write_file")
    )
    existing_path = VfsPath(("docs", "info.txt"))
    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=existing_path, content="original"),
        session=session,
    )

    result = invoke_tool(
        bus,
        tool,
        EvalParams(
            code="0",
            writes=(
                EvalFileWrite(path=existing_path, content="new content", mode="create"),
            ),
        ),
        session=session,
    )

    _assert_success_message(result)
    snapshot = select_latest(session, VirtualFileSystem)
    assert snapshot is not None
    files = {file.path.segments: file.content for file in snapshot.files}
    assert files[existing_path.segments] == "original"


def test_append_requires_existing_file() -> None:
    session, bus, _vfs_section, tool = _setup_sections()

    result = invoke_tool(
        bus,
        tool,
        EvalParams(
            code="0",
            writes=(
                EvalFileWrite(
                    path=VfsPath(("docs", "missing.txt")),
                    content="value",
                    mode="append",
                ),
            ),
        ),
        session=session,
    )

    _assert_success_message(result)
    snapshot = select_latest(session, VirtualFileSystem)
    assert snapshot is None or not snapshot.files


def test_overwrite_updates_existing_file() -> None:
    session, bus, vfs_section, tool = _setup_sections()

    write_tool = cast(
        Tool[WriteFile, WriteFile], find_tool(vfs_section, "vfs_write_file")
    )
    path = VfsPath(("docs", "info.txt"))
    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=path, content="old"),
        session=session,
    )

    result = invoke_tool(
        bus,
        tool,
        EvalParams(
            code="0",
            writes=(EvalFileWrite(path=path, content="updated", mode="overwrite"),),
        ),
        session=session,
    )

    _assert_success_message(result)
    snapshot = select_latest(session, VirtualFileSystem)
    assert snapshot is not None
    files = {file.path.segments: file.content for file in snapshot.files}
    assert files[path.segments] == "updated"


def test_message_summarizes_multiple_writes() -> None:
    session, bus, _vfs_section, tool = _setup_sections()

    writes = tuple(
        EvalFileWrite(path=VfsPath(("output", f"file{i}.txt")), content="x")
        for i in range(4)
    )

    result = invoke_tool(
        bus,
        tool,
        EvalParams(code="0", mode="statements", writes=writes),
        session=session,
    )

    _assert_success_message(result)
    assert (
        "writes=4 file(s): output/file0.txt, output/file1.txt, output/file2.txt, +1 more"
        in result.message
    )
    snapshot = select_latest(session, VirtualFileSystem)
    assert snapshot is not None
    paths = sorted(file.path.segments for file in snapshot.files)
    assert paths == [("output", f"file{i}.txt") for i in range(4)]


def test_read_text_uses_persisted_mount(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    sunfish = root / "sunfish"
    sunfish.mkdir()
    readme = sunfish / "README.md"
    readme.write_text("hello mount", encoding="utf-8")

    bus = InProcessEventBus()
    session = Session(bus=bus)
    vfs_section = VfsToolsSection(
        mounts=(HostMount(host_path="sunfish", mount_path=VfsPath(("sunfish",))),),
        allowed_host_roots=(root,),
    )
    list_tool = cast(
        Tool[ListDirectory, ListDirectoryResult],
        find_tool(vfs_section, "vfs_list_directory"),
    )
    invoke_tool(bus, list_tool, ListDirectory(), session=session)
    section = AstevalSection()
    tool = cast(Tool[EvalParams, EvalResult], find_tool(section, "evaluate_python"))

    result = invoke_tool(
        bus,
        tool,
        EvalParams(code="read_text('sunfish/README.md')", mode="expr"),
        session=session,
    )

    _assert_success_message(result)
    assert result.value is not None
    payload = result.value
    assert payload.value_repr == "'hello mount'"


def test_write_text_rejects_empty_path() -> None:
    session, bus, _vfs_section, tool = _setup_sections()

    result = invoke_tool(
        bus,
        tool,
        EvalParams(code="write_text('', 'data')", mode="statements"),
        session=session,
    )

    _assert_failure_message(result, had_pending_writes=True)
    assert result.value is not None
    assert "Path must be non-empty." in result.value.stderr


def test_globals_formatting_covers_primitives() -> None:
    session, bus, _vfs_section, tool = _setup_sections()

    code = (
        "text = 'hello'\n"
        "number = 7\n"
        "pi = 3.5\n"
        "flag = True\n"
        "empty = None\n"
        "value = {'key': 1}\n"
    )

    result = invoke_tool(
        bus,
        tool,
        EvalParams(code=code, mode="statements"),
        session=session,
    )

    assert result.value is not None
    payload = result.value
    assert payload.globals["text"] == "hello"
    assert payload.globals["number"] == "7"
    assert payload.globals["pi"] == "3.5"
    assert payload.globals["flag"] == "true"
    assert payload.globals["empty"] == "null"
    assert payload.globals["value"].startswith("!repr:")


def test_interpreter_error_surfaces_in_stderr() -> None:
    session, bus, _vfs_section, tool = _setup_sections()

    result = invoke_tool(
        bus,
        tool,
        EvalParams(code="unknown_name + 1", mode="statements"),
        session=session,
    )

    _assert_failure_message(result, had_pending_writes=False)
    assert result.value is not None
    payload = result.value
    assert "unknown_name" in payload.stderr


def test_write_text_conflict_with_read_path() -> None:
    session, bus, vfs_section, tool = _setup_sections()

    write_tool = cast(
        Tool[WriteFile, WriteFile], find_tool(vfs_section, "vfs_write_file")
    )
    path = VfsPath(("docs", "info.txt"))
    invoke_tool(bus, write_tool, WriteFile(path=path, content="seed"), session=session)

    params = EvalParams(
        code="write_text('docs/info.txt', 'data')",
        mode="statements",
        reads=(EvalFileRead(path=path),),
    )

    result = invoke_tool(bus, tool, params, session=session)
    _assert_failure_message(result, had_pending_writes=True)
    assert result.value is not None
    assert (
        "Writes queued during execution must not target read paths."
        in result.value.stderr
    )


def test_write_text_duplicate_targets() -> None:
    session, bus, _vfs_section, tool = _setup_sections()

    params = EvalParams(
        code=("write_text('logs/out.txt', 'a')\nwrite_text('logs/out.txt', 'b')"),
        mode="statements",
    )

    result = invoke_tool(bus, tool, params, session=session)
    _assert_failure_message(result, had_pending_writes=True)
    assert result.value is not None
    assert "Duplicate write targets detected." in result.value.stderr


def test_missing_template_variable_raises() -> None:
    session, bus, _vfs_section, tool = _setup_sections()

    params = EvalParams(
        code="0",
        mode="statements",
        writes=(
            EvalFileWrite(
                path=VfsPath(("output", "report.txt")),
                content="Value: {missing}",
            ),
        ),
    )

    with pytest.raises(ToolValidationError):
        invoke_tool(bus, tool, params, session=session)


def test_duplicate_final_writes_detected() -> None:
    session, bus, _vfs_section, tool = _setup_sections()

    params = EvalParams(
        code="write_text('output/data.txt', 'helper')",
        mode="statements",
        writes=(
            EvalFileWrite(
                path=VfsPath(("output", "data.txt")),
                content="base",
            ),
        ),
    )

    result = invoke_tool(bus, tool, params, session=session)
    _assert_failure_message(result, had_pending_writes=True)
    assert result.value is not None
    assert "Duplicate write targets detected." in result.value.stderr


def test_overwrite_requires_existing_file() -> None:
    session, bus, _vfs_section, tool = _setup_sections()

    params = EvalParams(
        code="0",
        mode="statements",
        writes=(
            EvalFileWrite(
                path=VfsPath(("docs", "missing.txt")),
                content="value",
                mode="overwrite",
            ),
        ),
    )

    result = invoke_tool(bus, tool, params, session=session)
    _assert_success_message(result)
    snapshot = select_latest(session, VirtualFileSystem)
    assert snapshot is None or not snapshot.files


def test_execute_with_timeout_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asteval_module, "_TIMEOUT_SECONDS", 0.01, raising=False)

    def sleeper() -> None:
        time.sleep(0.05)

    timed_out, value, message = asteval_module._execute_with_timeout(sleeper)

    assert timed_out is True
    assert value is None
    assert message == "Execution timed out."


def test_execute_with_timeout_returns_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asteval_module, "_TIMEOUT_SECONDS", 0.01, raising=False)

    timed_out, value, message = asteval_module._execute_with_timeout(lambda: "ok")

    assert timed_out is False
    assert value == "ok"
    assert message == ""


def test_execute_with_timeout_handles_timeout_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raiser() -> None:
        raise TimeoutError

    timed_out, value, message = asteval_module._execute_with_timeout(raiser)

    assert timed_out is False
    assert value is None
    assert message == "Execution timed out."


def test_execute_with_timeout_propagates_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        asteval_module._execute_with_timeout(explode)


def test_merge_globals_combines_mappings() -> None:
    merged = asteval_module._merge_globals({"alpha": 1}, {"beta": 2})
    assert merged == {"alpha": 1, "beta": 2}
