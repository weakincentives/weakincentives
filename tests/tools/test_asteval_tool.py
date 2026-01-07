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

import weakincentives.contrib.tools.asteval as asteval_module
import weakincentives.contrib.tools.vfs_types as vfs_types_module
from tests.tools.helpers import find_tool, invoke_tool
from weakincentives import ToolValidationError
from weakincentives.contrib.tools import (
    READ_ENTIRE_FILE,
    AstevalSection,
    EvalFileRead,
    EvalFileWrite,
    EvalParams,
    EvalResult,
    HostMount,
    InMemoryFilesystem,
    ListDirectory,
    ListDirectoryResult,
    VfsPath,
    VfsToolsSection,
    WriteFile,
    WriteFileParams,
)
from weakincentives.prompt.tool import Tool, ToolResult
from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.session import Session


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
    VfsToolsSection,
    Tool[EvalParams, EvalResult],
]:
    bus = InProcessDispatcher()
    session = Session(bus=bus)
    vfs_section = VfsToolsSection(session=session)
    section = AstevalSection(session=session, filesystem=vfs_section.filesystem)
    tool = cast(Tool[EvalParams, EvalResult], find_tool(section, "evaluate_python"))
    return session, vfs_section, tool


def _get_filesystem(section: VfsToolsSection) -> InMemoryFilesystem:
    """Get the InMemoryFilesystem from a section."""
    fs = section.filesystem
    assert isinstance(fs, InMemoryFilesystem)
    return fs


def _get_files_dict(section: VfsToolsSection) -> dict[tuple[str, ...], str]:
    """Get all files as a dict of path segments to content."""
    fs = _get_filesystem(section)
    result: dict[tuple[str, ...], str] = {}
    for path, file in fs._files.items():
        segments = tuple(path.split("/")) if path else ()
        result[segments] = file.content.decode("utf-8")
    return result


def test_eval_file_render_helpers() -> None:
    read = EvalFileRead(path=VfsPath(("src", "app.py")))
    write = EvalFileWrite(
        path=VfsPath(("src", "out.py")), content="print()", mode="overwrite"
    )

    assert "src/app.py" in read.render()
    assert "overwrite" in write.render()


def test_eval_result_render_summarizes_payload() -> None:
    read = EvalFileRead(path=VfsPath(("src", "input.py")))
    write = EvalFileWrite(
        path=VfsPath(("src", "out.py")), content="print()", mode="create"
    )
    result = EvalResult(
        value_repr="42",
        stdout="output",
        stderr="",
        globals={"x": "1"},
        reads=(read,),
        writes=(write,),
    )

    rendered = result.render()
    assert "42" in rendered
    assert "Writes" in rendered


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
    bus = InProcessDispatcher()
    session = Session(bus=bus)
    section = AstevalSection(session=session)
    tool = cast(Tool[EvalParams, EvalResult], find_tool(section, "evaluate_python"))

    assert section.accepts_overrides is False
    assert tool.accepts_overrides is False


def test_asteval_section_override_flags_opt_in() -> None:
    bus = InProcessDispatcher()
    session = Session(bus=bus)
    section = AstevalSection(
        session=session,
        accepts_overrides=True,
    )
    tool = cast(Tool[EvalParams, EvalResult], find_tool(section, "evaluate_python"))

    assert section.accepts_overrides is True
    assert tool.accepts_overrides is True


def test_expression_success() -> None:
    session, vfs_section, tool = _setup_sections()

    result = invoke_tool(
        tool,
        EvalParams(code="1 + 2"),
        session=session,
        filesystem=vfs_section.filesystem,
    )

    _assert_success_message(result)
    assert result.value is not None
    payload = result.value
    assert payload.value_repr == "3"
    assert payload.stdout == ""
    assert payload.stderr == ""
    assert payload.writes == ()
    assert payload.reads == ()
    assert payload.globals == {}

    fs = _get_filesystem(vfs_section)
    assert not fs._files


def test_success_without_writes_message_is_simple() -> None:
    session, vfs_section, tool = _setup_sections()

    result = invoke_tool(
        tool, EvalParams(code="2"), session=session, filesystem=vfs_section.filesystem
    )

    assert result.message == "Evaluation succeeded without pending file writes."
    assert result.value is not None
    assert result.value.writes == ()


def test_multiline_reads_and_writes() -> None:
    session, vfs_section, tool = _setup_sections()

    write_tool = cast(
        Tool[WriteFileParams, WriteFile], find_tool(vfs_section, "write_file")
    )
    invoke_tool(
        write_tool,
        WriteFileParams(file_path="docs/info.txt", content="sample text"),
        session=session,
        filesystem=vfs_section.filesystem,
    )

    params = EvalParams(
        code=(
            "content = read_text('docs/info.txt')\n"
            "print(content.upper())\n"
            "summary = greeting + ':' + str(len(content))\n"
            "summary"
        ),
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

    result = invoke_tool(
        tool, params, session=session, filesystem=vfs_section.filesystem
    )

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

    written = _get_files_dict(vfs_section)
    assert written["output", "summary.txt"] == "Report: hello"


def test_helper_write_appends() -> None:
    session, vfs_section, tool = _setup_sections()

    params = EvalParams(
        code="write_text('logs/activity.log', 'started')",
    )

    invoke_tool(tool, params, session=session, filesystem=vfs_section.filesystem)

    params_append = EvalParams(
        code=("write_text('logs/activity.log', '-continued', mode='append')\n'done'"),
    )

    result = invoke_tool(
        tool, params_append, session=session, filesystem=vfs_section.filesystem
    )

    assert result.value is not None
    payload = result.value
    assert payload.value_repr == "'done'"
    assert payload.stderr == ""

    files = _get_files_dict(vfs_section)
    assert files["logs", "activity.log"] == "started-continued"


def test_helper_write_append_preserves_content_beyond_default_read_limit() -> None:
    """Test that append mode doesn't truncate files with more than 2000 lines."""
    session, vfs_section, tool = _setup_sections()
    fs = vfs_section.filesystem

    # Create a large file with more than 2000 lines
    lines = [f"line{i}" for i in range(2500)]
    initial_content = "\n".join(lines)
    fs.write("big.txt", initial_content, mode="create")

    # Append to it
    params = EvalParams(
        code="write_text('big.txt', '\\nAPPENDED', mode='append')\n'done'",
    )
    result = invoke_tool(
        tool, params, session=session, filesystem=vfs_section.filesystem
    )

    assert result.value is not None
    assert result.value.value_repr == "'done'"

    # Verify the file wasn't truncated
    read_result = fs.read("big.txt", limit=READ_ENTIRE_FILE)
    assert "line0" in read_result.content
    assert "line2499" in read_result.content  # Last original line
    assert "APPENDED" in read_result.content  # The appended content


def test_helper_write_success_reports_pending_writes() -> None:
    session, vfs_section, tool = _setup_sections()

    result = invoke_tool(
        tool,
        EvalParams(code="write_text('logs/activity.log', 'started')\n'done'"),
        session=session,
        filesystem=vfs_section.filesystem,
    )

    assert result.message.startswith("Evaluation succeeded with 1 pending file write.")
    assert result.value is not None
    assert result.value.writes == (
        EvalFileWrite(
            path=VfsPath(("logs", "activity.log")),
            content="started",
            mode="create",
        ),
    )


def test_invalid_globals_raise() -> None:
    session, vfs_section, tool = _setup_sections()

    with pytest.raises(ToolValidationError):
        invoke_tool(
            tool,
            EvalParams(code="0", globals={"bad": "not json"}),
            session=session,
            filesystem=vfs_section.filesystem,
        )


def test_timeout_discards_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    session, vfs_section, tool = _setup_sections()

    def fake_timeout(func: Callable[[], object]) -> tuple[bool, object | None, str]:
        return True, None, "Execution timed out."

    monkeypatch.setattr(
        "weakincentives.contrib.tools.asteval._execute_with_timeout", fake_timeout
    )

    params = EvalParams(
        code="1 + 1",
        writes=(
            EvalFileWrite(
                path=VfsPath(("tmp", "file.txt")),
                content="should not write",
            ),
        ),
    )

    result = invoke_tool(
        tool, params, session=session, filesystem=vfs_section.filesystem
    )

    assert result.value is not None
    payload = result.value
    assert payload.value_repr is None
    assert payload.stderr == "Execution timed out."
    assert payload.writes == ()

    fs = _get_filesystem(vfs_section)
    assert not fs._files


def test_timeout_reports_discarded_writes_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, vfs_section, tool = _setup_sections()

    def fake_timeout(func: Callable[[], object]) -> tuple[bool, object | None, str]:
        _ = func
        return True, None, "Execution timed out."

    monkeypatch.setattr(
        "weakincentives.contrib.tools.asteval._execute_with_timeout", fake_timeout
    )

    result = invoke_tool(
        tool,
        EvalParams(
            code="1 + 1",
            writes=(
                EvalFileWrite(
                    path=VfsPath(("tmp", "file.txt")),
                    content="should not write",
                ),
            ),
        ),
        session=session,
        filesystem=vfs_section.filesystem,
    )

    _assert_failure_message(result, had_pending_writes=True)
    assert result.value is not None
    assert result.value.writes == ()
    assert result.value.stderr == "Execution timed out."


def test_stdout_truncation_and_flush() -> None:
    session, vfs_section, tool = _setup_sections()

    extra = asteval_module._MAX_STREAM_LENGTH + 100
    params = EvalParams(
        code=f"print('A' * {extra}, flush=True)",
    )

    result = invoke_tool(
        tool, params, session=session, filesystem=vfs_section.filesystem
    )
    assert result.value is not None
    payload = result.value
    assert payload.stdout.endswith("...")
    assert len(payload.stdout) == asteval_module._MAX_STREAM_LENGTH


def test_print_invalid_sep_reports_error() -> None:
    session, vfs_section, tool = _setup_sections()

    params = EvalParams(code="print('value', sep=0)")

    result = invoke_tool(
        tool, params, session=session, filesystem=vfs_section.filesystem
    )

    assert result.message == "Evaluation failed; review stderr details in the payload."
    assert result.value is not None
    payload = result.value
    assert "sep must be None or a string." in payload.stderr


def test_print_invalid_end_reports_error() -> None:
    session, vfs_section, tool = _setup_sections()

    params = EvalParams(code="print('value', end=0)")

    result = invoke_tool(
        tool, params, session=session, filesystem=vfs_section.filesystem
    )

    assert result.message == "Evaluation failed; review stderr details in the payload."
    assert result.value is not None
    payload = result.value
    assert "end must be None or a string." in payload.stderr


def test_stdout_preserved_when_exception_occurs() -> None:
    session, vfs_section, tool = _setup_sections()

    result = invoke_tool(
        tool,
        EvalParams(
            code="print('before error')\n1/0",
            writes=(
                EvalFileWrite(
                    path=VfsPath(("logs", "pending.txt")),
                    content="value",
                ),
            ),
        ),
        session=session,
        filesystem=vfs_section.filesystem,
    )

    _assert_failure_message(result, had_pending_writes=True)
    assert result.value is not None
    payload = result.value
    assert payload.stdout == "before error\n"
    assert "division by zero" in payload.stderr
    assert payload.writes == ()


def test_invalid_global_names_raise() -> None:
    session, vfs_section, tool = _setup_sections()

    with pytest.raises(ToolValidationError):
        invoke_tool(
            tool,
            EvalParams(code="0", globals={"": "0"}),
            session=session,
            filesystem=vfs_section.filesystem,
        )

    with pytest.raises(ToolValidationError):
        invoke_tool(
            tool,
            EvalParams(code="0", globals={"not valid": "0"}),
            session=session,
            filesystem=vfs_section.filesystem,
        )


def test_code_length_validation() -> None:
    session, vfs_section, tool = _setup_sections()

    long_code = "x" * (asteval_module._MAX_CODE_LENGTH + 1)
    with pytest.raises(ToolValidationError):
        invoke_tool(
            tool,
            EvalParams(code=long_code),
            session=session,
            filesystem=vfs_section.filesystem,
        )


def test_control_character_validation() -> None:
    session, vfs_section, tool = _setup_sections()

    with pytest.raises(ToolValidationError):
        invoke_tool(
            tool,
            EvalParams(code="print('ok')\x01"),
            session=session,
            filesystem=vfs_section.filesystem,
        )


def test_normalize_path_skips_blank_segments() -> None:
    normalized = vfs_types_module.normalize_path(VfsPath(("docs", " ", "file.txt")))

    assert normalized.segments == ("docs", "file.txt")

    nested = vfs_types_module.normalize_path(VfsPath(("logs//", "item.txt")))
    assert nested.segments == ("logs", "item.txt")


@pytest.mark.parametrize(
    "path",
    [
        VfsPath(("/absolute",)),
        VfsPath((".", "file.txt")),
        VfsPath(("..", "file.txt")),
        VfsPath(("a" * (vfs_types_module._MAX_SEGMENT_LENGTH + 1),)),
        VfsPath(tuple(f"seg{i}" for i in range(vfs_types_module._MAX_PATH_DEPTH + 1))),
    ],
)
def test_normalize_path_invalid_segments(path: VfsPath) -> None:
    with pytest.raises(ToolValidationError):
        vfs_types_module.normalize_path(path)


def test_read_requires_existing_file() -> None:
    session, vfs_section, tool = _setup_sections()

    with pytest.raises(ToolValidationError):
        invoke_tool(
            tool,
            EvalParams(
                code="0",
                reads=(EvalFileRead(path=VfsPath(("docs", "missing.txt"))),),
            ),
            session=session,
            filesystem=vfs_section.filesystem,
        )


def test_duplicate_reads_rejected() -> None:
    session, vfs_section, tool = _setup_sections()

    read = EvalFileRead(path=VfsPath(("docs", "info.txt")))

    with pytest.raises(ToolValidationError):
        invoke_tool(
            tool,
            EvalParams(code="0", reads=(read, read)),
            session=session,
            filesystem=vfs_section.filesystem,
        )


def test_duplicate_writes_rejected() -> None:
    session, vfs_section, tool = _setup_sections()

    write = EvalFileWrite(path=VfsPath(("output", "data.txt")), content="x")

    with pytest.raises(ToolValidationError):
        invoke_tool(
            tool,
            EvalParams(code="0", writes=(write, write)),
            session=session,
            filesystem=vfs_section.filesystem,
        )


def test_read_write_conflict_rejected() -> None:
    session, vfs_section, tool = _setup_sections()

    path = VfsPath(("docs", "info.txt"))

    with pytest.raises(ToolValidationError):
        invoke_tool(
            tool,
            EvalParams(
                code="0",
                reads=(EvalFileRead(path=path),),
                writes=(EvalFileWrite(path=path, content="x"),),
            ),
            session=session,
            filesystem=vfs_section.filesystem,
        )


def test_invalid_write_mode_rejected() -> None:
    session, vfs_section, tool = _setup_sections()

    write = EvalFileWrite(
        path=VfsPath(("output", "data.txt")),
        content="x",
        mode="invalid",  # type: ignore[arg-type]
    )

    with pytest.raises(ToolValidationError):
        invoke_tool(
            tool,
            EvalParams(code="0", writes=(write,)),
            session=session,
            filesystem=vfs_section.filesystem,
        )


def test_write_content_length_validation() -> None:
    session, vfs_section, tool = _setup_sections()

    long_content = "x" * (asteval_module._MAX_WRITE_LENGTH + 1)
    write = EvalFileWrite(path=VfsPath(("output", "large.txt")), content=long_content)

    with pytest.raises(ToolValidationError):
        invoke_tool(
            tool,
            EvalParams(code="0", writes=(write,)),
            session=session,
            filesystem=vfs_section.filesystem,
        )


def test_write_content_requires_ascii() -> None:
    session, vfs_section, tool = _setup_sections()

    write = EvalFileWrite(path=VfsPath(("output", "data.txt")), content="café")

    with pytest.raises(ToolValidationError):
        invoke_tool(
            tool,
            EvalParams(code="0", writes=(write,)),
            session=session,
            filesystem=vfs_section.filesystem,
        )


def test_create_mode_rejects_existing_file() -> None:
    session, vfs_section, tool = _setup_sections()

    write_tool = cast(
        Tool[WriteFileParams, WriteFile], find_tool(vfs_section, "write_file")
    )
    existing_path = VfsPath(("docs", "info.txt"))
    invoke_tool(
        write_tool,
        WriteFileParams(file_path="docs/info.txt", content="original"),
        session=session,
        filesystem=vfs_section.filesystem,
    )

    result = invoke_tool(
        tool,
        EvalParams(
            code="0",
            writes=(
                EvalFileWrite(path=existing_path, content="new content", mode="create"),
            ),
        ),
        session=session,
        filesystem=vfs_section.filesystem,
    )

    _assert_success_message(result)
    files = _get_files_dict(vfs_section)
    assert files[existing_path.segments] == "original"


def test_append_requires_existing_file() -> None:
    session, vfs_section, tool = _setup_sections()

    result = invoke_tool(
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
        filesystem=vfs_section.filesystem,
    )

    _assert_success_message(result)
    fs = _get_filesystem(vfs_section)
    assert not fs._files


def test_overwrite_updates_existing_file() -> None:
    session, vfs_section, tool = _setup_sections()

    write_tool = cast(
        Tool[WriteFileParams, WriteFile], find_tool(vfs_section, "write_file")
    )
    path = VfsPath(("docs", "info.txt"))
    invoke_tool(
        write_tool,
        WriteFileParams(file_path="docs/info.txt", content="old"),
        session=session,
        filesystem=vfs_section.filesystem,
    )

    result = invoke_tool(
        tool,
        EvalParams(
            code="0",
            writes=(EvalFileWrite(path=path, content="updated", mode="overwrite"),),
        ),
        session=session,
        filesystem=vfs_section.filesystem,
    )

    _assert_success_message(result)
    files = _get_files_dict(vfs_section)
    assert files[path.segments] == "updated"


def test_message_summarizes_multiple_writes() -> None:
    session, vfs_section, tool = _setup_sections()

    writes = tuple(
        EvalFileWrite(path=VfsPath(("output", f"file{i}.txt")), content="x")
        for i in range(4)
    )

    result = invoke_tool(
        tool,
        EvalParams(code="0", writes=writes),
        session=session,
        filesystem=vfs_section.filesystem,
    )

    _assert_success_message(result)
    assert (
        "writes=4 file(s): output/file0.txt, output/file1.txt, output/file2.txt, +1 more"
        in result.message
    )
    files = _get_files_dict(vfs_section)
    paths = sorted(files.keys())
    assert paths == [("output", f"file{i}.txt") for i in range(4)]


def test_read_text_uses_persisted_mount(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    sunfish = root / "sunfish"
    sunfish.mkdir()
    readme = sunfish / "README.md"
    readme.write_text("hello mount", encoding="utf-8")

    bus = InProcessDispatcher()
    session = Session(bus=bus)
    vfs_section = VfsToolsSection(
        session=session,
        mounts=(HostMount(host_path="sunfish", mount_path=VfsPath(("sunfish",))),),
        allowed_host_roots=(root,),
    )
    list_tool = cast(
        Tool[ListDirectory, ListDirectoryResult],
        find_tool(vfs_section, "ls"),
    )
    invoke_tool(
        list_tool, ListDirectory(), session=session, filesystem=vfs_section.filesystem
    )
    section = AstevalSection(session=session, filesystem=vfs_section.filesystem)
    tool = cast(Tool[EvalParams, EvalResult], find_tool(section, "evaluate_python"))

    result = invoke_tool(
        tool,
        EvalParams(code="read_text('sunfish/README.md')"),
        session=session,
        filesystem=vfs_section.filesystem,
    )

    _assert_success_message(result)
    assert result.value is not None
    payload = result.value
    assert payload.value_repr == "'hello mount'"


def test_list_directory_result_renders_readable() -> None:
    result = ListDirectoryResult(
        path=VfsPath(("workspace", "project")),
        directories=("src", "tests"),
        files=("README.md",),
    )

    rendered = result.render()

    assert "workspace/project" in rendered
    assert "Directories:" in rendered
    assert "src" in rendered
    assert "Files:" in rendered
    assert "README.md" in rendered


def test_write_text_rejects_empty_path() -> None:
    session, vfs_section, tool = _setup_sections()

    result = invoke_tool(
        tool,
        EvalParams(code="write_text('', 'data')"),
        session=session,
        filesystem=vfs_section.filesystem,
    )

    _assert_failure_message(result, had_pending_writes=True)
    assert result.value is not None
    assert "Path must be non-empty." in result.value.stderr


def test_helper_write_validation_error_bubbles() -> None:
    session, vfs_section, tool = _setup_sections()

    result = invoke_tool(
        tool,
        EvalParams(code="write_text('../invalid', 'data')"),
        session=session,
        filesystem=vfs_section.filesystem,
    )

    _assert_failure_message(result, had_pending_writes=True)
    assert result.value is not None
    assert "Path segments may not include '.' or '..'." in result.value.stderr
    assert result.value.writes == ()


def test_globals_formatting_covers_primitives() -> None:
    session, vfs_section, tool = _setup_sections()

    code = (
        "text = 'hello'\n"
        "number = 7\n"
        "pi = 3.5\n"
        "flag = True\n"
        "empty = None\n"
        "value = {'key': 1}\n"
    )

    result = invoke_tool(
        tool,
        EvalParams(code=code),
        session=session,
        filesystem=vfs_section.filesystem,
    )

    assert result.value is not None
    payload = result.value
    assert payload.globals["text"] == "hello"
    assert payload.globals["number"] == "7"
    assert payload.globals["pi"] == "3.5"
    assert payload.globals["flag"] == "true"
    assert payload.globals["empty"] == "null"
    assert payload.globals["value"].startswith("!repr:")


def test_eval_helper_passthrough_functions() -> None:
    path = VfsPath(("docs", "notes.txt"))
    read = EvalFileRead(path=path)
    write = EvalFileWrite(path=path, content="value", mode="create")

    reads = asteval_module.normalize_eval_reads((read,))
    assert reads == (read,)

    writes = asteval_module.normalize_eval_writes((write,))
    assert writes == (write,)
    assert asteval_module.normalize_eval_write(write) == write

    globals_payload = asteval_module.parse_eval_globals({"value": "1"})
    assert globals_payload["value"] == 1

    assert asteval_module.alias_for_eval_path(path) == "docs/notes.txt"
    summary = asteval_module.summarize_eval_writes(writes)
    assert summary is not None and "docs/notes.txt" in summary


def test_interpreter_error_surfaces_in_stderr() -> None:
    session, vfs_section, tool = _setup_sections()

    result = invoke_tool(
        tool,
        EvalParams(code="unknown_name + 1"),
        session=session,
        filesystem=vfs_section.filesystem,
    )

    _assert_failure_message(result, had_pending_writes=False)
    assert result.value is not None
    payload = result.value
    assert "unknown_name" in payload.stderr


def test_write_text_conflict_with_read_path() -> None:
    session, vfs_section, tool = _setup_sections()

    write_tool = cast(
        Tool[WriteFileParams, WriteFile], find_tool(vfs_section, "write_file")
    )
    path = VfsPath(("docs", "info.txt"))
    invoke_tool(
        write_tool,
        WriteFileParams(file_path="docs/info.txt", content="seed"),
        session=session,
        filesystem=vfs_section.filesystem,
    )

    params = EvalParams(
        code="write_text('docs/info.txt', 'data')",
        reads=(EvalFileRead(path=path),),
    )

    result = invoke_tool(
        tool, params, session=session, filesystem=vfs_section.filesystem
    )
    _assert_failure_message(result, had_pending_writes=True)
    assert result.value is not None
    assert (
        "Writes queued during execution must not target read paths."
        in result.value.stderr
    )


def test_write_text_duplicate_targets() -> None:
    session, vfs_section, tool = _setup_sections()

    params = EvalParams(
        code=("write_text('logs/out.txt', 'a')\nwrite_text('logs/out.txt', 'b')"),
    )

    result = invoke_tool(
        tool, params, session=session, filesystem=vfs_section.filesystem
    )
    _assert_failure_message(result, had_pending_writes=True)
    assert result.value is not None
    assert "Duplicate write targets detected." in result.value.stderr


def test_missing_template_variable_raises() -> None:
    session, vfs_section, tool = _setup_sections()

    params = EvalParams(
        code="0",
        writes=(
            EvalFileWrite(
                path=VfsPath(("output", "report.txt")),
                content="Value: {missing}",
            ),
        ),
    )

    with pytest.raises(ToolValidationError):
        invoke_tool(tool, params, session=session, filesystem=vfs_section.filesystem)


def test_duplicate_final_writes_detected() -> None:
    session, vfs_section, tool = _setup_sections()

    params = EvalParams(
        code="write_text('output/data.txt', 'helper')",
        writes=(
            EvalFileWrite(
                path=VfsPath(("output", "data.txt")),
                content="base",
            ),
        ),
    )

    result = invoke_tool(
        tool, params, session=session, filesystem=vfs_section.filesystem
    )
    _assert_failure_message(result, had_pending_writes=True)
    assert result.value is not None
    assert "Duplicate write targets detected." in result.value.stderr


def test_overwrite_requires_existing_file() -> None:
    session, vfs_section, tool = _setup_sections()

    params = EvalParams(
        code="0",
        writes=(
            EvalFileWrite(
                path=VfsPath(("docs", "missing.txt")),
                content="value",
                mode="overwrite",
            ),
        ),
    )

    result = invoke_tool(
        tool, params, session=session, filesystem=vfs_section.filesystem
    )
    _assert_success_message(result)
    fs = _get_filesystem(vfs_section)
    assert not fs._files


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


# -----------------------------------------------------------------------------
# Config Tests
# -----------------------------------------------------------------------------


def test_asteval_config_accepts_overrides() -> None:
    """Test that config accepts_overrides is respected."""
    from weakincentives.contrib.tools import AstevalConfig

    bus = InProcessDispatcher()
    session = Session(bus=bus)
    config = AstevalConfig(accepts_overrides=True)
    section = AstevalSection(session=session, config=config)

    assert section.accepts_overrides is True


def test_asteval_filesystem_property_returns_filesystem() -> None:
    """Test that the filesystem property returns the filesystem instance."""
    from weakincentives.contrib.tools import Filesystem

    bus = InProcessDispatcher()
    session = Session(bus=bus)
    section = AstevalSection(session=session)

    fs = section.filesystem
    assert isinstance(fs, Filesystem)


def test_asteval_clone_rejects_invalid_filesystem_type() -> None:
    """Test that clone raises TypeError for invalid filesystem argument."""
    bus = InProcessDispatcher()
    session = Session(bus=bus)
    section = AstevalSection(session=session)

    new_session = Session(bus=bus)
    with pytest.raises(TypeError, match="filesystem must be a Filesystem instance"):
        section.clone(session=new_session, filesystem="not a filesystem")
