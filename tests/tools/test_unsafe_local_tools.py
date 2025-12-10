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

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Iterator, Sequence
from datetime import datetime
from pathlib import Path

import pytest

import weakincentives.tools.unsafe_local as unsafe_local_module
import weakincentives.tools.vfs as vfs_module
from tests.tools.helpers import build_tool_context, find_tool
from weakincentives.runtime.events import InProcessEventBus
from weakincentives.runtime.session import Session
from weakincentives.tools import (
    EditFileParams,
    EvalParams,
    GlobParams,
    GrepParams,
    HostMount,
    ListDirectoryParams,
    LocalShellParams,
    LocalShellResult,
    ReadFileParams,
    RemoveParams,
    UnsafeLocalSandboxConfig,
    UnsafeLocalSandboxSection,
    UnsafeLocalWorkspace,
    WriteFileParams,
)
from weakincentives.tools.asteval import EvalResult
from weakincentives.tools.errors import ToolValidationError
from weakincentives.tools.vfs import (
    FileInfo,
    GlobMatch,
    GrepMatch,
    ReadFileResult,
    VfsPath,
    WriteFile,
)


def _make_section(
    *,
    session: Session,
    workspace_root: Path | None = None,
    mounts: Sequence[HostMount] = (),
    allowed_host_roots: Sequence[os.PathLike[str] | str] = (),
    clock: Callable[[], datetime] | None = None,
) -> UnsafeLocalSandboxSection:
    config = UnsafeLocalSandboxConfig(
        workspace_root=workspace_root,
        mounts=mounts,
        allowed_host_roots=allowed_host_roots,
        base_environment={"PATH": "/usr/bin"},
        clock=clock,
    )
    return UnsafeLocalSandboxSection(session=session, config=config)


def _setup_host_mount(
    tmp_path: Path, *, content: str = "hello world"
) -> tuple[Path, HostMount, Path]:
    host_root = tmp_path / "host-root"
    repo = host_root / "sunfish"
    repo.mkdir(parents=True, exist_ok=True)
    file_path = repo / "README.md"
    file_path.write_text(content, encoding="utf-8")
    mount = HostMount(
        host_path="sunfish",
        mount_path=vfs_module.VfsPath(("sunfish",)),
    )
    return host_root, mount, file_path


@pytest.fixture()
def session_and_bus() -> tuple[Session, InProcessEventBus]:
    bus = InProcessEventBus()
    return Session(bus=bus), bus


class TestSectionRegistration:
    def test_section_registers_shell_tool(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)

        tool = find_tool(section, "shell_execute")
        assert tool.description.startswith("Run a short command")

    def test_section_registers_vfs_tools(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)

        tool = find_tool(section, "ls")
        assert tool.description.startswith("List directory entries")

    def test_section_registers_eval_tool(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)

        tool = find_tool(section, "evaluate_python")
        assert tool.description.startswith("Run a short Python")

    def test_section_registers_all_vfs_tools(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)

        tool_names = {tool.name for tool in section.tools()}
        expected = {
            "ls",
            "read_file",
            "write_file",
            "edit_file",
            "glob",
            "grep",
            "rm",
            "shell_execute",
            "evaluate_python",
        }
        assert expected <= tool_names


class TestWorkspaceLifecycle:
    def test_workspace_created_on_first_tool_invocation(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        handler = find_tool(section, "shell_execute").handler
        assert handler is not None

        handler(
            LocalShellParams(command=("true",)),
            context=build_tool_context(session),
        )

        assert section._workspace_handle is not None
        assert section._workspace_handle.workspace_path.exists()

    def test_workspace_directory_structure(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        handler = find_tool(section, "shell_execute").handler
        assert handler is not None

        handler(
            LocalShellParams(command=("true",)),
            context=build_tool_context(session),
        )

        handle = section._workspace_handle
        assert handle is not None
        assert f"wink-{session.session_id}" in str(handle.workspace_path)

    def test_workspace_state_registered_in_session(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        handler = find_tool(section, "shell_execute").handler
        assert handler is not None

        handler(
            LocalShellParams(command=("true",)),
            context=build_tool_context(session),
        )

        workspace = session.query(UnsafeLocalWorkspace).latest()
        assert workspace is not None
        assert workspace.workspace_path.endswith(str(session.session_id))

    def test_workspace_cleanup_on_close(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        handler = find_tool(section, "shell_execute").handler
        assert handler is not None

        handler(
            LocalShellParams(command=("true",)),
            context=build_tool_context(session),
        )

        workspace_path = section._workspace_handle.workspace_path
        assert workspace_path.exists()

        section.close()

        assert not workspace_path.exists()
        assert section._workspace_handle is None


class TestHostMounts:
    def test_host_mount_populates_prompt(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        host_root, mount, _file_path = _setup_host_mount(tmp_path)
        section = _make_section(
            session=session,
            workspace_root=tmp_path / "workspace",
            mounts=(mount,),
            allowed_host_roots=(host_root,),
        )

        assert "Configured host mounts:" in section.template
        assert str(host_root / "sunfish") in section.template

    def test_host_mount_materializes_files(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        host_root, mount, file_path = _setup_host_mount(tmp_path)
        section = _make_section(
            session=session,
            workspace_root=tmp_path / "workspace",
            mounts=(mount,),
            allowed_host_roots=(host_root,),
        )
        handler = find_tool(section, "shell_execute").handler
        assert handler is not None

        handler(
            LocalShellParams(command=("true",)),
            context=build_tool_context(session),
        )

        handle = section._workspace_handle
        assert handle is not None
        mounted = handle.workspace_path / "sunfish" / file_path.name
        assert mounted.read_text(encoding="utf-8") == file_path.read_text(
            encoding="utf-8"
        )

    def test_host_mount_resolver_rejects_empty_path(self, tmp_path: Path) -> None:
        with pytest.raises(ToolValidationError):
            unsafe_local_module._resolve_single_host_mount(
                HostMount(host_path=""),
                (tmp_path,),
            )

    def test_resolve_host_path_requires_allowed_roots(self) -> None:
        with pytest.raises(ToolValidationError):
            unsafe_local_module._resolve_host_path("docs", ())

    def test_resolve_host_path_rejects_outside_root(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        with pytest.raises(ToolValidationError):
            unsafe_local_module._resolve_host_path("../outside", (root,))

    def test_normalize_mount_globs_discards_empty_entries(self) -> None:
        result = unsafe_local_module._normalize_mount_globs(
            (" *.py ", " ", "*.md"),
            "include_glob",
        )
        assert result == ("*.py", "*.md")

    def test_preview_mount_entries_handles_file(self, tmp_path: Path) -> None:
        file_path = tmp_path / "item.txt"
        file_path.write_text("payload", encoding="utf-8")
        result = unsafe_local_module._preview_mount_entries(file_path)
        assert result == ("item.txt",)

    def test_preview_mount_entries_raises_on_oserror(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        directory = tmp_path / "items"
        directory.mkdir()
        original_iterdir = Path.iterdir

        def _raise(self: Path) -> Iterator[Path]:
            if self == directory:
                raise OSError("boom")
            return original_iterdir(self)

        monkeypatch.setattr(Path, "iterdir", _raise)
        with pytest.raises(ToolValidationError):
            unsafe_local_module._preview_mount_entries(directory)


class TestShellExecution:
    def test_shell_execute_runs_command(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        handler = find_tool(section, "shell_execute").handler
        assert handler is not None

        result = handler(
            LocalShellParams(command=("echo", "hello")),
            context=build_tool_context(session),
        )

        assert result.success
        assert isinstance(result.value, LocalShellResult)
        assert result.value.exit_code == 0
        assert "hello" in result.value.stdout

    def test_shell_execute_captures_exit_code(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        handler = find_tool(section, "shell_execute").handler
        assert handler is not None

        result = handler(
            LocalShellParams(command=("false",)),
            context=build_tool_context(session),
        )

        assert isinstance(result.value, LocalShellResult)
        assert result.value.exit_code == 1

    def test_shell_execute_validates_empty_command(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        handler = find_tool(section, "shell_execute").handler
        assert handler is not None

        with pytest.raises(ToolValidationError, match="at least one entry"):
            handler(
                LocalShellParams(command=()),
                context=build_tool_context(session),
            )

    def test_shell_execute_validates_command_length(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        handler = find_tool(section, "shell_execute").handler
        assert handler is not None

        long_cmd = "x" * 5000
        with pytest.raises(ToolValidationError, match="too long"):
            handler(
                LocalShellParams(command=(long_cmd,)),
                context=build_tool_context(session),
            )

    def test_shell_execute_uses_workspace_cwd(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        handler = find_tool(section, "shell_execute").handler
        assert handler is not None

        result = handler(
            LocalShellParams(command=("pwd",)),
            context=build_tool_context(session),
        )

        assert isinstance(result.value, LocalShellResult)
        assert str(session.session_id) in result.value.stdout

    def test_shell_execute_respects_relative_cwd(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        _ = section.ensure_workspace()
        subdir = section._workspace_handle.workspace_path / "subdir"
        subdir.mkdir()
        handler = find_tool(section, "shell_execute").handler
        assert handler is not None

        result = handler(
            LocalShellParams(command=("pwd",), cwd="subdir"),
            context=build_tool_context(session),
        )

        assert isinstance(result.value, LocalShellResult)
        assert "subdir" in result.value.stdout

    def test_shell_execute_rejects_absolute_cwd(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        handler = find_tool(section, "shell_execute").handler
        assert handler is not None

        with pytest.raises(ToolValidationError, match="relative to /workspace"):
            handler(
                LocalShellParams(command=("pwd",), cwd="/absolute/path"),
                context=build_tool_context(session),
            )

    def test_shell_execute_handles_timeout(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        handler = find_tool(section, "shell_execute").handler
        assert handler is not None

        result = handler(
            LocalShellParams(command=("sleep", "10"), timeout_seconds=1.0),
            context=build_tool_context(session),
        )

        assert isinstance(result.value, LocalShellResult)
        assert result.value.timed_out
        assert result.value.exit_code == 124

    def test_shell_execute_captures_stderr(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        handler = find_tool(section, "shell_execute").handler
        assert handler is not None

        result = handler(
            LocalShellParams(command=("sh", "-c", "echo error >&2")),
            context=build_tool_context(session),
        )

        assert isinstance(result.value, LocalShellResult)
        assert "error" in result.value.stderr


class TestPythonEvaluation:
    def test_evaluate_python_runs_code(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        handler = find_tool(section, "evaluate_python").handler
        assert handler is not None

        result = handler(
            EvalParams(code="print(2 + 2)"),
            context=build_tool_context(session),
        )

        assert result.success
        assert isinstance(result.value, EvalResult)
        assert "4" in result.value.stdout

    def test_evaluate_python_handles_timeout(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        handler = find_tool(section, "evaluate_python").handler
        assert handler is not None

        result = handler(
            EvalParams(code="import time; time.sleep(10)"),
            context=build_tool_context(session),
        )

        assert not result.success
        assert isinstance(result.value, EvalResult)
        assert "timed out" in result.value.stderr.lower()

    def test_evaluate_python_rejects_reads(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        handler = find_tool(section, "evaluate_python").handler
        assert handler is not None

        from weakincentives.tools.asteval import EvalFileRead

        with pytest.raises(ToolValidationError, match="reads are not supported"):
            handler(
                EvalParams(
                    code="print(1)",
                    reads=(EvalFileRead(path=VfsPath(("test.txt",))),),
                ),
                context=build_tool_context(session),
            )

    def test_evaluate_python_rejects_writes(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        handler = find_tool(section, "evaluate_python").handler
        assert handler is not None

        from weakincentives.tools.asteval import EvalFileWrite

        with pytest.raises(ToolValidationError, match="writes are not supported"):
            handler(
                EvalParams(
                    code="print(1)",
                    writes=(
                        EvalFileWrite(path=VfsPath(("test.txt",)), content="test"),
                    ),
                ),
                context=build_tool_context(session),
            )

    def test_evaluate_python_rejects_globals(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        handler = find_tool(section, "evaluate_python").handler
        assert handler is not None

        with pytest.raises(ToolValidationError, match="globals are not supported"):
            handler(
                EvalParams(code="print(1)", globals={"x": "1"}),
                context=build_tool_context(session),
            )


class TestVfsOperations:
    def test_write_file_creates_file(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        handler = find_tool(section, "write_file").handler
        assert handler is not None

        result = handler(
            WriteFileParams(file_path="test.txt", content="hello world"),
            context=build_tool_context(session),
        )

        assert result.success
        assert isinstance(result.value, WriteFile)
        assert result.value.mode == "create"

        handle = section._workspace_handle
        assert handle is not None
        written = handle.workspace_path / "test.txt"
        assert written.read_text(encoding="utf-8") == "hello world"

    def test_write_file_rejects_existing_file(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        _ = section.ensure_workspace()
        existing = section._workspace_handle.workspace_path / "existing.txt"
        existing.write_text("existing", encoding="utf-8")
        handler = find_tool(section, "write_file").handler
        assert handler is not None

        with pytest.raises(ToolValidationError, match="already exists"):
            handler(
                WriteFileParams(file_path="existing.txt", content="new content"),
                context=build_tool_context(session),
            )

    def test_read_file_returns_content(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        _ = section.ensure_workspace()
        test_file = section._workspace_handle.workspace_path / "test.txt"
        test_file.write_text("line1\nline2\nline3", encoding="utf-8")
        handler = find_tool(section, "read_file").handler
        assert handler is not None

        result = handler(
            ReadFileParams(file_path="test.txt"),
            context=build_tool_context(session),
        )

        assert result.success
        assert isinstance(result.value, ReadFileResult)
        assert "line1" in result.value.content
        assert result.value.total_lines == 3

    def test_read_file_with_pagination(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        _ = section.ensure_workspace()
        test_file = section._workspace_handle.workspace_path / "test.txt"
        test_file.write_text("line1\nline2\nline3\nline4\nline5", encoding="utf-8")
        handler = find_tool(section, "read_file").handler
        assert handler is not None

        result = handler(
            ReadFileParams(file_path="test.txt", offset=1, limit=2),
            context=build_tool_context(session),
        )

        assert isinstance(result.value, ReadFileResult)
        assert result.value.offset == 1
        assert result.value.limit == 2
        assert "line2" in result.value.content
        assert "line3" in result.value.content
        assert "line1" not in result.value.content

    def test_edit_file_replaces_content(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        _ = section.ensure_workspace()
        test_file = section._workspace_handle.workspace_path / "test.txt"
        test_file.write_text("old content", encoding="utf-8")
        handler = find_tool(section, "edit_file").handler
        assert handler is not None

        result = handler(
            EditFileParams(file_path="test.txt", old_string="old", new_string="new"),
            context=build_tool_context(session),
        )

        assert result.success
        assert test_file.read_text(encoding="utf-8") == "new content"

    def test_edit_file_requires_unique_match(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        _ = section.ensure_workspace()
        test_file = section._workspace_handle.workspace_path / "test.txt"
        test_file.write_text("foo bar foo", encoding="utf-8")
        handler = find_tool(section, "edit_file").handler
        assert handler is not None

        with pytest.raises(ToolValidationError, match="exactly once"):
            handler(
                EditFileParams(
                    file_path="test.txt",
                    old_string="foo",
                    new_string="baz",
                    replace_all=False,
                ),
                context=build_tool_context(session),
            )

    def test_edit_file_replace_all(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        _ = section.ensure_workspace()
        test_file = section._workspace_handle.workspace_path / "test.txt"
        test_file.write_text("foo bar foo", encoding="utf-8")
        handler = find_tool(section, "edit_file").handler
        assert handler is not None

        result = handler(
            EditFileParams(
                file_path="test.txt",
                old_string="foo",
                new_string="baz",
                replace_all=True,
            ),
            context=build_tool_context(session),
        )

        assert result.success
        assert test_file.read_text(encoding="utf-8") == "baz bar baz"

    def test_list_directory_returns_entries(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        _ = section.ensure_workspace()
        workspace = section._workspace_handle.workspace_path
        (workspace / "file.txt").write_text("content", encoding="utf-8")
        (workspace / "subdir").mkdir()
        handler = find_tool(section, "ls").handler
        assert handler is not None

        result = handler(
            ListDirectoryParams(path=""),
            context=build_tool_context(session),
        )

        assert result.success
        assert isinstance(result.value, tuple)
        assert all(isinstance(v, FileInfo) for v in result.value)
        entries = [v for v in result.value if isinstance(v, FileInfo)]
        paths = [info.path.segments for info in entries]
        assert ("file.txt",) in paths
        assert ("subdir",) in paths

    def test_glob_matches_patterns(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        _ = section.ensure_workspace()
        workspace = section._workspace_handle.workspace_path
        (workspace / "test.py").write_text("python", encoding="utf-8")
        (workspace / "test.txt").write_text("text", encoding="utf-8")
        handler = find_tool(section, "glob").handler
        assert handler is not None

        result = handler(
            GlobParams(pattern="*.py", path=""),
            context=build_tool_context(session),
        )

        assert result.success
        assert isinstance(result.value, tuple)
        assert len(result.value) == 1
        assert isinstance(result.value[0], GlobMatch)
        assert result.value[0].path.segments == ("test.py",)

    def test_grep_finds_pattern(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        _ = section.ensure_workspace()
        workspace = section._workspace_handle.workspace_path
        (workspace / "test.py").write_text("# TODO: fix this", encoding="utf-8")
        handler = find_tool(section, "grep").handler
        assert handler is not None

        result = handler(
            GrepParams(pattern="TODO", path=""),
            context=build_tool_context(session),
        )

        assert result.success
        assert isinstance(result.value, tuple)
        assert len(result.value) == 1
        assert isinstance(result.value[0], GrepMatch)
        assert "TODO" in result.value[0].line

    def test_remove_deletes_file(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        _ = section.ensure_workspace()
        workspace = section._workspace_handle.workspace_path
        test_file = workspace / "test.txt"
        test_file.write_text("content", encoding="utf-8")
        handler = find_tool(section, "rm").handler
        assert handler is not None

        result = handler(
            RemoveParams(path="test.txt"),
            context=build_tool_context(session),
        )

        assert result.success
        assert not test_file.exists()

    def test_remove_deletes_directory(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        _ = section.ensure_workspace()
        workspace = section._workspace_handle.workspace_path
        subdir = workspace / "subdir"
        subdir.mkdir()
        (subdir / "file.txt").write_text("content", encoding="utf-8")
        handler = find_tool(section, "rm").handler
        assert handler is not None

        result = handler(
            RemoveParams(path="subdir"),
            context=build_tool_context(session),
        )

        assert result.success
        assert not subdir.exists()

    def test_remove_rejects_workspace_root(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        handler = find_tool(section, "rm").handler
        assert handler is not None

        # "/" normalizes to empty path, which triggers a different validation
        with pytest.raises(ToolValidationError, match="must reference a file"):
            handler(
                RemoveParams(path="/"),
                context=build_tool_context(session),
            )


class TestCloning:
    def test_clone_creates_new_section(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)

        new_session = Session(bus=InProcessEventBus())
        cloned = section.clone(session=new_session)

        assert cloned is not section
        assert cloned.session is new_session

    def test_clone_requires_session(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)

        with pytest.raises(TypeError, match="session is required"):
            section.clone(session="not a session")

    def test_clone_validates_bus_match(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)

        new_session = Session(bus=InProcessEventBus())
        with pytest.raises(TypeError, match="must match"):
            section.clone(session=new_session, bus=bus)


class TestHelperFunctions:
    def test_truncate_stream_limits_length(self) -> None:
        short = unsafe_local_module._truncate_stream("hello")
        assert short == "hello"

        long_text = "x" * (unsafe_local_module._MAX_STDIO_CHARS + 100)
        truncated = unsafe_local_module._truncate_stream(long_text)
        assert truncated.endswith("[truncated]")
        assert len(truncated) == unsafe_local_module._MAX_STDIO_CHARS

    def test_truncate_eval_stream_limits_length(self) -> None:
        short = unsafe_local_module._truncate_eval_stream("hello")
        assert short == "hello"

        long_text = "x" * (unsafe_local_module._EVAL_MAX_STREAM_LENGTH + 5)
        truncated = unsafe_local_module._truncate_eval_stream(long_text)
        assert truncated.endswith("...")
        assert len(truncated) == unsafe_local_module._EVAL_MAX_STREAM_LENGTH

    def test_ensure_ascii_rejects_non_ascii(self) -> None:
        with pytest.raises(ToolValidationError):
            unsafe_local_module._ensure_ascii("héllo", field="test")

    def test_normalize_command_rejects_empty_entry(self) -> None:
        with pytest.raises(ToolValidationError, match="must not be empty"):
            unsafe_local_module._normalize_command(("cmd", ""))

    def test_normalize_env_rejects_too_many_entries(self) -> None:
        env = {f"VAR{i}": f"value{i}" for i in range(100)}
        with pytest.raises(ToolValidationError, match="too many entries"):
            unsafe_local_module._normalize_env(env)

    def test_normalize_timeout_clamps_to_range(self) -> None:
        assert unsafe_local_module._normalize_timeout(0.1) == 1.0
        assert unsafe_local_module._normalize_timeout(500) == 120.0
        assert unsafe_local_module._normalize_timeout(60) == 60.0

    def test_normalize_cwd_rejects_dot_segments(self, tmp_path: Path) -> None:
        with pytest.raises(ToolValidationError, match="\\.\\."):
            unsafe_local_module._normalize_cwd("foo/../bar", tmp_path)


class TestPathBoundary:
    def test_assert_within_workspace_rejects_escape(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()

        with pytest.raises(ToolValidationError, match="escapes"):
            unsafe_local_module._assert_within_workspace(workspace, outside)

    def test_path_operations_respect_boundary(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        handler = find_tool(section, "read_file").handler
        assert handler is not None

        with pytest.raises(ToolValidationError):
            handler(
                ReadFileParams(file_path="../../../etc/passwd"),
                context=build_tool_context(session),
            )


class TestLocalShellResultRendering:
    def test_render_format(self) -> None:
        result = LocalShellResult(
            command=("echo", "hello"),
            cwd="/workspace",
            exit_code=0,
            stdout="hello",
            stderr="",
            duration_ms=123,
            timed_out=False,
        )
        rendered = result.render()
        assert "Shell command result:" in rendered
        assert "echo hello" in rendered
        assert "Exit code: 0" in rendered
        assert "Timed out: False" in rendered
        assert "Duration: 123 ms" in rendered
        assert "hello" in rendered

    def test_render_empty_stdout(self) -> None:
        result = LocalShellResult(
            command=("true",),
            cwd="/workspace",
            exit_code=0,
            stdout="",
            stderr="",
            duration_ms=1,
            timed_out=False,
        )
        rendered = result.render()
        assert "<empty>" in rendered


class TestHostMountEdgeCases:
    def test_host_mount_with_include_glob(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        host_root = tmp_path / "host-root"
        repo = host_root / "repo"
        repo.mkdir(parents=True)
        (repo / "file.py").write_text("python", encoding="utf-8")
        (repo / "file.txt").write_text("text", encoding="utf-8")

        mount = HostMount(
            host_path="repo",
            mount_path=VfsPath(("repo",)),
            include_glob=("*.py",),
        )
        section = _make_section(
            session=session,
            workspace_root=tmp_path / "workspace",
            mounts=(mount,),
            allowed_host_roots=(host_root,),
        )
        handler = find_tool(section, "shell_execute").handler
        assert handler is not None

        result = handler(
            LocalShellParams(command=("ls", "repo")),
            context=build_tool_context(session),
        )

        assert isinstance(result.value, LocalShellResult)
        assert "file.py" in result.value.stdout
        assert "file.txt" not in result.value.stdout

    def test_host_mount_with_exclude_glob(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        host_root = tmp_path / "host-root"
        repo = host_root / "repo"
        repo.mkdir(parents=True)
        (repo / "file.py").write_text("python", encoding="utf-8")
        (repo / "file.txt").write_text("text", encoding="utf-8")

        mount = HostMount(
            host_path="repo",
            mount_path=VfsPath(("repo",)),
            exclude_glob=("*.txt",),
        )
        section = _make_section(
            session=session,
            workspace_root=tmp_path / "workspace",
            mounts=(mount,),
            allowed_host_roots=(host_root,),
        )
        handler = find_tool(section, "shell_execute").handler
        assert handler is not None

        result = handler(
            LocalShellParams(command=("ls", "repo")),
            context=build_tool_context(session),
        )

        assert isinstance(result.value, LocalShellResult)
        assert "file.py" in result.value.stdout
        assert "file.txt" not in result.value.stdout

    def test_host_mount_with_max_bytes_exceeded(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        host_root = tmp_path / "host-root"
        repo = host_root / "repo"
        repo.mkdir(parents=True)
        # Create multiple files that exceed the byte budget
        for i in range(10):
            (repo / f"file{i}.txt").write_text("x" * 100, encoding="utf-8")

        mount = HostMount(
            host_path="repo",
            mount_path=VfsPath(("repo",)),
            max_bytes=50,  # Very small budget
        )
        with pytest.raises(ToolValidationError, match="byte budget"):
            section = _make_section(
                session=session,
                workspace_root=tmp_path / "workspace",
                mounts=(mount,),
                allowed_host_roots=(host_root,),
            )
            # Force hydration by ensuring workspace
            section.ensure_workspace()

    def test_host_mount_file_instead_of_directory(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        host_root = tmp_path / "host-root"
        host_root.mkdir(parents=True)
        single_file = host_root / "single.txt"
        single_file.write_text("content", encoding="utf-8")

        mount = HostMount(
            host_path="single.txt",
            mount_path=VfsPath(("mounted",)),
        )
        section = _make_section(
            session=session,
            workspace_root=tmp_path / "workspace",
            mounts=(mount,),
            allowed_host_roots=(host_root,),
        )
        handler = find_tool(section, "shell_execute").handler
        assert handler is not None

        result = handler(
            LocalShellParams(command=("cat", "mounted/single.txt")),
            context=build_tool_context(session),
        )

        assert isinstance(result.value, LocalShellResult)
        assert "content" in result.value.stdout


class TestMoreHelperFunctions:
    def test_normalize_timeout_rejects_nan(self) -> None:
        import math

        with pytest.raises(ToolValidationError, match="real number"):
            unsafe_local_module._normalize_timeout(math.nan)

    def test_normalize_env_rejects_long_key(self) -> None:
        long_key = "K" * 100
        with pytest.raises(ToolValidationError, match="longer than"):
            unsafe_local_module._normalize_env({long_key: "value"})

    def test_normalize_env_rejects_long_value(self) -> None:
        long_value = "V" * 600
        with pytest.raises(ToolValidationError, match="exceeds"):
            unsafe_local_module._normalize_env({"KEY": long_value})

    def test_normalize_env_rejects_empty_key(self) -> None:
        with pytest.raises(ToolValidationError, match="must not be empty"):
            unsafe_local_module._normalize_env({"": "value"})

    def test_normalize_cwd_rejects_single_dot(self, tmp_path: Path) -> None:
        with pytest.raises(ToolValidationError, match="\\.\\."):
            unsafe_local_module._normalize_cwd("foo/./bar", tmp_path)

    def test_normalize_cwd_rejects_deep_path(self, tmp_path: Path) -> None:
        deep_path = "/".join(["dir"] * 20)
        with pytest.raises(ToolValidationError, match="maximum depth"):
            unsafe_local_module._normalize_cwd(deep_path, tmp_path)

    def test_normalize_cwd_rejects_long_segment(self, tmp_path: Path) -> None:
        long_segment = "x" * 100
        with pytest.raises(ToolValidationError, match="exceeds"):
            unsafe_local_module._normalize_cwd(long_segment, tmp_path)


class TestVfsOperationsEdgeCases:
    def test_read_file_nonexistent(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        handler = find_tool(section, "read_file").handler
        assert handler is not None

        with pytest.raises(ToolValidationError, match="does not exist"):
            handler(
                ReadFileParams(file_path="nonexistent.txt"),
                context=build_tool_context(session),
            )

    def test_edit_file_nonexistent(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        handler = find_tool(section, "edit_file").handler
        assert handler is not None

        with pytest.raises(ToolValidationError, match="does not exist"):
            handler(
                EditFileParams(
                    file_path="nonexistent.txt",
                    old_string="old",
                    new_string="new",
                ),
                context=build_tool_context(session),
            )

    def test_edit_file_old_string_not_found(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        _ = section.ensure_workspace()
        test_file = section._workspace_handle.workspace_path / "test.txt"
        test_file.write_text("hello world", encoding="utf-8")
        handler = find_tool(section, "edit_file").handler
        assert handler is not None

        with pytest.raises(ToolValidationError, match="not found"):
            handler(
                EditFileParams(
                    file_path="test.txt",
                    old_string="goodbye",
                    new_string="hi",
                ),
                context=build_tool_context(session),
            )

    def test_glob_empty_pattern(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        handler = find_tool(section, "glob").handler
        assert handler is not None

        with pytest.raises(ToolValidationError, match="must not be empty"):
            handler(
                GlobParams(pattern="   ", path=""),
                context=build_tool_context(session),
            )

    def test_grep_invalid_regex(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        _ = section.ensure_workspace()
        handler = find_tool(section, "grep").handler
        assert handler is not None

        result = handler(
            GrepParams(pattern="[invalid", path=""),
            context=build_tool_context(session),
        )

        assert not result.success

    def test_remove_nonexistent(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        handler = find_tool(section, "rm").handler
        assert handler is not None

        with pytest.raises(ToolValidationError, match="No files matched"):
            handler(
                RemoveParams(path="nonexistent"),
                context=build_tool_context(session),
            )

    def test_list_directory_on_file(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        _ = section.ensure_workspace()
        test_file = section._workspace_handle.workspace_path / "file.txt"
        test_file.write_text("content", encoding="utf-8")
        handler = find_tool(section, "ls").handler
        assert handler is not None

        with pytest.raises(ToolValidationError, match="Cannot list a file"):
            handler(
                ListDirectoryParams(path="file.txt"),
                context=build_tool_context(session),
            )


class TestShellExecutionEdgeCases:
    def test_shell_execute_with_env(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        handler = find_tool(section, "shell_execute").handler
        assert handler is not None

        result = handler(
            LocalShellParams(
                command=("sh", "-c", "echo $TEST_VAR"),
                env={"test_var": "test_value"},
            ),
            context=build_tool_context(session),
        )

        assert isinstance(result.value, LocalShellResult)
        assert "test_value" in result.value.stdout

    def test_shell_execute_capture_disabled(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        handler = find_tool(section, "shell_execute").handler
        assert handler is not None

        result = handler(
            LocalShellParams(command=("echo", "hello"), capture_output=False),
            context=build_tool_context(session),
        )

        assert isinstance(result.value, LocalShellResult)
        assert result.value.stdout == "capture disabled"
        assert result.value.stderr == "capture disabled"

    def test_shell_execute_command_not_found(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        handler = find_tool(section, "shell_execute").handler
        assert handler is not None

        with pytest.raises(ToolValidationError, match="not found"):
            handler(
                LocalShellParams(command=("nonexistent-command-12345",)),
                context=build_tool_context(session),
            )


class TestGrepWithGlob:
    def test_grep_with_glob_filter(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        _ = section.ensure_workspace()
        workspace = section._workspace_handle.workspace_path
        (workspace / "test.py").write_text("# TODO: fix", encoding="utf-8")
        (workspace / "test.txt").write_text("# TODO: fix", encoding="utf-8")
        handler = find_tool(section, "grep").handler
        assert handler is not None

        result = handler(
            GrepParams(pattern="TODO", path="", glob="*.py"),
            context=build_tool_context(session),
        )

        assert result.success
        assert isinstance(result.value, tuple)
        assert len(result.value) == 1
        assert result.value[0].path.segments == ("test.py",)


class TestWorkspaceCoverage:
    def test_default_workspace_root_env_override(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        custom_root = tmp_path / "custom-root"
        custom_root.mkdir()
        monkeypatch.setenv("WEAKINCENTIVES_WORKSPACE_ROOT", str(custom_root))

        result = unsafe_local_module._default_workspace_root()
        assert result == custom_root

    def test_default_workspace_root_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Ensure environment variable is not set
        monkeypatch.delenv("WEAKINCENTIVES_WORKSPACE_ROOT", raising=False)

        result = unsafe_local_module._default_workspace_root()
        # Should return the system temp directory
        assert result == Path(tempfile.gettempdir())

    def test_normalize_cwd_empty_after_normalization(self, tmp_path: Path) -> None:
        # Single slash normalizes to empty path which returns workspace root
        result = unsafe_local_module._normalize_cwd("", tmp_path)
        assert result == str(tmp_path)

    def test_normalize_local_eval_code_rejects_control_chars(self) -> None:
        with pytest.raises(ToolValidationError, match="control characters"):
            unsafe_local_module._normalize_local_eval_code("print(\x00)")

    def test_touch_workspace_when_none(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        # Workspace is None before first use - touch should be a no-op
        section._touch_workspace()
        assert section._workspace_handle is None

    def test_teardown_workspace_when_already_none(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        # Workspace is None before first use - teardown should be a no-op
        section._teardown_workspace()
        assert section._workspace_handle is None


class TestVfsCoverage:
    def test_read_file_non_utf8(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        _ = section.ensure_workspace()
        test_file = section._workspace_handle.workspace_path / "binary.dat"
        test_file.write_bytes(b"\xff\xfe\x00\x00")
        handler = find_tool(section, "read_file").handler
        assert handler is not None

        with pytest.raises(ToolValidationError, match="not valid UTF-8"):
            handler(
                ReadFileParams(file_path="binary.dat"),
                context=build_tool_context(session),
            )

    def test_edit_file_non_utf8(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        _ = section.ensure_workspace()
        test_file = section._workspace_handle.workspace_path / "binary.dat"
        test_file.write_bytes(b"\xff\xfe\x00\x00")
        handler = find_tool(section, "edit_file").handler
        assert handler is not None

        with pytest.raises(ToolValidationError, match="not valid UTF-8"):
            handler(
                EditFileParams(
                    file_path="binary.dat", old_string="old", new_string="new"
                ),
                context=build_tool_context(session),
            )

    def test_edit_file_old_string_too_long(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        _ = section.ensure_workspace()
        test_file = section._workspace_handle.workspace_path / "test.txt"
        test_file.write_text("content", encoding="utf-8")
        handler = find_tool(section, "edit_file").handler
        assert handler is not None

        long_string = "x" * 50000
        with pytest.raises(ToolValidationError, match="old_string exceeds"):
            handler(
                EditFileParams(
                    file_path="test.txt", old_string=long_string, new_string="new"
                ),
                context=build_tool_context(session),
            )

    def test_edit_file_new_string_too_long(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        _ = section.ensure_workspace()
        test_file = section._workspace_handle.workspace_path / "test.txt"
        test_file.write_text("old", encoding="utf-8")
        handler = find_tool(section, "edit_file").handler
        assert handler is not None

        long_string = "x" * 50000
        with pytest.raises(ToolValidationError, match="new_string exceeds"):
            handler(
                EditFileParams(
                    file_path="test.txt", old_string="old", new_string=long_string
                ),
                context=build_tool_context(session),
            )


class TestShellEvalCoverage:
    def test_shell_execute_with_stdin(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        handler = find_tool(section, "shell_execute").handler
        assert handler is not None

        result = handler(
            LocalShellParams(command=("cat",), stdin="hello stdin"),
            context=build_tool_context(session),
        )

        assert isinstance(result.value, LocalShellResult)
        assert "hello stdin" in result.value.stdout

    def test_shell_execute_stdin_non_ascii(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        handler = find_tool(section, "shell_execute").handler
        assert handler is not None

        with pytest.raises(ToolValidationError, match="ASCII"):
            handler(
                LocalShellParams(command=("cat",), stdin="héllo"),
                context=build_tool_context(session),
            )

    def test_evaluate_python_failure(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        handler = find_tool(section, "evaluate_python").handler
        assert handler is not None

        result = handler(
            EvalParams(code="import sys; sys.exit(1)"),
            context=build_tool_context(session),
        )

        assert not result.success
        assert isinstance(result.value, EvalResult)
        assert "failed" in result.message.lower()

    def test_grep_with_none_path(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        _ = section.ensure_workspace()
        workspace = section._workspace_handle.workspace_path
        (workspace / "test.txt").write_text("hello", encoding="utf-8")
        handler = find_tool(section, "grep").handler
        assert handler is not None

        # Test with path=None
        result = handler(
            GrepParams(pattern="hello", path=None),
            context=build_tool_context(session),
        )

        assert result.success

    def test_grep_with_empty_glob(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        _ = section.ensure_workspace()
        workspace = section._workspace_handle.workspace_path
        (workspace / "test.txt").write_text("hello", encoding="utf-8")
        handler = find_tool(section, "grep").handler
        assert handler is not None

        # Test with empty glob (whitespace only)
        result = handler(
            GrepParams(pattern="hello", path="", glob="   "),
            context=build_tool_context(session),
        )

        assert result.success

    def test_grep_binary_file_skipped(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        _ = section.ensure_workspace()
        workspace = section._workspace_handle.workspace_path
        binary_file = workspace / "binary.dat"
        binary_file.write_bytes(b"\xff\xfe\x00\x00hello\x00world")
        handler = find_tool(section, "grep").handler
        assert handler is not None

        # Binary file should be skipped (no matches)
        result = handler(
            GrepParams(pattern="hello", path=""),
            context=build_tool_context(session),
        )

        assert result.success
        assert isinstance(result.value, tuple)
        assert len(result.value) == 0

    def test_glob_with_many_files(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        _ = section.ensure_workspace()
        workspace = section._workspace_handle.workspace_path
        # Create multiple files to test glob iteration
        for i in range(10):
            (workspace / f"file{i:03d}.txt").write_text(
                f"content {i}", encoding="utf-8"
            )
        handler = find_tool(section, "glob").handler
        assert handler is not None

        result = handler(
            GlobParams(pattern="*.txt", path=""),
            context=build_tool_context(session),
        )

        assert result.success
        assert isinstance(result.value, tuple)
        assert len(result.value) == 10

    def test_ls_nonexistent_directory_returns_empty(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        _ = section.ensure_workspace()
        handler = find_tool(section, "ls").handler
        assert handler is not None

        # List a non-existent directory returns empty results
        result = handler(
            ListDirectoryParams(path="nonexistent"),
            context=build_tool_context(session),
        )

        assert result.success
        assert isinstance(result.value, tuple)
        assert len(result.value) == 0


class TestRemoveCopyCoverage:
    def test_remove_directory_with_nested_files(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        _ = section.ensure_workspace()
        workspace = section._workspace_handle.workspace_path
        subdir = workspace / "deep"
        subdir.mkdir()
        (subdir / "nested").mkdir()
        (subdir / "file1.txt").write_text("1", encoding="utf-8")
        (subdir / "nested" / "file2.txt").write_text("2", encoding="utf-8")
        handler = find_tool(section, "rm").handler
        assert handler is not None

        result = handler(
            RemoveParams(path="deep"),
            context=build_tool_context(session),
        )

        assert result.success
        assert not subdir.exists()
        # Message should mention multiple entries removed
        assert "2" in result.message or "entries" in result.message.lower()

    def test_read_file_oserror(
        self,
        session_and_bus: tuple[Session, InProcessEventBus],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Test read_file OSError handling (lines 1099-1100)
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        _ = section.ensure_workspace()
        test_file = section._workspace_handle.workspace_path / "test.txt"
        test_file.write_text("content", encoding="utf-8")

        original_read = Path.read_text

        def _raise_read(
            path: Path,
            encoding: str | None = None,
            errors: str | None = None,
        ) -> str:
            if path.name == "test.txt":
                raise OSError("read failed")
            return original_read(path, encoding=encoding, errors=errors)

        monkeypatch.setattr(Path, "read_text", _raise_read)
        handler = find_tool(section, "read_file").handler
        assert handler is not None

        with pytest.raises(ToolValidationError, match="Failed to read"):
            handler(
                ReadFileParams(file_path="test.txt"),
                context=build_tool_context(section.session),
            )

    def test_host_mount_copy_error(
        self,
        session_and_bus: tuple[Session, InProcessEventBus],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import shutil

        session, _bus = session_and_bus
        host_root = tmp_path / "host-root"
        repo = host_root / "repo"
        repo.mkdir(parents=True)
        test_file = repo / "file.txt"
        test_file.write_text("content", encoding="utf-8")

        mount = HostMount(host_path="repo", mount_path=VfsPath(("repo",)))

        original_copy2 = shutil.copy2

        def _raise_copy2(
            src: str | Path, dst: str | Path, *, follow_symlinks: bool = True
        ) -> str:
            if isinstance(src, Path) and src.name == "file.txt":
                raise OSError("copy failed")
            return str(original_copy2(src, dst, follow_symlinks=follow_symlinks))

        monkeypatch.setattr(shutil, "copy2", _raise_copy2)

        with pytest.raises(ToolValidationError, match="materialize"):
            section = _make_section(
                session=session,
                workspace_root=tmp_path / "workspace",
                mounts=(mount,),
                allowed_host_roots=(host_root,),
            )
            section.ensure_workspace()


class TestPathResolutionCoverage:
    def test_format_cwd_display_no_workspace(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)

        # Access shell suite directly to test _format_cwd_display
        shell_suite = unsafe_local_module._LocalShellSuite(section=section)
        result = shell_suite._format_cwd_display("/some/path")

        # Without workspace, should return default
        assert result == "/workspace"

    def test_format_cwd_display_relative_path(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        handle = section.ensure_workspace()

        shell_suite = unsafe_local_module._LocalShellSuite(section=section)
        subdir = handle.workspace_path / "subdir"
        subdir.mkdir()
        result = shell_suite._format_cwd_display(str(subdir))

        assert result == "/workspace/subdir"

    def test_format_cwd_display_outside_workspace(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        _ = section.ensure_workspace()

        shell_suite = unsafe_local_module._LocalShellSuite(section=section)
        outside_path = "/some/other/path"
        result = shell_suite._format_cwd_display(outside_path)

        # Should return the path as-is when outside workspace
        assert result == outside_path

    def test_write_file_creates_parent_directories(
        self, session_and_bus: tuple[Session, InProcessEventBus], tmp_path: Path
    ) -> None:
        session, _bus = session_and_bus
        section = _make_section(session=session, workspace_root=tmp_path)
        handler = find_tool(section, "write_file").handler
        assert handler is not None

        result = handler(
            WriteFileParams(file_path="nested/deep/test.txt", content="hello"),
            context=build_tool_context(session),
        )

        assert result.success
        handle = section._workspace_handle
        assert handle is not None
        written = handle.workspace_path / "nested" / "deep" / "test.txt"
        assert written.read_text(encoding="utf-8") == "hello"

    def test_assert_within_workspace_parent_resolve(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        # Non-existent path within workspace - should resolve parent
        nonexistent = workspace / "nonexistent" / "file.txt"

        # This should work since parent is within workspace
        unsafe_local_module._assert_within_workspace(workspace, nonexistent)

    def test_assert_within_workspace_resolve_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        candidate = workspace / "file.txt"

        # First resolve raises FileNotFoundError, second succeeds
        call_count = [0]
        original_resolve = Path.resolve

        def _resolve_fallback(path: Path, strict: bool = False) -> Path:
            call_count[0] += 1
            if call_count[0] == 1 and path == candidate:
                raise FileNotFoundError("mocked failure")
            return original_resolve(path, strict=strict)

        monkeypatch.setattr(Path, "resolve", _resolve_fallback)
        # Should succeed by falling back to parent resolution
        unsafe_local_module._assert_within_workspace(workspace, candidate)

    def test_assert_within_workspace_parent_also_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        candidate = workspace / "deep" / "file.txt"

        # Both resolve calls raise FileNotFoundError
        def _always_fail(path: Path, strict: bool = False) -> Path:
            del path, strict  # Unused
            raise FileNotFoundError("mocked failure")

        monkeypatch.setattr(Path, "resolve", _always_fail)

        with pytest.raises(ToolValidationError, match="unavailable"):
            unsafe_local_module._assert_within_workspace(workspace, candidate)
