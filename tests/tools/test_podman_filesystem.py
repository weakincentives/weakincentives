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

"""Filesystem integration tests for PodmanSandboxSection."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from subprocess import CompletedProcess
from types import MethodType, SimpleNamespace
from typing import IO, Any, cast

import pytest

import weakincentives.contrib.tools.podman as podman_module
import weakincentives.filesystem._host as filesystem_host_module
import weakincentives.contrib.tools.vfs as vfs_module
from tests.tools.helpers import build_tool_context, find_tool
from tests.tools.podman_test_helpers import (
    ExecResponse,
    FakeCliRunner,
    FakePodmanClient,
    make_section,
    setup_host_mount,
)
from weakincentives import ToolValidationError
from weakincentives.contrib.tools import (
    EditFileParams,
    FileInfo,
    GlobMatch,
    GlobParams,
    GrepMatch,
    GrepParams,
    HostMount,
    ListDirectoryParams,
    PodmanShellParams,
    ReadFileParams,
    ReadFileResult,
    RemoveParams,
    WriteFileParams,
)
from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.session import Session


def test_section_registers_vfs_tool(
    session_and_bus: tuple[Session, InProcessDispatcher], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)

    tool = find_tool(section, "ls")
    assert tool.description.startswith("List directory entries")


def test_host_mount_filesystem_is_populated_at_construction(
    session_and_bus: tuple[Session, InProcessDispatcher], tmp_path: Path
) -> None:
    """Verify mounts are hydrated eagerly during section construction.

    This enables filesystem operations to work before a container starts,
    which is required for workspace digest optimization.
    """
    session, _bus = session_and_bus
    client = FakePodmanClient()
    host_root, mount, file_path = setup_host_mount(tmp_path)
    cache_dir = tmp_path / "cache"
    section = make_section(
        session=session,
        client=client,
        cache_dir=cache_dir,
        mounts=(mount,),
        allowed_host_roots=(host_root,),
    )

    fs = section.filesystem

    # Filesystem should now contain the mounted files
    entries = fs.list(".")
    assert len(entries) == 1
    assert entries[0].name == "sunfish"
    # The mounted file should be readable
    sunfish_entries = fs.list("sunfish")
    assert len(sunfish_entries) == 1
    assert sunfish_entries[0].name == file_path.name


def test_host_mount_populates_prompt_copy(
    session_and_bus: tuple[Session, InProcessDispatcher], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    host_root, mount, _file_path = setup_host_mount(tmp_path)
    section = make_section(
        session=session,
        client=client,
        cache_dir=tmp_path / "cache",
        mounts=(mount,),
        allowed_host_roots=(host_root,),
    )

    assert "Configured host mounts:" in section.template
    assert str(host_root / "sunfish") in section.template


def test_host_mount_materializes_overlay(
    session_and_bus: tuple[Session, InProcessDispatcher], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    host_root, mount, file_path = setup_host_mount(tmp_path)
    cache_dir = tmp_path / "cache"
    section = make_section(
        session=session,
        client=client,
        cache_dir=cache_dir,
        mounts=(mount,),
        allowed_host_roots=(host_root,),
    )
    handler = find_tool(section, "shell_execute").handler
    assert handler is not None

    handler(
        PodmanShellParams(command=("true",)),
        context=build_tool_context(session, filesystem=section.filesystem),
    )

    handle = section._workspace_handle
    assert handle is not None
    mounted = handle.overlay_path / "sunfish" / file_path.name
    assert mounted.read_text(encoding="utf-8") == file_path.read_text(encoding="utf-8")


def test_preview_mount_entries_handles_file(tmp_path: Path) -> None:
    file_path = tmp_path / "item.txt"
    file_path.write_text("payload", encoding="utf-8")
    result = podman_module._preview_mount_entries(file_path)
    assert result == ("item.txt",)


def test_preview_mount_entries_raises_on_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
        podman_module._preview_mount_entries(directory)


def test_iter_host_mount_files_handles_file(tmp_path: Path) -> None:
    file_path = tmp_path / "item.txt"
    file_path.write_text("payload", encoding="utf-8")
    entries = tuple(podman_module._iter_host_mount_files(file_path, False))
    assert entries == (file_path,)


def test_host_mount_allows_binary_files(
    session_and_bus: tuple[Session, InProcessDispatcher], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    host_root = tmp_path / "host-root"
    repo = host_root / "sunfish"
    repo.mkdir(parents=True, exist_ok=True)
    file_path = repo / "payload.bin"
    payload = b"\x00\xffbinary\x01"
    file_path.write_bytes(payload)
    cache_dir = tmp_path / "cache"
    section = make_section(
        session=session,
        client=client,
        cache_dir=cache_dir,
        mounts=(
            HostMount(host_path="sunfish", mount_path=vfs_module.VfsPath(("sunfish",))),
        ),
        allowed_host_roots=(host_root,),
    )
    handler = find_tool(section, "shell_execute").handler
    assert handler is not None

    handler(
        PodmanShellParams(command=("true",)),
        context=build_tool_context(session, filesystem=section.filesystem),
    )

    handle = section._workspace_handle
    assert handle is not None
    mounted = handle.overlay_path / "sunfish" / file_path.name
    assert mounted.read_bytes() == payload


def test_host_mount_hydration_skips_existing_overlay(
    session_and_bus: tuple[Session, InProcessDispatcher], tmp_path: Path
) -> None:
    """Verify _hydrate_overlay_mounts is a no-op when overlay is non-empty.

    Note: Mounts are now hydrated eagerly during section construction, so
    this test verifies that re-calling _hydrate_overlay_mounts on an
    already-populated overlay doesn't duplicate files.
    """
    session, _bus = session_and_bus
    client = FakePodmanClient()
    host_root, mount, file_path = setup_host_mount(tmp_path)
    cache_dir = tmp_path / "cache"
    section = make_section(
        session=session,
        client=client,
        cache_dir=cache_dir,
        mounts=(mount,),
        allowed_host_roots=(host_root,),
    )
    overlay = section._workspace_overlay_path()
    # Mounts are now hydrated during __init__, so the file should exist
    mounted = overlay / "sunfish" / file_path.name
    assert mounted.exists()

    # Add a placeholder file
    placeholder = overlay / "existing.txt"
    placeholder.write_text("keep", encoding="utf-8")

    # Get the mtime of the mounted file before re-hydration
    mtime_before = mounted.stat().st_mtime

    # Re-calling _hydrate_overlay_mounts should be a no-op since overlay is non-empty
    section._hydrate_overlay_mounts(overlay)

    # Verify: placeholder preserved, mounted file unchanged
    assert placeholder.read_text(encoding="utf-8") == "keep"
    assert mounted.stat().st_mtime == mtime_before


def test_host_mount_hydration_raises_on_write_error(
    session_and_bus: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify hydration raises ToolValidationError on copy failure.

    Note: Mounts are now hydrated eagerly during section construction,
    so we must patch shutil.copy2 before creating the section.
    """
    session, _bus = session_and_bus
    client = FakePodmanClient()
    host_root, mount, file_path = setup_host_mount(tmp_path)
    cache_dir = tmp_path / "cache"
    # Compute expected target path before section creation
    overlay = cache_dir / str(session.session_id)
    target = overlay / "sunfish" / file_path.name
    original_copy = podman_module.shutil.copy2

    def _fail_on_target(
        src: Path,
        dst: Path,
        *,
        follow_symlinks: bool = True,
    ) -> object:
        if dst == target:
            raise OSError("boom")
        return original_copy(src, dst, follow_symlinks=follow_symlinks)

    # Patch before section creation since hydration happens in __init__
    monkeypatch.setattr(podman_module.shutil, "copy2", _fail_on_target)

    with pytest.raises(ToolValidationError):
        make_section(
            session=session,
            client=client,
            cache_dir=cache_dir,
            mounts=(mount,),
            allowed_host_roots=(host_root,),
        )


def test_ls_lists_workspace_files(
    session_and_bus: tuple[Session, InProcessDispatcher], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    docs = handle.overlay_path / "docs"
    docs.mkdir(parents=True)
    (docs / "README.md").write_text("hello world", encoding="utf-8")

    tool = find_tool(section, "ls")
    handler = tool.handler
    assert handler is not None
    result = handler(
        ListDirectoryParams(path="docs"),
        context=build_tool_context(session, filesystem=section.filesystem),
    )
    assert result.value is not None
    entries = cast(tuple[FileInfo, ...], result.value)
    assert any(entry.path.segments == ("docs", "README.md") for entry in entries)


def test_read_file_returns_numbered_lines(
    session_and_bus: tuple[Session, InProcessDispatcher], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    target = handle.overlay_path / "notes.txt"
    target.write_text("first\nsecond\nthird", encoding="utf-8")

    tool = find_tool(section, "read_file")
    handler = tool.handler
    assert handler is not None
    result = handler(
        ReadFileParams(file_path="notes.txt", limit=2),
        context=build_tool_context(session, filesystem=section.filesystem),
    )
    assert result.value is not None
    read_result = cast(ReadFileResult, result.value)
    assert "1 | first" in read_result.content
    assert "2 | second" in read_result.content


def test_write_via_container_appends_existing_content(
    session_and_bus: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    target = handle.overlay_path / "script.sh"
    target.write_text("base", encoding="utf-8")
    exec_calls: list[list[str]] = []
    cp_payloads: list[str] = []

    def _fake_exec(
        self: podman_module.PodmanSandboxSection, *, config: podman_module._ExecConfig
    ) -> CompletedProcess[str]:
        del self
        _ = (
            config.stdin,
            config.environment,
            config.timeout,
            config.capture_output,
        )
        exec_calls.append(list(config.command))
        return CompletedProcess(config.command, 0, stdout="", stderr="")

    def _fake_cp(
        self: podman_module.PodmanSandboxSection,
        *,
        source: str,
        destination: str,
        timeout: float | None = None,
    ) -> CompletedProcess[str]:
        del destination, timeout
        cp_payloads.append(Path(source).read_text(encoding="utf-8"))
        return CompletedProcess(["podman", "cp"], 0, stdout="", stderr="")

    monkeypatch.setattr(section, "run_cli_exec", MethodType(_fake_exec, section))
    monkeypatch.setattr(section, "run_cli_cp", MethodType(_fake_cp, section))

    section.write_via_container(
        path=vfs_module.VfsPath(("script.sh",)),
        content="+",
        mode="append",
    )

    assert exec_calls[-1] == ["mkdir", "-p", "/workspace"]
    assert cp_payloads[-1] == "base+"


def test_write_via_container_reports_mkdir_failure(
    session_and_bus: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)
    section.ensure_workspace()
    cp_calls = 0

    def _fail_exec(
        self: podman_module.PodmanSandboxSection, *, config: podman_module._ExecConfig
    ) -> CompletedProcess[str]:
        del self
        _ = (
            config.stdin,
            config.environment,
            config.timeout,
            config.capture_output,
        )
        return CompletedProcess(list(config.command), 1, stdout="", stderr="boom")

    def _fake_cp(
        self: podman_module.PodmanSandboxSection,
        *,
        source: str,
        destination: str,
        timeout: float | None = None,
    ) -> CompletedProcess[str]:
        nonlocal cp_calls
        cp_calls += 1
        return CompletedProcess(["podman", "cp"], 0, stdout="", stderr="")

    monkeypatch.setattr(section, "run_cli_exec", MethodType(_fail_exec, section))
    monkeypatch.setattr(section, "run_cli_cp", MethodType(_fake_cp, section))

    with pytest.raises(ToolValidationError):
        section.write_via_container(
            path=vfs_module.VfsPath(("nested", "file.txt")),
            content="data",
            mode="create",
        )

    assert cp_calls == 0


def test_write_via_container_rejects_non_utf8_append(
    session_and_bus: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    target = handle.overlay_path / "data.bin"
    target.write_bytes(b"\xff\xff")

    def _unexpected(
        self: podman_module.PodmanSandboxSection, *, config: podman_module._ExecConfig
    ) -> CompletedProcess[str]:
        raise AssertionError("run_cli_exec should not be called")

    def _unexpected_cp(
        self: podman_module.PodmanSandboxSection,
        *,
        source: str,
        destination: str,
        timeout: float | None = None,
    ) -> CompletedProcess[str]:
        raise AssertionError("run_cli_cp should not be called")

    monkeypatch.setattr(section, "run_cli_exec", MethodType(_unexpected, section))
    monkeypatch.setattr(section, "run_cli_cp", MethodType(_unexpected_cp, section))

    with pytest.raises(ToolValidationError):
        section.write_via_container(
            path=vfs_module.VfsPath(("data.bin",)),
            content="payload",
            mode="append",
        )


def test_write_via_container_propagates_read_oserror(
    session_and_bus: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    target = handle.overlay_path / "data.txt"
    target.write_text("payload", encoding="utf-8")
    original_read_text = Path.read_text

    def _raise(
        self: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        if self == target:
            raise OSError("boom")
        return original_read_text(self, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", _raise)

    def _unexpected(
        self: podman_module.PodmanSandboxSection, *, config: podman_module._ExecConfig
    ) -> CompletedProcess[str]:
        raise AssertionError("run_cli_exec should not be called")

    def _unexpected_cp(
        self: podman_module.PodmanSandboxSection,
        *,
        source: str,
        destination: str,
        timeout: float | None = None,
    ) -> CompletedProcess[str]:
        raise AssertionError("run_cli_cp should not be called")

    monkeypatch.setattr(section, "run_cli_exec", MethodType(_unexpected, section))
    monkeypatch.setattr(section, "run_cli_cp", MethodType(_unexpected_cp, section))

    with pytest.raises(ToolValidationError):
        section.write_via_container(
            path=vfs_module.VfsPath(("data.txt",)),
            content="payload",
            mode="append",
        )


def test_glob_matches_files(
    session_and_bus: tuple[Session, InProcessDispatcher], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    src = handle.overlay_path / "src"
    src.mkdir()
    (src / "main.py").write_text("print('hi')", encoding="utf-8")
    (src / "README.md").write_text("details", encoding="utf-8")
    tool = find_tool(section, "glob")
    handler = tool.handler
    assert handler is not None
    result = handler(
        GlobParams(pattern="*.py", path="src"),
        context=build_tool_context(session, filesystem=section.filesystem),
    )
    assert result.value is not None
    matches = cast(tuple[GlobMatch, ...], result.value)
    assert any(match.path.segments == ("src", "main.py") for match in matches)


def test_grep_finds_pattern(
    session_and_bus: tuple[Session, InProcessDispatcher], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    target = handle.overlay_path / "log.txt"
    target.write_text("first\nmatch line\nlast", encoding="utf-8")
    tool = find_tool(section, "grep")
    handler = tool.handler
    assert handler is not None
    result = handler(
        GrepParams(pattern="match", path="/", glob=None),
        context=build_tool_context(session, filesystem=section.filesystem),
    )
    assert result.value is not None
    grep_matches = cast(tuple[GrepMatch, ...], result.value)
    assert any(match.line_number == 2 for match in grep_matches)


def test_ls_rejects_file_path(
    session_and_bus: tuple[Session, InProcessDispatcher], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    target = handle.overlay_path / "file.txt"
    target.write_text("data", encoding="utf-8")
    tool = find_tool(section, "ls")
    handler = tool.handler
    assert handler is not None

    with pytest.raises(ToolValidationError):
        handler(
            ListDirectoryParams(path="file.txt"),
            context=build_tool_context(session, filesystem=section.filesystem),
        )


def test_read_file_missing_path(
    session_and_bus: tuple[Session, InProcessDispatcher], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)
    tool = find_tool(section, "read_file")
    handler = tool.handler
    assert handler is not None

    with pytest.raises(ToolValidationError):
        handler(
            ReadFileParams(file_path="missing.txt"),
            context=build_tool_context(session, filesystem=section.filesystem),
        )


def test_read_file_rejects_invalid_encoding(
    session_and_bus: tuple[Session, InProcessDispatcher], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    target = handle.overlay_path / "binary.bin"
    target.write_bytes(b"\xff")
    tool = find_tool(section, "read_file")
    handler = tool.handler
    assert handler is not None

    with pytest.raises(ToolValidationError):
        handler(
            ReadFileParams(file_path="binary.bin"),
            context=build_tool_context(session, filesystem=section.filesystem),
        )


def test_write_file_rejects_existing_file(
    session_and_bus: tuple[Session, InProcessDispatcher], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    runner = FakeCliRunner([ExecResponse(exit_code=0)])
    section = make_section(
        session=session,
        client=client,
        cache_dir=tmp_path,
        runner=runner,
    )
    handle = section.ensure_workspace()
    target = handle.overlay_path / "file.txt"
    target.write_text("data", encoding="utf-8")
    tool = find_tool(section, "write_file")
    handler = tool.handler
    assert handler is not None

    with pytest.raises(ToolValidationError):
        handler(
            WriteFileParams(file_path="file.txt", content="other"),
            context=build_tool_context(session, filesystem=section.filesystem),
        )


def test_edit_file_rejects_long_strings(
    session_and_bus: tuple[Session, InProcessDispatcher], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)
    tool = find_tool(section, "edit_file")
    handler = tool.handler
    assert handler is not None
    long_text = "x" * (vfs_module.MAX_WRITE_LENGTH + 1)

    with pytest.raises(ToolValidationError):
        handler(
            EditFileParams(
                file_path="file.txt",
                old_string=long_text,
                new_string="short",
            ),
            context=build_tool_context(session, filesystem=section.filesystem),
        )

    with pytest.raises(ToolValidationError):
        handler(
            EditFileParams(
                file_path="file.txt",
                old_string="short",
                new_string=long_text,
            ),
            context=build_tool_context(session, filesystem=section.filesystem),
        )


def test_edit_file_missing_path(
    session_and_bus: tuple[Session, InProcessDispatcher], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)
    tool = find_tool(section, "edit_file")
    handler = tool.handler
    assert handler is not None

    with pytest.raises(ToolValidationError):
        handler(
            EditFileParams(
                file_path="file.txt",
                old_string="a",
                new_string="b",
            ),
            context=build_tool_context(session, filesystem=section.filesystem),
        )


def test_edit_file_rejects_invalid_encoding(
    session_and_bus: tuple[Session, InProcessDispatcher], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    runner = FakeCliRunner([ExecResponse(exit_code=0)])
    section = make_section(
        session=session,
        client=client,
        cache_dir=tmp_path,
        runner=runner,
    )
    handle = section.ensure_workspace()
    target = handle.overlay_path / "file.txt"
    target.write_bytes(b"\xff")
    tool = find_tool(section, "edit_file")
    handler = tool.handler
    assert handler is not None

    with pytest.raises(ToolValidationError):
        handler(
            EditFileParams(
                file_path="file.txt",
                old_string="a",
                new_string="b",
            ),
            context=build_tool_context(session, filesystem=section.filesystem),
        )


def test_edit_file_requires_occurrence(
    session_and_bus: tuple[Session, InProcessDispatcher], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    runner = FakeCliRunner([ExecResponse(exit_code=0)])
    section = make_section(
        session=session,
        client=client,
        cache_dir=tmp_path,
        runner=runner,
    )
    handle = section.ensure_workspace()
    target = handle.overlay_path / "file.txt"
    target.write_text("hello", encoding="utf-8")
    tool = find_tool(section, "edit_file")
    handler = tool.handler
    assert handler is not None

    with pytest.raises(ToolValidationError):
        handler(
            EditFileParams(
                file_path="file.txt",
                old_string="missing",
                new_string="new",
            ),
            context=build_tool_context(session, filesystem=section.filesystem),
        )


def test_edit_file_requires_unique_match(
    session_and_bus: tuple[Session, InProcessDispatcher], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    runner = FakeCliRunner([ExecResponse(exit_code=0)])
    section = make_section(
        session=session,
        client=client,
        cache_dir=tmp_path,
        runner=runner,
    )
    handle = section.ensure_workspace()
    target = handle.overlay_path / "file.txt"
    target.write_text("foo foo", encoding="utf-8")
    tool = find_tool(section, "edit_file")
    handler = tool.handler
    assert handler is not None

    with pytest.raises(ToolValidationError):
        handler(
            EditFileParams(
                file_path="file.txt",
                old_string="foo",
                new_string="bar",
            ),
            context=build_tool_context(session, filesystem=section.filesystem),
        )


def test_edit_file_replace_all_branch(
    session_and_bus: tuple[Session, InProcessDispatcher], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    runner = FakeCliRunner([ExecResponse(exit_code=0)])
    section = make_section(
        session=session,
        client=client,
        cache_dir=tmp_path,
        runner=runner,
    )
    handle = section.ensure_workspace()
    target = handle.overlay_path / "file.txt"
    target.write_text("foo foo", encoding="utf-8")
    tool = find_tool(section, "edit_file")
    handler = tool.handler
    assert handler is not None

    handler(
        EditFileParams(
            file_path="file.txt",
            old_string="foo",
            new_string="bar",
            replace_all=True,
        ),
        context=build_tool_context(session, filesystem=section.filesystem),
    )


def test_glob_rejects_empty_pattern(
    session_and_bus: tuple[Session, InProcessDispatcher], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)
    tool = find_tool(section, "glob")
    handler = tool.handler
    assert handler is not None

    with pytest.raises(ToolValidationError):
        handler(
            GlobParams(pattern="   ", path="/"),
            context=build_tool_context(session, filesystem=section.filesystem),
        )


def test_grep_rejects_invalid_regex(
    session_and_bus: tuple[Session, InProcessDispatcher], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)
    tool = find_tool(section, "grep")
    handler = tool.handler
    assert handler is not None

    result = handler(
        GrepParams(pattern="[", path="/", glob=None),
        context=build_tool_context(session, filesystem=section.filesystem),
    )
    assert not result.success
    assert result.value is None
    assert "Invalid regular expression" in result.message


def test_grep_honors_glob_argument(
    session_and_bus: tuple[Session, InProcessDispatcher], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    (handle.overlay_path / "main.py").write_text("match", encoding="utf-8")
    (handle.overlay_path / "notes.txt").write_text("match", encoding="utf-8")
    tool = find_tool(section, "grep")
    handler = tool.handler
    assert handler is not None

    result = handler(
        GrepParams(pattern="match", path="/", glob="*.py"),
        context=build_tool_context(session, filesystem=section.filesystem),
    )
    assert result.value is not None
    matches = cast(tuple[GrepMatch, ...], result.value)
    assert matches[0].path.segments == ("main.py",)
    assert len(matches) == 1


def test_grep_supports_default_path(
    session_and_bus: tuple[Session, InProcessDispatcher], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    (handle.overlay_path / "main.txt").write_text("match", encoding="utf-8")
    tool = find_tool(section, "grep")
    handler = tool.handler
    assert handler is not None

    result = handler(
        GrepParams(pattern="match", path=None, glob=None),
        context=build_tool_context(session, filesystem=section.filesystem),
    )

    assert result.value is not None
    matches = cast(tuple[GrepMatch, ...], result.value)
    assert matches == (
        GrepMatch(path=vfs_module.VfsPath(("main.txt",)), line_number=1, line="match"),
    )


def test_grep_respects_glob_filter(
    session_and_bus: tuple[Session, InProcessDispatcher], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    (handle.overlay_path / "notes.txt").write_text("match", encoding="utf-8")
    tool = find_tool(section, "grep")
    handler = tool.handler
    assert handler is not None

    result = handler(
        GrepParams(pattern="match", path="/", glob="*.py"),
        context=build_tool_context(session, filesystem=section.filesystem),
    )
    assert result.value == ()


def test_grep_ignores_blank_glob(
    session_and_bus: tuple[Session, InProcessDispatcher], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    (handle.overlay_path / "notes.txt").write_text("match", encoding="utf-8")
    tool = find_tool(section, "grep")
    handler = tool.handler
    assert handler is not None

    result = handler(
        GrepParams(pattern="match", path="/", glob="  "),
        context=build_tool_context(session, filesystem=section.filesystem),
    )

    assert result.value is not None
    matches = cast(tuple[GrepMatch, ...], result.value)
    assert matches == (
        GrepMatch(path=vfs_module.VfsPath(("notes.txt",)), line_number=1, line="match"),
    )


def test_grep_skips_invalid_file_encoding(
    session_and_bus: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    # Write invalid UTF-8 bytes to trigger encoding error during grep
    target = handle.overlay_path / "binary.bin"
    target.write_bytes(b"\xff\xfe invalid utf-8")
    tool = find_tool(section, "grep")
    handler = tool.handler
    assert handler is not None

    result = handler(
        GrepParams(pattern="match", path="/", glob=None),
        context=build_tool_context(session, filesystem=section.filesystem),
    )
    # Binary file should be skipped, resulting in no matches
    assert result.value == ()


def test_grep_handles_oserror(
    monkeypatch: pytest.MonkeyPatch,
    session_and_bus: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    target = handle.overlay_path / "main.txt"
    target.write_text("match", encoding="utf-8")
    secondary = handle.overlay_path / "other.txt"
    secondary.write_text("match", encoding="utf-8")
    original_open = Path.open

    def _fake_open(self: Path, *args: object, **kwargs: object) -> IO[Any]:
        if self == target:
            raise OSError("boom")
        return cast(IO[Any], original_open(self, *args, **kwargs))  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "open", _fake_open)
    tool = find_tool(section, "grep")
    handler = tool.handler
    assert handler is not None

    result = handler(
        GrepParams(pattern="match", path="/", glob=None),
        context=build_tool_context(session, filesystem=section.filesystem),
    )
    assert result.value is not None
    matches = cast(tuple[GrepMatch, ...], result.value)
    assert matches == (
        GrepMatch(path=vfs_module.VfsPath(("other.txt",)), line_number=1, line="match"),
    )


def test_grep_honors_result_limit(
    monkeypatch: pytest.MonkeyPatch,
    session_and_bus: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    (handle.overlay_path / "one.txt").write_text("match", encoding="utf-8")
    (handle.overlay_path / "two.txt").write_text("match", encoding="utf-8")
    monkeypatch.setattr(filesystem_host_module, "MAX_GREP_MATCHES", 1)
    tool = find_tool(section, "grep")
    handler = tool.handler
    assert handler is not None

    result = handler(
        GrepParams(pattern="match", path="/", glob=None),
        context=build_tool_context(session, filesystem=section.filesystem),
    )
    assert result.value is not None
    grep_matches = cast(tuple[GrepMatch, ...], result.value)
    assert len(grep_matches) == 1


def test_grep_skips_binary_and_collects_match(
    session_and_bus: tuple[Session, InProcessDispatcher], tmp_path: Path
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)
    handle = section.ensure_workspace()
    (handle.overlay_path / "binary.bin").write_bytes(b"\xff\xfe\x00")
    (handle.overlay_path / "valid.txt").write_text("line with match", encoding="utf-8")
    tool = find_tool(section, "grep")
    handler = tool.handler
    assert handler is not None

    result = handler(
        GrepParams(pattern="match", path="/", glob=None),
        context=build_tool_context(session, filesystem=section.filesystem),
    )

    assert result.value is not None
    matches = cast(tuple[GrepMatch, ...], result.value)
    assert matches == (
        GrepMatch(
            path=vfs_module.VfsPath(("valid.txt",)),
            line_number=1,
            line="line with match",
        ),
    )


def test_remove_rejects_root_and_missing(
    monkeypatch: pytest.MonkeyPatch,
    session_and_bus: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)
    tool = find_tool(section, "rm")
    handler = tool.handler
    assert handler is not None

    with pytest.raises(ToolValidationError):
        handler(
            RemoveParams(path="/"),
            context=build_tool_context(session, filesystem=section.filesystem),
        )

    with pytest.raises(ToolValidationError):
        handler(
            RemoveParams(path="missing.txt"),
            context=build_tool_context(session, filesystem=section.filesystem),
        )

    original = vfs_module.normalize_string_path

    def _fake(
        path: str | None,
        *,
        allow_empty: bool = False,
        field: str,
        mount_point: str | None = None,
    ) -> vfs_module.VfsPath:
        if path == "trigger":
            return vfs_module.VfsPath(())
        return original(
            path, allow_empty=allow_empty, field=field, mount_point=mount_point
        )

    monkeypatch.setattr(vfs_module, "normalize_string_path", _fake)
    with pytest.raises(ToolValidationError):
        handler(
            RemoveParams(path="trigger"),
            context=build_tool_context(session, filesystem=section.filesystem),
        )


def test_write_via_container_handles_cli_failures(
    monkeypatch: pytest.MonkeyPatch,
    session_and_bus: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
) -> None:
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)
    path = vfs_module.VfsPath(("file.txt",))

    def _raise(*_: object, **__: object) -> None:
        raise FileNotFoundError("podman")

    monkeypatch.setattr(section, "run_cli_cp", _raise)
    with pytest.raises(ToolValidationError):
        section.write_via_container(path=path, content="data", mode="create")

    failure = SimpleNamespace(returncode=1, stdout="", stderr="boom")
    monkeypatch.setattr(section, "run_cli_cp", lambda **__: failure)
    with pytest.raises(ToolValidationError):
        section.write_via_container(path=path, content="data", mode="create")


def test_write_via_container_skips_mkdir_for_root_level_file(
    monkeypatch: pytest.MonkeyPatch,
    session_and_bus: tuple[Session, InProcessDispatcher],
    tmp_path: Path,
) -> None:
    """Test branch 1285->1296: skip mkdir when parent is '/' (empty VfsPath)."""
    session, _bus = session_and_bus
    client = FakePodmanClient()
    section = make_section(session=session, client=client, cache_dir=tmp_path)

    # Track if mkdir is called
    mkdir_called = False

    def _fake_exec(self: object, *, config: object) -> CompletedProcess[str]:
        nonlocal mkdir_called
        mkdir_called = True
        return CompletedProcess([], 0, stdout="", stderr="")

    def _fake_cp(
        self: object,
        *,
        source: str,
        destination: str,
        timeout: float | None = None,
    ) -> CompletedProcess[str]:
        del source, destination, timeout
        return CompletedProcess(["podman", "cp"], 0, stdout="", stderr="")

    monkeypatch.setattr(section, "run_cli_exec", MethodType(_fake_exec, section))
    monkeypatch.setattr(section, "run_cli_cp", MethodType(_fake_cp, section))

    # First: test with a single-segment path - parent is "/work", mkdir SHOULD be called
    section.write_via_container(
        path=vfs_module.VfsPath(("file.txt",)),
        content="data",
        mode="create",
    )
    assert mkdir_called, "mkdir should be called for file.txt (parent=/work)"

    # Reset and test with empty VfsPath - parent is "/", mkdir should be skipped
    mkdir_called = False
    # Empty VfsPath results in container_path="/work", parent="/"
    # This skips the mkdir block (branch 1285->1296)
    section.write_via_container(
        path=vfs_module.VfsPath(()),
        content="data",
        mode="create",
    )
    assert not mkdir_called, "mkdir should be skipped for empty VfsPath (parent=/)"


def test_tools_module_missing_attr_raises() -> None:
    from weakincentives.contrib import tools

    with pytest.raises(AttributeError):
        _ = tools.TOTALLY_UNKNOWN
