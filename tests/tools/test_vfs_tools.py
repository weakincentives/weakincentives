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

"""Unit tests for the virtual filesystem tools."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import IO, Any, cast

import pytest

import weakincentives.tools.vfs as vfs_module
from tests.tools.helpers import find_tool, invoke_tool
from weakincentives.adapters.core import SessionProtocol
from weakincentives.prompt.tool import ToolContext, ToolResult
from weakincentives.runtime.events import InProcessEventBus
from weakincentives.runtime.session import ReducerEvent, Session, select_latest
from weakincentives.tools import (
    DeleteFile,
    FileReadResult,
    HostMount,
    ListDirectory,
    ListDirectoryResult,
    ReadFile,
    ToolValidationError,
    VfsFile,
    VfsPath,
    VfsToolsSection,
    VirtualFileSystem,
    WriteFile,
)


def _read_disk_bytes(snapshot: VirtualFileSystem, path: VfsPath) -> bytes:
    return Path(snapshot.root_path).joinpath(*path.segments).read_bytes()


def test_write_file_creates_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = find_tool(section, "write_file")

    params = WriteFile(path=VfsPath(("docs", "intro.md")), content=b"hello world")
    invoke_tool(bus, write_tool, params, session=session)

    snapshot = select_latest(session, VirtualFileSystem)
    assert snapshot is not None
    assert (
        Path(snapshot.root_path) / "docs" / "intro.md"
    ).read_bytes() == b"hello world"
    assert snapshot.files == (
        VfsFile(
            path=VfsPath(("docs", "intro.md")),
            encoding="utf-8",
            size_bytes=len(b"hello world"),
            version=1,
            created_at=timestamp,
            updated_at=timestamp,
        ),
    )


def test_requires_session_in_context(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = find_tool(section, "write_file")

    params = WriteFile(path=VfsPath(("docs", "intro.md")), content=b"hello world")
    handler = write_tool.handler
    assert handler is not None
    context = ToolContext(
        prompt=None,
        rendered_prompt=None,
        adapter=None,
        session=cast(SessionProtocol, object()),
        event_bus=bus,
    )
    with pytest.raises(ToolValidationError, match="Session instance"):
        handler(params, context=context)


def test_rejects_mismatched_session(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    primary = Session(bus=bus)
    secondary = Session(bus=bus)
    section = VfsToolsSection(session=primary)
    write_tool = find_tool(section, "write_file")

    params = WriteFile(path=VfsPath(("docs", "intro.md")), content=b"hello world")
    handler = write_tool.handler
    assert handler is not None
    context = ToolContext(
        prompt=None,
        rendered_prompt=None,
        adapter=None,
        session=secondary,
        event_bus=bus,
    )
    with pytest.raises(
        ToolValidationError, match="ToolContext for a different session"
    ):
        handler(params, context=context)


def test_write_file_appends(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamps = iter(
        (
            datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            datetime(2024, 1, 1, 12, 1, tzinfo=UTC),
        )
    )
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: next(timestamps))

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = find_tool(section, "write_file")

    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("log.txt",)), content=b"start"),
        session=session,
    )
    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("log.txt",)), content=b" -> next", mode="append"),
        session=session,
    )

    snapshot = select_latest(session, VirtualFileSystem)
    assert snapshot is not None
    file = snapshot.files[0]
    assert _read_disk_bytes(snapshot, file.path) == b"start -> next"
    assert file.version == 2
    assert file.size_bytes == len(b"start -> next")


def test_delete_directory_removes_nested(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = find_tool(section, "write_file")
    delete_tool = find_tool(section, "delete_file")

    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("src", "main.py")), content=b"print('hello')"),
        session=session,
    )
    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("src", "utils", "helpers.py")), content=b"# helpers"),
        session=session,
    )

    invoke_tool(bus, delete_tool, DeleteFile(path=VfsPath(("src",))), session=session)

    snapshot = select_latest(session, VirtualFileSystem)
    assert snapshot is not None
    assert snapshot.files == ()


def test_mount_snapshot_persisted_in_session(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    sunfish = root / "sunfish"
    sunfish.mkdir()
    readme = sunfish / "README.md"
    readme.write_text("hello mount", encoding="utf-8")

    bus = InProcessEventBus()
    session = Session(bus=bus)
    mount = HostMount(host_path="sunfish", mount_path=VfsPath(("sunfish",)))
    section = VfsToolsSection(
        session=session,
        mounts=(mount,),
        allowed_host_roots=(root,),
    )
    list_tool = find_tool(section, "list_directory")
    invoke_tool(bus, list_tool, ListDirectory(), session=session)

    snapshot = select_latest(session, VirtualFileSystem)
    assert snapshot is not None
    files_by_path = {file.path.segments: file for file in snapshot.files}
    assert ("sunfish", "README.md") in files_by_path
    mounted = files_by_path["sunfish", "README.md"]
    assert _read_disk_bytes(snapshot, mounted.path) == b"hello mount"


def test_mount_permission_error_translated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    restricted = root / "restricted.txt"
    restricted.write_text("secret", encoding="utf-8")

    original_read_bytes = Path.read_bytes

    def fake_read_bytes(self: Path) -> bytes:
        if self == restricted:
            raise PermissionError("access denied")
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", fake_read_bytes)

    mount = HostMount(host_path="restricted.txt", mount_path=VfsPath(("restricted",)))
    bus = InProcessEventBus()
    session = Session(bus=bus)

    with pytest.raises(ToolValidationError, match=str(restricted)):
        VfsToolsSection(
            session=session,
            mounts=(mount,),
            allowed_host_roots=(root,),
        )


def test_list_directory_shows_children(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = find_tool(section, "write_file")
    list_tool = find_tool(section, "list_directory")

    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("workspace", "notes.md")), content=b"notes"),
        session=session,
    )
    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("workspace", "drafts", "todo.md")), content=b"todo"),
        session=session,
    )

    result = invoke_tool(
        bus, list_tool, ListDirectory(path=VfsPath(("workspace",))), session=session
    )
    assert result.value == ListDirectoryResult(
        path=VfsPath(("workspace",)),
        directories=("drafts",),
        files=("notes.md",),
    )


def test_read_file_returns_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = find_tool(section, "write_file")
    read_tool = find_tool(section, "read_file")

    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("README.md",)), content=b"intro"),
        session=session,
    )

    result = invoke_tool(
        bus, read_tool, ReadFile(path=VfsPath(("README.md",))), session=session
    )
    file = result.value
    assert isinstance(file, FileReadResult)
    assert file.content == b"intro"
    assert file.file.version == 1
    assert file.file.created_at == timestamp
    assert file.file.updated_at == timestamp


def test_read_file_missing_disk_entry_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = find_tool(section, "write_file")
    read_tool = find_tool(section, "read_file")

    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("README.md",)), content=b"intro"),
        session=session,
    )
    snapshot = select_latest(session, VirtualFileSystem)
    assert snapshot is not None
    missing = Path(snapshot.root_path) / "README.md"
    missing.unlink()

    with pytest.raises(ToolValidationError, match="disappeared"):
        invoke_tool(
            bus, read_tool, ReadFile(path=VfsPath(("README.md",))), session=session
        )


def test_read_file_disk_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = find_tool(section, "write_file")
    read_tool = find_tool(section, "read_file")

    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("README.md",)), content=b"intro"),
        session=session,
    )
    original_read_bytes = Path.read_bytes

    def fail_read_bytes(self: Path) -> bytes:
        if self.name == "README.md":
            raise OSError("boom")
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)
    with pytest.raises(ToolValidationError, match="Failed to read VFS file"):
        invoke_tool(
            bus, read_tool, ReadFile(path=VfsPath(("README.md",))), session=session
        )


def test_write_file_rejects_invalid_path(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = find_tool(section, "write_file")
    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            write_tool,
            WriteFile(path=VfsPath(("/etc", "passwd")), content=b"x"),
            session=session,
        )


def test_host_mount_materialises_files(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    root = tmp_path_factory.mktemp("source")
    docs = root / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("guide", encoding="utf-8")
    (docs / "skip.bin").write_text("binary", encoding="utf-8")

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(
        session=session,
        mounts=(
            HostMount(
                host_path="docs",
                mount_path=VfsPath(("reference",)),
                include_glob=("*.md",),
            ),
        ),
        allowed_host_roots=(root,),
    )
    read_tool = find_tool(section, "read_file")

    result = invoke_tool(
        bus,
        read_tool,
        ReadFile(path=VfsPath(("reference", "guide.md"))),
        session=session,
    )
    file = result.value
    assert isinstance(file, FileReadResult)
    assert file.content == b"guide"
    assert file.file == VfsFile(
        path=VfsPath(("reference", "guide.md")),
        encoding="utf-8",
        size_bytes=len(b"guide"),
        version=1,
        created_at=timestamp,
        updated_at=timestamp,
    )


def test_host_mount_copy_failure_propagates(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path_factory.mktemp("workspace")
    docs = root / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("guide", encoding="utf-8")

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(
        session=session,
        mounts=(HostMount(host_path="docs"),),
        allowed_host_roots=(root,),
    )
    original_copy = shutil.copy2

    def fail_copy(
        src: str | Path,
        dst: str | Path,
        *args: object,
        follow_symlinks: bool = True,
        **kwargs: object,
    ) -> Path | str:
        if Path(dst).name == "guide.md":
            raise OSError("boom")
        return original_copy(src, dst, *args, follow_symlinks=follow_symlinks, **kwargs)

    monkeypatch.setattr(shutil, "copy2", fail_copy)
    list_tool = find_tool(section, "list_directory")
    session.reset()
    with pytest.raises(ToolValidationError, match="Failed to copy mounted file"):
        invoke_tool(bus, list_tool, ListDirectory(), session=session)


def test_delete_requires_existing_path(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    delete_tool = find_tool(section, "delete_file")
    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus, delete_tool, DeleteFile(path=VfsPath(("missing",))), session=session
        )


def test_delete_rejects_subpath_without_match(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = find_tool(section, "write_file")
    delete_tool = find_tool(section, "delete_file")

    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("logs", "events.log")), content=b"start"),
        session=session,
    )

    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            delete_tool,
            DeleteFile(path=VfsPath(("logs", "events.log", "old"))),
            session=session,
        )


def test_list_directory_rejects_file_path(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = find_tool(section, "write_file")
    list_tool = find_tool(section, "list_directory")

    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("src", "module.py")), content=b"print('hi')"),
        session=session,
    )

    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            list_tool,
            ListDirectory(path=VfsPath(("src", "module.py"))),
            session=session,
        )


def test_list_directory_defaults_to_root(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = find_tool(section, "write_file")
    list_tool = find_tool(section, "list_directory")

    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("notes.md",)), content=b"notes"),
        session=session,
    )

    result = cast(
        ToolResult[ListDirectoryResult],
        invoke_tool(bus, list_tool, ListDirectory(), session=session),
    )
    assert result.value is not None
    assert result.value.path == VfsPath(())
    assert result.value.files == ("notes.md",)


def test_list_directory_ignores_unrelated_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = find_tool(section, "write_file")
    list_tool = find_tool(section, "list_directory")

    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("src", "app.py")), content=b"print('x')"),
        session=session,
    )

    result = cast(
        ToolResult[ListDirectoryResult],
        invoke_tool(
            bus, list_tool, ListDirectory(path=VfsPath(("docs",))), session=session
        ),
    )
    assert result.value is not None
    assert result.value.directories == ()
    assert result.value.files == ()


def test_read_file_requires_existing_path(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    read_tool = find_tool(section, "read_file")
    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus, read_tool, ReadFile(path=VfsPath(("missing.txt",))), session=session
        )


def test_write_file_accepts_alt_encodings(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = find_tool(section, "write_file")
    read_tool = find_tool(section, "read_file")

    invoke_tool(
        bus,
        write_tool,
        WriteFile(
            path=VfsPath(("README.md",)),
            content=b"ol\xe1",
            encoding="latin-1",
        ),
        session=session,
    )

    result = cast(
        ToolResult[FileReadResult],
        invoke_tool(
            bus, read_tool, ReadFile(path=VfsPath(("README.md",))), session=session
        ),
    )
    assert result.value is not None
    assert result.value.content == b"ol\xe1"
    assert result.value.file.encoding == "latin-1"


def test_write_file_allows_utf8_content(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = find_tool(section, "write_file")
    read_tool = find_tool(section, "read_file")

    content = "café ☕"
    encoded = content.encode("utf-8")
    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("README.md",)), content=encoded),
        session=session,
    )

    result = cast(
        ToolResult[FileReadResult],
        invoke_tool(
            bus, read_tool, ReadFile(path=VfsPath(("README.md",))), session=session
        ),
    )
    assert result.value is not None
    assert result.value.content == encoded


def test_write_file_duplicate_create(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = find_tool(section, "write_file")

    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("config.yaml",)), content=b"first"),
        session=session,
    )

    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            write_tool,
            WriteFile(path=VfsPath(("config.yaml",)), content=b"second"),
            session=session,
        )


def test_write_file_requires_existing_for_overwrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = find_tool(section, "write_file")
    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            write_tool,
            WriteFile(
                path=VfsPath(("config.yaml",)),
                content=b"value",
                mode="overwrite",
            ),
            session=session,
        )


def test_write_file_limits_content_length(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = find_tool(section, "write_file")
    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            write_tool,
            WriteFile(path=VfsPath(("large.txt",)), content=b"x" * 48_001),
            session=session,
        )


def test_write_file_requires_nonempty_path(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = find_tool(section, "write_file")
    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            write_tool,
            WriteFile(path=VfsPath(()), content=b"body"),
            session=session,
        )


def test_path_depth_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = find_tool(section, "write_file")
    deep_path = tuple(f"segment{i}" for i in range(17))
    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            write_tool,
            WriteFile(path=VfsPath(deep_path), content=b"body"),
            session=session,
        )


def test_path_normalization_collapses_duplicate_slashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = find_tool(section, "write_file")

    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("src//module.py",)), content=b"pass"),
        session=session,
    )

    snapshot = select_latest(session, VirtualFileSystem)
    assert snapshot is not None
    assert snapshot.files[0].path == VfsPath(("src", "module.py"))


def test_path_rejects_dot_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = find_tool(section, "write_file")
    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            write_tool,
            WriteFile(path=VfsPath((".", "file.txt")), content=b"body"),
            session=session,
        )


def test_path_segment_length_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = find_tool(section, "write_file")
    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            write_tool,
            WriteFile(path=VfsPath(("a" * 81,)), content=b"body"),
            session=session,
        )


def test_path_normalization_ignores_blank_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = find_tool(section, "write_file")

    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("   ", "docs", "file.txt")), content=b"body"),
        session=session,
    )

    snapshot = select_latest(session, VirtualFileSystem)
    assert snapshot is not None
    assert snapshot.files[0].path == VfsPath(("docs", "file.txt"))


def test_host_mount_requires_existing_root(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)

    with pytest.raises(ToolValidationError):
        VfsToolsSection(
            session=session,
            mounts=(),
            allowed_host_roots=(Path("/tmp/nonexistent/root"),),
        )


def test_host_mount_requires_host_path(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    root = Path("/tmp").resolve()
    bus = InProcessEventBus()
    session = Session(bus=bus)
    with pytest.raises(ToolValidationError):
        VfsToolsSection(
            session=session,
            mounts=(HostMount(host_path="   "),),
            allowed_host_roots=(root,),
        )


def test_host_mount_respects_exclude_patterns(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    root = tmp_path_factory.mktemp("workspace")
    docs = root / "docs"
    docs.mkdir()
    (docs / "notes.md").write_text("keep", encoding="utf-8")

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(
        session=session,
        mounts=(
            HostMount(
                host_path="docs",
                mount_path=VfsPath(("docs",)),
                exclude_glob=("*.md",),
            ),
        ),
        allowed_host_roots=(root,),
    )
    read_tool = find_tool(section, "read_file")
    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            read_tool,
            ReadFile(path=VfsPath(("docs", "notes.md"))),
            session=session,
        )


def test_host_mount_enforces_byte_limit(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    root = tmp_path_factory.mktemp("workspace")
    data = root / "data"
    data.mkdir()
    (data / "big.txt").write_text("abc", encoding="utf-8")

    bus = InProcessEventBus()
    session = Session(bus=bus)
    with pytest.raises(ToolValidationError):
        VfsToolsSection(
            session=session,
            mounts=(
                HostMount(
                    host_path="data",
                    mount_path=VfsPath(("data",)),
                    include_glob=("*.txt",),
                    max_bytes=2,
                ),
            ),
            allowed_host_roots=(root,),
        )


def test_host_mount_requires_allowed_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    with pytest.raises(ToolValidationError):
        VfsToolsSection(
            session=session,
            mounts=(HostMount(host_path="docs"),),
            allowed_host_roots=(),
        )


def test_host_mount_prevents_directory_escape(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    root = tmp_path_factory.mktemp("workspace")
    bus = InProcessEventBus()
    session = Session(bus=bus)

    with pytest.raises(ToolValidationError):
        VfsToolsSection(
            session=session,
            mounts=(HostMount(host_path="../other"),),
            allowed_host_roots=(root,),
        )


def test_host_mount_trims_blank_globs(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    root = tmp_path_factory.mktemp("workspace")
    docs = root / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("guide", encoding="utf-8")

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(
        session=session,
        mounts=(
            HostMount(
                host_path="docs",
                include_glob=("*.md", ""),
                mount_path=VfsPath(("docs",)),
            ),
        ),
        allowed_host_roots=(root,),
    )
    read_tool = find_tool(section, "read_file")
    result = cast(
        ToolResult[FileReadResult],
        invoke_tool(
            bus,
            read_tool,
            ReadFile(path=VfsPath(("docs", "guide.md"))),
            session=session,
        ),
    )
    assert result.value is not None
    assert result.value.content == b"guide"


def test_host_mount_handles_file_targets(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    root = tmp_path_factory.mktemp("workspace")
    (root / "README.md").write_text("hello", encoding="utf-8")

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(
        session=session,
        mounts=(HostMount(host_path="README.md", mount_path=VfsPath(("docs",))),),
        allowed_host_roots=(root,),
    )
    read_tool = find_tool(section, "read_file")
    result = cast(
        ToolResult[FileReadResult],
        invoke_tool(
            bus,
            read_tool,
            ReadFile(path=VfsPath(("docs", "README.md"))),
            session=session,
        ),
    )
    assert result.value is not None
    assert result.value.content == b"hello"


def test_host_mount_preserves_utf8_content(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    root = tmp_path_factory.mktemp("workspace")
    content = "naïve résumé"
    (root / "notes.txt").write_text(content, encoding="utf-8")

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(
        session=session,
        mounts=(HostMount(host_path="notes.txt"),),
        allowed_host_roots=(root,),
    )
    read_tool = find_tool(section, "read_file")
    result = cast(
        ToolResult[FileReadResult],
        invoke_tool(
            bus,
            read_tool,
            ReadFile(path=VfsPath(("notes.txt",))),
            session=session,
        ),
    )
    assert result.value is not None
    assert result.value.content == content.encode("utf-8")


def test_host_mount_missing_path(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    root = tmp_path_factory.mktemp("workspace")
    bus = InProcessEventBus()
    session = Session(bus=bus)

    with pytest.raises(ToolValidationError):
        VfsToolsSection(
            session=session,
            mounts=(HostMount(host_path="missing"),),
            allowed_host_roots=(root,),
        )


def test_write_file_overwrite_updates_version(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamps = iter(
        (
            datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            datetime(2024, 1, 1, 12, 1, tzinfo=UTC),
        )
    )
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: next(timestamps))

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = find_tool(section, "write_file")

    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("report.txt",)), content=b"v1"),
        session=session,
    )
    invoke_tool(
        bus,
        write_tool,
        WriteFile(
            path=VfsPath(("report.txt",)),
            content=b"v2",
            mode="overwrite",
        ),
        session=session,
    )

    snapshot = select_latest(session, VirtualFileSystem)
    assert snapshot is not None
    file = snapshot.files[0]
    assert _read_disk_bytes(snapshot, file.path) == b"v2"
    assert file.version == 2


def test_write_to_mounted_file_uses_snapshot(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    timestamps = iter(
        (
            datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            datetime(2024, 1, 1, 12, 5, tzinfo=UTC),
        )
    )
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: next(timestamps))

    root = tmp_path_factory.mktemp("workspace")
    docs = root / "docs"
    docs.mkdir()
    (docs / "story.md").write_text("draft", encoding="utf-8")

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(
        session=session,
        mounts=(HostMount(host_path="docs", mount_path=VfsPath(("docs",))),),
        allowed_host_roots=(root,),
    )
    write_tool = find_tool(section, "write_file")
    invoke_tool(
        bus,
        write_tool,
        WriteFile(
            path=VfsPath(("docs", "story.md")),
            content=b"updated",
            mode="overwrite",
        ),
        session=session,
    )

    snapshot = select_latest(session, VirtualFileSystem)
    assert snapshot is not None
    file = snapshot.files[0]
    assert file.version == 2
    assert _read_disk_bytes(snapshot, file.path) == b"updated"


def test_now_truncates_to_milliseconds() -> None:
    timestamp = vfs_module._now()
    assert timestamp.tzinfo is UTC
    assert timestamp.microsecond % 1000 == 0


def test_prompt_section_exposes_all_tools() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)

    tool_names = {tool.name for tool in section.tools()}
    assert tool_names == {
        "list_directory",
        "read_file",
        "write_file",
        "delete_file",
    }
    assert all(tool.accepts_overrides is False for tool in section.tools())
    template = section.template
    assert "virtual filesystem starts empty" in template.lower()


def test_latest_snapshot_recreates_after_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    initial_snapshot = section.latest_snapshot()
    initial_root = initial_snapshot.root_path
    session.reset()

    snapshot = section.latest_snapshot()
    assert snapshot.files == ()
    assert Path(snapshot.root_path).exists()
    assert snapshot.root_path != initial_root


def test_write_file_invalid_utf8_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = find_tool(section, "write_file")

    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            write_tool,
            WriteFile(
                path=VfsPath(("notes.txt",)),
                content=b"\xff",
                encoding="utf-8",
            ),
            session=session,
        )


def test_write_file_binary_length_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = find_tool(section, "write_file")

    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            write_tool,
            WriteFile(
                path=VfsPath(("data.bin",)),
                content=b"0" * (vfs_module._MAX_WRITE_LENGTH + 1),
                encoding="binary",
            ),
            session=session,
        )


def test_append_blank_encoding_inherits_previous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = find_tool(section, "write_file")
    read_tool = find_tool(section, "read_file")

    invoke_tool(
        bus,
        write_tool,
        WriteFile(
            path=VfsPath(("data.bin",)),
            content=b"seed",
            encoding="binary",
        ),
        session=session,
    )

    invoke_tool(
        bus,
        write_tool,
        WriteFile(
            path=VfsPath(("data.bin",)),
            content=b"more",
            mode="append",
            encoding="  ",
        ),
        session=session,
    )

    result = cast(
        ToolResult[FileReadResult],
        invoke_tool(
            bus, read_tool, ReadFile(path=VfsPath(("data.bin",))), session=session
        ),
    )
    assert result.value is not None
    assert result.value.file.encoding == "binary"
    assert result.value.content == b"seedmore"


def test_mount_marks_binary_encoding(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    binary = root / "blob.bin"
    binary.write_bytes(b"\xff\x00\xff")

    bus = InProcessEventBus()
    session = Session(bus=bus)
    mount = HostMount(host_path="blob.bin")
    section = VfsToolsSection(
        session=session, mounts=(mount,), allowed_host_roots=(root,)
    )
    list_tool = find_tool(section, "list_directory")

    invoke_tool(bus, list_tool, ListDirectory(), session=session)
    snapshot = select_latest(session, VirtualFileSystem)
    assert snapshot is not None
    file = snapshot.files[0]
    assert file.encoding == "binary"


def test_write_file_disk_failure_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = find_tool(section, "write_file")

    original_open = Path.open

    def fail_open(
        self: Path,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> IO[Any]:
        if self.name == "fail.txt" and any(flag in mode for flag in {"w", "a", "x"}):
            raise OSError("boom")
        return original_open(self, mode, buffering, encoding, errors, newline)

    monkeypatch.setattr(Path, "open", fail_open)

    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            write_tool,
            WriteFile(path=VfsPath(("fail.txt",)), content=b"oops"),
            session=session,
        )


def test_delete_file_disk_failure_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = find_tool(section, "write_file")
    delete_tool = find_tool(section, "delete_file")

    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("scratch.txt",)), content=b"temp"),
        session=session,
    )

    snapshot = section.latest_snapshot()

    original_unlink = Path.unlink

    def fail_unlink(self: Path, missing_ok: bool = False) -> None:
        if self == Path(snapshot.root_path) / "scratch.txt":
            raise OSError("boom")
        return original_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_unlink)

    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            delete_tool,
            DeleteFile(path=VfsPath(("scratch.txt",))),
            session=session,
        )


def test_prune_empty_parents_handles_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "other"
    outside.mkdir()

    vfs_module._prune_empty_parents(root, outside)


def test_prune_empty_parents_removes_directories(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    target = root / "nested" / "leaf"
    target.mkdir(parents=True)
    file = target / "data.txt"
    file.write_text("hello", encoding="utf-8")
    file.unlink()

    vfs_module._prune_empty_parents(root, target)
    vfs_module._prune_empty_parents(root, target)

    assert not target.exists()
    assert not (root / "nested").exists()


def test_normalize_encoding_none_returns_none() -> None:
    assert vfs_module._normalize_encoding(None) is None


def test_delete_files_from_disk_ignores_missing(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    ghost = VfsFile(
        path=VfsPath(("missing.txt",)),
        encoding="utf-8",
        size_bytes=0,
        version=1,
        created_at=timestamp,
        updated_at=timestamp,
    )
    vfs_module._delete_files_from_disk(root, (ghost,))


def test_write_reducer_requires_state() -> None:
    reducer = vfs_module._make_write_reducer()
    event = cast(
        ReducerEvent,
        SimpleNamespace(value=WriteFile(path=VfsPath(("a",)), content=b"")),
    )
    context = cast(
        ToolContext,
        SimpleNamespace(session=Session(bus=InProcessEventBus())),
    )
    with pytest.raises(RuntimeError):
        reducer((), event, context=context)


def test_write_reducer_inherits_existing_encoding(tmp_path: Path) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    existing = VfsFile(
        path=VfsPath(("data.bin",)),
        encoding="binary",
        size_bytes=4,
        version=1,
        created_at=timestamp,
        updated_at=timestamp,
    )
    reducer = vfs_module._make_write_reducer()
    snapshot = VirtualFileSystem(root_path=str(tmp_path), files=(existing,))
    event = cast(
        ReducerEvent,
        SimpleNamespace(
            value=WriteFile(
                path=existing.path,
                content=b"seed",  # content not used in branch
                mode="overwrite",
                encoding=None,
            )
        ),
    )
    context = cast(
        ToolContext,
        SimpleNamespace(session=Session(bus=InProcessEventBus())),
    )
    updated = reducer((snapshot,), event, context=context)[0]
    updated_file = updated.files[0]
    assert updated_file.encoding == "binary"


def test_delete_reducer_requires_state() -> None:
    reducer = vfs_module._make_delete_reducer()
    event = cast(
        ReducerEvent,
        SimpleNamespace(value=DeleteFile(path=VfsPath(("a",)))),
    )
    context = cast(
        ToolContext,
        SimpleNamespace(session=Session(bus=InProcessEventBus())),
    )
    with pytest.raises(RuntimeError):
        reducer((), event, context=context)


def test_vfs_tools_section_allows_selective_override_opt_in() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(
        session=session,
        accepts_overrides=True,
    )

    read_tool = find_tool(section, "read_file")
    list_tool = find_tool(section, "list_directory")

    assert section.accepts_overrides is True
    assert read_tool.accepts_overrides is True
    assert list_tool.accepts_overrides is True


def test_constructor_reuses_existing_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    first = VfsToolsSection(session=session)
    first_snapshot = first.latest_snapshot()

    second = VfsToolsSection(session=session)
    second_snapshot = second.latest_snapshot()
    assert second_snapshot.root_path == first_snapshot.root_path


def test_latest_snapshot_raises_when_refresh_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    failing_refresh = MethodType(lambda self: None, section)
    monkeypatch.setattr(section, "_refresh_snapshot", failing_refresh)
    session.reset()

    with pytest.raises(ToolValidationError, match="Virtual filesystem not initialized"):
        section.latest_snapshot()
