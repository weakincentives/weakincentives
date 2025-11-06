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

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

import weakincentives.tools.vfs as vfs_module
from tests.tools.helpers import find_tool, invoke_tool
from weakincentives.adapters.core import SessionProtocol
from weakincentives.prompt.tool import ToolContext, ToolResult
from weakincentives.runtime.events import InProcessEventBus
from weakincentives.runtime.session import Session, select_latest
from weakincentives.tools import (
    DeleteEntry,
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


def test_write_file_creates_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection()
    write_tool = find_tool(section, "vfs_write_file")

    params = WriteFile(path=VfsPath(("docs", "intro.md")), content="hello world")
    invoke_tool(bus, write_tool, params, session=session)

    snapshot = select_latest(session, VirtualFileSystem)
    assert snapshot is not None
    assert snapshot.files == (
        VfsFile(
            path=VfsPath(("docs", "intro.md")),
            content="hello world",
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
    Session(bus=bus)
    section = VfsToolsSection()
    write_tool = find_tool(section, "vfs_write_file")

    params = WriteFile(path=VfsPath(("docs", "intro.md")), content="hello world")
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
    section = VfsToolsSection()
    write_tool = find_tool(section, "vfs_write_file")

    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("log.txt",)), content="start"),
        session=session,
    )
    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("log.txt",)), content=" -> next", mode="append"),
        session=session,
    )

    snapshot = select_latest(session, VirtualFileSystem)
    assert snapshot is not None
    file = snapshot.files[0]
    assert file.content == "start -> next"
    assert file.version == 2
    assert file.size_bytes == len(b"start -> next")


def test_delete_directory_removes_nested(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection()
    write_tool = find_tool(section, "vfs_write_file")
    delete_tool = find_tool(section, "vfs_delete_entry")

    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("src", "main.py")), content="print('hello')"),
        session=session,
    )
    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("src", "utils", "helpers.py")), content="# helpers"),
        session=session,
    )

    invoke_tool(bus, delete_tool, DeleteEntry(path=VfsPath(("src",))), session=session)

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
        mounts=(mount,),
        allowed_host_roots=(root,),
    )
    list_tool = find_tool(section, "vfs_list_directory")
    invoke_tool(bus, list_tool, ListDirectory(), session=session)

    snapshot = select_latest(session, VirtualFileSystem)
    assert snapshot is not None
    files_by_path = {file.path.segments: file for file in snapshot.files}
    assert ("sunfish", "README.md") in files_by_path
    assert files_by_path["sunfish", "README.md"].content == "hello mount"


def test_mount_permission_error_translated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    restricted = root / "restricted.txt"
    restricted.write_text("secret", encoding="utf-8")

    original_read_text = Path.read_text

    def fake_read_text(
        self: Path, encoding: str | None = None, errors: str | None = None
    ) -> str:
        if self == restricted:
            raise PermissionError("access denied")
        return original_read_text(self, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", fake_read_text)

    mount = HostMount(host_path="restricted.txt", mount_path=VfsPath(("restricted",)))

    with pytest.raises(ToolValidationError, match=str(restricted)):
        VfsToolsSection(
            mounts=(mount,),
            allowed_host_roots=(root,),
        )


def test_list_directory_shows_children(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection()
    write_tool = find_tool(section, "vfs_write_file")
    list_tool = find_tool(section, "vfs_list_directory")

    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("workspace", "notes.md")), content="notes"),
        session=session,
    )
    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("workspace", "drafts", "todo.md")), content="todo"),
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
    section = VfsToolsSection()
    write_tool = find_tool(section, "vfs_write_file")
    read_tool = find_tool(section, "vfs_read_file")

    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("README.md",)), content="intro"),
        session=session,
    )

    result = invoke_tool(
        bus, read_tool, ReadFile(path=VfsPath(("README.md",))), session=session
    )
    file = result.value
    assert isinstance(file, VfsFile)
    assert file.version == 1
    assert file.created_at == timestamp
    assert file.updated_at == timestamp


def test_write_file_rejects_invalid_path(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection()
    write_tool = find_tool(section, "vfs_write_file")
    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            write_tool,
            WriteFile(path=VfsPath(("/etc", "passwd")), content="x"),
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
        mounts=(
            HostMount(
                host_path="docs",
                mount_path=VfsPath(("reference",)),
                include_glob=("*.md",),
            ),
        ),
        allowed_host_roots=(root,),
    )
    read_tool = find_tool(section, "vfs_read_file")

    result = invoke_tool(
        bus,
        read_tool,
        ReadFile(path=VfsPath(("reference", "guide.md"))),
        session=session,
    )
    file = result.value
    assert file == VfsFile(
        path=VfsPath(("reference", "guide.md")),
        content="guide",
        encoding="utf-8",
        size_bytes=len(b"guide"),
        version=1,
        created_at=timestamp,
        updated_at=timestamp,
    )


def test_delete_requires_existing_path(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection()
    delete_tool = find_tool(section, "vfs_delete_entry")
    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus, delete_tool, DeleteEntry(path=VfsPath(("missing",))), session=session
        )


def test_delete_rejects_subpath_without_match(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection()
    write_tool = find_tool(section, "vfs_write_file")
    delete_tool = find_tool(section, "vfs_delete_entry")

    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("logs", "events.log")), content="start"),
        session=session,
    )

    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            delete_tool,
            DeleteEntry(path=VfsPath(("logs", "events.log", "old"))),
            session=session,
        )


def test_list_directory_rejects_file_path(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection()
    write_tool = find_tool(section, "vfs_write_file")
    list_tool = find_tool(section, "vfs_list_directory")

    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("src", "module.py")), content="print('hi')"),
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
    section = VfsToolsSection()
    write_tool = find_tool(section, "vfs_write_file")
    list_tool = find_tool(section, "vfs_list_directory")

    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("notes.md",)), content="notes"),
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
    section = VfsToolsSection()
    write_tool = find_tool(section, "vfs_write_file")
    list_tool = find_tool(section, "vfs_list_directory")

    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("src", "app.py")), content="print('x')"),
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
    section = VfsToolsSection()
    read_tool = find_tool(section, "vfs_read_file")
    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus, read_tool, ReadFile(path=VfsPath(("missing.txt",))), session=session
        )


def test_write_file_rejects_non_utf8_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection()
    write_tool = find_tool(section, "vfs_write_file")
    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            write_tool,
            WriteFile(
                path=VfsPath(("README.md",)),
                content="hello",
                encoding="latin-1",  # type: ignore[arg-type]
            ),
            session=session,
        )


def test_write_file_allows_utf8_content(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection()
    write_tool = find_tool(section, "vfs_write_file")
    read_tool = find_tool(section, "vfs_read_file")

    content = "café ☕"
    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("README.md",)), content=content),
        session=session,
    )

    result = cast(
        ToolResult[VfsFile],
        invoke_tool(
            bus, read_tool, ReadFile(path=VfsPath(("README.md",))), session=session
        ),
    )
    assert result.value is not None
    assert result.value.content == content


def test_write_file_duplicate_create(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection()
    write_tool = find_tool(section, "vfs_write_file")

    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("config.yaml",)), content="first"),
        session=session,
    )

    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            write_tool,
            WriteFile(path=VfsPath(("config.yaml",)), content="second"),
            session=session,
        )


def test_write_file_requires_existing_for_overwrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection()
    write_tool = find_tool(section, "vfs_write_file")
    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            write_tool,
            WriteFile(
                path=VfsPath(("config.yaml",)),
                content="value",
                mode="overwrite",
            ),
            session=session,
        )


def test_write_file_limits_content_length(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection()
    write_tool = find_tool(section, "vfs_write_file")
    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            write_tool,
            WriteFile(path=VfsPath(("large.txt",)), content="x" * 48_001),
            session=session,
        )


def test_write_file_requires_nonempty_path(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection()
    write_tool = find_tool(section, "vfs_write_file")
    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            write_tool,
            WriteFile(path=VfsPath(()), content="body"),
            session=session,
        )


def test_path_depth_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection()
    write_tool = find_tool(section, "vfs_write_file")
    deep_path = tuple(f"segment{i}" for i in range(17))
    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            write_tool,
            WriteFile(path=VfsPath(deep_path), content="body"),
            session=session,
        )


def test_path_normalization_collapses_duplicate_slashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection()
    write_tool = find_tool(section, "vfs_write_file")

    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("src//module.py",)), content="pass"),
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
    section = VfsToolsSection()
    write_tool = find_tool(section, "vfs_write_file")
    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            write_tool,
            WriteFile(path=VfsPath((".", "file.txt")), content="body"),
            session=session,
        )


def test_path_segment_length_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection()
    write_tool = find_tool(section, "vfs_write_file")
    with pytest.raises(ToolValidationError):
        invoke_tool(
            bus,
            write_tool,
            WriteFile(path=VfsPath(("a" * 81,)), content="body"),
            session=session,
        )


def test_path_normalization_ignores_blank_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection()
    write_tool = find_tool(section, "vfs_write_file")

    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("   ", "docs", "file.txt")), content="body"),
        session=session,
    )

    snapshot = select_latest(session, VirtualFileSystem)
    assert snapshot is not None
    assert snapshot.files[0].path == VfsPath(("docs", "file.txt"))


def test_host_mount_requires_existing_root(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    with pytest.raises(ToolValidationError):
        VfsToolsSection(
            mounts=(),
            allowed_host_roots=(Path("/tmp/nonexistent/root"),),
        )


def test_host_mount_requires_host_path(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    root = Path("/tmp").resolve()
    with pytest.raises(ToolValidationError):
        VfsToolsSection(
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
        mounts=(
            HostMount(
                host_path="docs",
                mount_path=VfsPath(("docs",)),
                exclude_glob=("*.md",),
            ),
        ),
        allowed_host_roots=(root,),
    )
    read_tool = find_tool(section, "vfs_read_file")
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

    with pytest.raises(ToolValidationError):
        VfsToolsSection(
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

    with pytest.raises(ToolValidationError):
        VfsToolsSection(
            mounts=(HostMount(host_path="docs"),),
            allowed_host_roots=(),
        )


def test_host_mount_prevents_directory_escape(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    root = tmp_path_factory.mktemp("workspace")

    with pytest.raises(ToolValidationError):
        VfsToolsSection(
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
        mounts=(
            HostMount(
                host_path="docs",
                include_glob=("*.md", ""),
                mount_path=VfsPath(("docs",)),
            ),
        ),
        allowed_host_roots=(root,),
    )
    read_tool = find_tool(section, "vfs_read_file")
    result = cast(
        ToolResult[VfsFile],
        invoke_tool(
            bus,
            read_tool,
            ReadFile(path=VfsPath(("docs", "guide.md"))),
            session=session,
        ),
    )
    assert result.value is not None
    assert result.value.content == "guide"


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
        mounts=(HostMount(host_path="README.md", mount_path=VfsPath(("docs",))),),
        allowed_host_roots=(root,),
    )
    read_tool = find_tool(section, "vfs_read_file")
    result = cast(
        ToolResult[VfsFile],
        invoke_tool(
            bus,
            read_tool,
            ReadFile(path=VfsPath(("docs", "README.md"))),
            session=session,
        ),
    )
    assert result.value is not None
    assert result.value.content == "hello"


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
        mounts=(HostMount(host_path="notes.txt"),),
        allowed_host_roots=(root,),
    )
    read_tool = find_tool(section, "vfs_read_file")
    result = cast(
        ToolResult[VfsFile],
        invoke_tool(
            bus,
            read_tool,
            ReadFile(path=VfsPath(("notes.txt",))),
            session=session,
        ),
    )
    assert result.value is not None
    assert result.value.content == content


def test_host_mount_missing_path(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    root = tmp_path_factory.mktemp("workspace")

    with pytest.raises(ToolValidationError):
        VfsToolsSection(
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
    section = VfsToolsSection()
    write_tool = find_tool(section, "vfs_write_file")

    invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("report.txt",)), content="v1"),
        session=session,
    )
    invoke_tool(
        bus,
        write_tool,
        WriteFile(
            path=VfsPath(("report.txt",)),
            content="v2",
            mode="overwrite",
        ),
        session=session,
    )

    snapshot = select_latest(session, VirtualFileSystem)
    assert snapshot is not None
    file = snapshot.files[0]
    assert file.content == "v2"
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
        mounts=(HostMount(host_path="docs", mount_path=VfsPath(("docs",))),),
        allowed_host_roots=(root,),
    )
    write_tool = find_tool(section, "vfs_write_file")
    invoke_tool(
        bus,
        write_tool,
        WriteFile(
            path=VfsPath(("docs", "story.md")),
            content="updated",
            mode="overwrite",
        ),
        session=session,
    )

    snapshot = select_latest(session, VirtualFileSystem)
    assert snapshot is not None
    file = snapshot.files[0]
    assert file.version == 2
    assert file.content == "updated"


def test_now_truncates_to_milliseconds() -> None:
    timestamp = vfs_module._now()
    assert timestamp.tzinfo is UTC
    assert timestamp.microsecond % 1000 == 0


def test_prompt_section_exposes_all_tools() -> None:
    section = VfsToolsSection()

    tool_names = {tool.name for tool in section.tools()}
    assert tool_names == {
        "vfs_list_directory",
        "vfs_read_file",
        "vfs_write_file",
        "vfs_delete_entry",
    }
    assert all(tool.accepts_overrides is False for tool in section.tools())
    template = section.template
    assert "virtual filesystem starts empty" in template.lower()


def test_vfs_tools_section_allows_selective_override_opt_in() -> None:
    section = VfsToolsSection(
        accepts_overrides=True,
    )

    read_tool = find_tool(section, "vfs_read_file")
    list_tool = find_tool(section, "vfs_list_directory")

    assert section.accepts_overrides is True
    assert read_tool.accepts_overrides is True
    assert list_tool.accepts_overrides is True
