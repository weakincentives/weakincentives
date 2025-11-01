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
from typing import TypeVar, cast

import pytest

import weakincentives.tools.vfs as vfs_module
from weakincentives.events import InProcessEventBus, ToolInvoked
from weakincentives.prompt import SupportsDataclass
from weakincentives.prompt.tool import Tool, ToolResult
from weakincentives.session import Session, select_latest
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

ParamsT = TypeVar("ParamsT", bound=SupportsDataclass)
ResultT = TypeVar("ResultT", bound=SupportsDataclass)


def _find_tool(
    section: VfsToolsSection, name: str
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


def test_write_file_creates_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = _find_tool(section, "vfs_write_file")

    params = WriteFile(path=VfsPath(("docs", "intro.md")), content="hello world")
    _invoke_tool(bus, write_tool, params)

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
    write_tool = _find_tool(section, "vfs_write_file")

    _invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("log.txt",)), content="start"),
    )
    _invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("log.txt",)), content=" -> next", mode="append"),
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
    section = VfsToolsSection(session=session)
    write_tool = _find_tool(section, "vfs_write_file")
    delete_tool = _find_tool(section, "vfs_delete_entry")

    _invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("src", "main.py")), content="print('hello')"),
    )
    _invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("src", "utils", "helpers.py")), content="# helpers"),
    )

    _invoke_tool(bus, delete_tool, DeleteEntry(path=VfsPath(("src",))))

    snapshot = select_latest(session, VirtualFileSystem)
    assert snapshot is not None
    assert snapshot.files == ()


def test_list_directory_shows_children(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = _find_tool(section, "vfs_write_file")
    list_tool = _find_tool(section, "vfs_list_directory")

    _invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("workspace", "notes.md")), content="notes"),
    )
    _invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("workspace", "drafts", "todo.md")), content="todo"),
    )

    result = _invoke_tool(bus, list_tool, ListDirectory(path=VfsPath(("workspace",))))
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
    write_tool = _find_tool(section, "vfs_write_file")
    read_tool = _find_tool(section, "vfs_read_file")

    _invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("README.md",)), content="intro"),
    )

    result = _invoke_tool(bus, read_tool, ReadFile(path=VfsPath(("README.md",))))
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
    section = VfsToolsSection(session=session)
    write_tool = _find_tool(section, "vfs_write_file")
    handler = write_tool.handler
    assert handler is not None

    with pytest.raises(ToolValidationError):
        handler(WriteFile(path=VfsPath(("/etc", "passwd")), content="x"))


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
    read_tool = _find_tool(section, "vfs_read_file")

    result = _invoke_tool(
        bus,
        read_tool,
        ReadFile(path=VfsPath(("reference", "guide.md"))),
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
    section = VfsToolsSection(session=session)
    delete_tool = _find_tool(section, "vfs_delete_entry")
    handler = delete_tool.handler
    assert handler is not None

    with pytest.raises(ToolValidationError):
        handler(DeleteEntry(path=VfsPath(("missing",))))


def test_delete_rejects_subpath_without_match(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = _find_tool(section, "vfs_write_file")
    delete_tool = _find_tool(section, "vfs_delete_entry")

    _invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("logs", "events.log")), content="start"),
    )

    handler = delete_tool.handler
    assert handler is not None
    with pytest.raises(ToolValidationError):
        handler(DeleteEntry(path=VfsPath(("logs", "events.log", "old"))))


def test_list_directory_rejects_file_path(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = _find_tool(section, "vfs_write_file")
    list_tool = _find_tool(section, "vfs_list_directory")

    _invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("src", "module.py")), content="print('hi')"),
    )

    handler = list_tool.handler
    assert handler is not None
    with pytest.raises(ToolValidationError):
        handler(ListDirectory(path=VfsPath(("src", "module.py"))))


def test_list_directory_defaults_to_root(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = _find_tool(section, "vfs_write_file")
    list_tool = _find_tool(section, "vfs_list_directory")

    _invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("notes.md",)), content="notes"),
    )

    result = cast(
        ToolResult[ListDirectoryResult],
        _invoke_tool(bus, list_tool, ListDirectory()),
    )
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
    write_tool = _find_tool(section, "vfs_write_file")
    list_tool = _find_tool(section, "vfs_list_directory")

    _invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("src", "app.py")), content="print('x')"),
    )

    result = cast(
        ToolResult[ListDirectoryResult],
        _invoke_tool(bus, list_tool, ListDirectory(path=VfsPath(("docs",)))),
    )
    assert result.value.directories == ()
    assert result.value.files == ()


def test_read_file_requires_existing_path(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    read_tool = _find_tool(section, "vfs_read_file")
    handler = read_tool.handler
    assert handler is not None

    with pytest.raises(ToolValidationError):
        handler(ReadFile(path=VfsPath(("missing.txt",))))


def test_write_file_rejects_non_utf8_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = _find_tool(section, "vfs_write_file")
    handler = write_tool.handler
    assert handler is not None

    with pytest.raises(ToolValidationError):
        handler(
            WriteFile(
                path=VfsPath(("README.md",)),
                content="hello",
                encoding="latin-1",  # type: ignore[arg-type]
            )
        )


def test_write_file_duplicate_create(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = _find_tool(section, "vfs_write_file")

    _invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("config.yaml",)), content="first"),
    )

    handler = write_tool.handler
    assert handler is not None
    with pytest.raises(ToolValidationError):
        handler(WriteFile(path=VfsPath(("config.yaml",)), content="second"))


def test_write_file_requires_existing_for_overwrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = _find_tool(section, "vfs_write_file")
    handler = write_tool.handler
    assert handler is not None

    with pytest.raises(ToolValidationError):
        handler(
            WriteFile(
                path=VfsPath(("config.yaml",)),
                content="value",
                mode="overwrite",
            )
        )


def test_write_file_limits_content_length(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = _find_tool(section, "vfs_write_file")
    handler = write_tool.handler
    assert handler is not None

    with pytest.raises(ToolValidationError):
        handler(WriteFile(path=VfsPath(("large.txt",)), content="x" * 48_001))


def test_write_file_requires_nonempty_path(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = _find_tool(section, "vfs_write_file")
    handler = write_tool.handler
    assert handler is not None

    with pytest.raises(ToolValidationError):
        handler(WriteFile(path=VfsPath(()), content="body"))


def test_path_depth_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = _find_tool(section, "vfs_write_file")
    handler = write_tool.handler
    assert handler is not None

    deep_path = tuple(f"segment{i}" for i in range(17))
    with pytest.raises(ToolValidationError):
        handler(WriteFile(path=VfsPath(deep_path), content="body"))


def test_path_normalization_collapses_duplicate_slashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = _find_tool(section, "vfs_write_file")

    _invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("src//module.py",)), content="pass"),
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
    write_tool = _find_tool(section, "vfs_write_file")
    handler = write_tool.handler
    assert handler is not None

    with pytest.raises(ToolValidationError):
        handler(WriteFile(path=VfsPath((".", "file.txt")), content="body"))


def test_path_segment_length_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = _find_tool(section, "vfs_write_file")
    handler = write_tool.handler
    assert handler is not None

    with pytest.raises(ToolValidationError):
        handler(WriteFile(path=VfsPath(("a" * 81,)), content="body"))


def test_path_normalization_ignores_blank_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    write_tool = _find_tool(section, "vfs_write_file")

    _invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("   ", "docs", "file.txt")), content="body"),
    )

    snapshot = select_latest(session, VirtualFileSystem)
    assert snapshot is not None
    assert snapshot.files[0].path == VfsPath(("docs", "file.txt"))


def test_host_mount_requires_existing_root(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    with pytest.raises(ToolValidationError):
        VfsToolsSection(
            session=Session(bus=InProcessEventBus()),
            mounts=(),
            allowed_host_roots=(Path("/tmp/nonexistent/root"),),
        )


def test_host_mount_requires_host_path(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    root = Path("/tmp").resolve()
    with pytest.raises(ToolValidationError):
        VfsToolsSection(
            session=Session(bus=InProcessEventBus()),
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
    read_tool = _find_tool(section, "vfs_read_file")
    with pytest.raises(ToolValidationError):
        _invoke_tool(bus, read_tool, ReadFile(path=VfsPath(("docs", "notes.md"))))


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
            session=Session(bus=InProcessEventBus()),
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
            session=Session(bus=InProcessEventBus()),
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
            session=Session(bus=InProcessEventBus()),
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
    read_tool = _find_tool(section, "vfs_read_file")
    result = cast(
        ToolResult[VfsFile],
        _invoke_tool(
            bus,
            read_tool,
            ReadFile(path=VfsPath(("docs", "guide.md"))),
        ),
    )
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
        session=session,
        mounts=(HostMount(host_path="README.md", mount_path=VfsPath(("docs",))),),
        allowed_host_roots=(root,),
    )
    read_tool = _find_tool(section, "vfs_read_file")
    result = cast(
        ToolResult[VfsFile],
        _invoke_tool(
            bus,
            read_tool,
            ReadFile(path=VfsPath(("docs", "README.md"))),
        ),
    )
    assert result.value.content == "hello"


def test_host_mount_missing_path(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.tools.vfs._now", lambda: timestamp)

    root = tmp_path_factory.mktemp("workspace")

    with pytest.raises(ToolValidationError):
        VfsToolsSection(
            session=Session(bus=InProcessEventBus()),
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
    write_tool = _find_tool(section, "vfs_write_file")

    _invoke_tool(
        bus,
        write_tool,
        WriteFile(path=VfsPath(("report.txt",)), content="v1"),
    )
    _invoke_tool(
        bus,
        write_tool,
        WriteFile(
            path=VfsPath(("report.txt",)),
            content="v2",
            mode="overwrite",
        ),
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
        session=session,
        mounts=(HostMount(host_path="docs", mount_path=VfsPath(("docs",))),),
        allowed_host_roots=(root,),
    )
    write_tool = _find_tool(section, "vfs_write_file")
    _invoke_tool(
        bus,
        write_tool,
        WriteFile(
            path=VfsPath(("docs", "story.md")),
            content="updated",
            mode="overwrite",
        ),
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
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)

    tool_names = {tool.name for tool in section.tools()}
    assert tool_names == {
        "vfs_list_directory",
        "vfs_read_file",
        "vfs_write_file",
        "vfs_delete_entry",
    }
    template = section.template
    assert "virtual filesystem starts empty" in template.lower()
