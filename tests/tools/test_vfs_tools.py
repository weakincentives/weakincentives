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

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

import weakincentives.contrib.tools.vfs as vfs_module
from tests.tools.helpers import find_tool, invoke_tool
from weakincentives import ToolValidationError
from weakincentives.contrib.tools import (
    DeleteEntry,
    EditFileParams,
    FileInfo,
    GlobMatch,
    GlobParams,
    GrepMatch,
    GrepParams,
    HostMount,
    InMemoryFilesystem,
    ListDirectoryParams,
    ListDirectoryResult,
    ReadFileParams,
    ReadFileResult,
    RemoveParams,
    VfsPath,
    VfsToolsSection,
    WriteFile,
    WriteFileParams,
)
from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate
from weakincentives.runtime.events import InProcessEventBus
from weakincentives.runtime.session import Session


def _make_section(
    *,
    session: Session | None = None,
    mounts: Sequence[HostMount] = (),
    allowed_host_roots: Sequence[Path | str] = (),
    accepts_overrides: bool = False,
) -> VfsToolsSection:
    if session is None:
        session = Session(bus=InProcessEventBus())
    return VfsToolsSection(
        session=session,
        mounts=mounts,
        allowed_host_roots=allowed_host_roots,
        accepts_overrides=accepts_overrides,
    )


@pytest.fixture()
def session_and_bus() -> tuple[Session, InProcessEventBus]:
    bus = InProcessEventBus()
    return Session(bus=bus), bus


def test_file_info_render_formats_metadata() -> None:
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    file_info = FileInfo(
        path=VfsPath(("src", "app.py")),
        kind="file",
        size_bytes=24,
        version=3,
        updated_at=timestamp,
    )

    rendered_file = file_info.render()
    assert "src/app.py" in rendered_file
    assert "24 B" in rendered_file
    assert "v3" in rendered_file

    directory_info = FileInfo(path=VfsPath(("src",)), kind="directory")
    rendered_directory = directory_info.render()
    assert rendered_directory.startswith("DIR")
    assert rendered_directory.endswith("/")


def test_read_file_result_render_includes_window() -> None:
    payload = ReadFileResult(
        path=VfsPath(("docs", "note.txt")),
        content="3 | value",
        offset=2,
        limit=1,
        total_lines=5,
    )

    rendered = payload.render()
    assert "lines 3-3" in rendered
    assert "value" in rendered


def test_write_file_render_includes_preview() -> None:
    payload = WriteFile(
        path=VfsPath(("docs", "draft.md")),
        content="first line\nsecond line",
        mode="create",
    )

    rendered = payload.render()
    assert "mode create" in rendered
    assert "first line" in rendered


def test_glob_and_grep_render_strings() -> None:
    timestamp = datetime(2024, 2, 1, tzinfo=UTC)
    glob_match = GlobMatch(
        path=VfsPath(("docs", "readme.md")),
        size_bytes=120,
        version=5,
        updated_at=timestamp,
    )
    grep_match = GrepMatch(
        path=VfsPath(("docs", "readme.md")),
        line_number=12,
        line="match content",
    )

    assert "docs/readme.md" in glob_match.render()
    assert "120 B" in glob_match.render()
    assert "docs/readme.md:12" in grep_match.render()


def test_delete_entry_render_mentions_path() -> None:
    payload = DeleteEntry(path=VfsPath(("tmp", "cache")))

    assert "tmp/cache" in payload.render()


def test_format_timestamp_helper_and_write_render() -> None:
    naive = datetime(2024, 1, 1, 12, 0, 0)
    aware = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    assert vfs_module._format_timestamp(naive).endswith("+00:00")
    assert vfs_module._format_timestamp(aware).endswith("+00:00")
    assert vfs_module._format_timestamp(None) == "-"

    write = WriteFile(
        path=VfsPath(("docs", "notes.txt")), content="full content", mode="create"
    )
    rendered = write.render()
    assert "full content" in rendered
    assert "create" in rendered


def _write(
    session: Session,
    section: VfsToolsSection,
    *,
    path: tuple[str, ...],
    content: str,
) -> None:
    tool = find_tool(section, "write_file")
    params = WriteFileParams(file_path="/".join(path), content=content)
    invoke_tool(tool, params, session=session)


def _get_filesystem(section: VfsToolsSection) -> InMemoryFilesystem:
    """Get the InMemoryFilesystem from a section."""
    fs = section.filesystem
    assert isinstance(fs, InMemoryFilesystem)
    return fs


def test_section_template_mentions_new_surface() -> None:
    section = _make_section()
    template = section.template
    assert "ls" in template
    assert "write_file" in template
    assert "edit_file" in template
    assert "rm" in template


def test_section_exposes_session_handle(
    session_and_bus: tuple[Session, InProcessEventBus],
) -> None:
    session, _bus = session_and_bus
    section = _make_section(session=session)

    assert section.session is session


def test_section_template_includes_host_mount_preview(tmp_path: Path) -> None:
    allowed_root = tmp_path / "workspace"
    mount_root = allowed_root / "project"
    data_dir = mount_root / "docs"
    data_dir.mkdir(parents=True)
    (mount_root / "README.md").write_text("hello", encoding="utf-8")
    (data_dir / "guide.md").write_text("guide", encoding="utf-8")

    section = _make_section(
        mounts=(HostMount(host_path="project", mount_path=VfsPath(("mnt",))),),
        allowed_host_roots=(allowed_root,),
    )

    template = section.template
    resolved_path = mount_root.resolve()
    assert "Configured host mounts:" in template
    assert f"`{resolved_path}`" in template
    assert "`README.md`" in template
    assert "`docs/`" in template
    assert "`mnt`" in template


def test_section_template_handles_file_mount(tmp_path: Path) -> None:
    allowed_root = tmp_path / "workspace"
    allowed_root.mkdir(parents=True)
    (allowed_root / "notes.txt").write_text("hello", encoding="utf-8")

    section = _make_section(
        mounts=(HostMount(host_path="notes.txt"),),
        allowed_host_roots=(allowed_root,),
    )

    template = section.template
    assert "Configured host mounts:" in template
    assert "  Contents: `notes.txt`" in template
    assert "  File:" not in template


def test_section_template_marks_empty_directory_mount(tmp_path: Path) -> None:
    allowed_root = tmp_path / "workspace"
    empty_dir = allowed_root / "empty"
    empty_dir.mkdir(parents=True)

    section = _make_section(
        mounts=(HostMount(host_path="empty"),),
        allowed_host_roots=(allowed_root,),
    )

    template = section.template
    assert "  Contents: <empty>" in template


def test_section_template_truncates_mount_preview(tmp_path: Path) -> None:
    allowed_root = tmp_path / "workspace"
    big_dir = allowed_root / "big"
    big_dir.mkdir(parents=True)
    limit = vfs_module._MAX_MOUNT_PREVIEW_ENTRIES
    for index in range(limit + 3):
        (big_dir / f"file{index:02d}.txt").write_text("sample", encoding="utf-8")

    section = _make_section(
        mounts=(HostMount(host_path="big"),),
        allowed_host_roots=(allowed_root,),
    )

    template = section.template
    assert "`file00.txt`" in template
    assert "`file19.txt`" in template
    assert "… (+3 more)" in template


def test_write_file_creates_snapshot(
    session_and_bus: tuple[Session, InProcessEventBus],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _bus = session_and_bus
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("weakincentives.contrib.tools.vfs._now", lambda: timestamp)

    section = _make_section(session=session)
    write_tool = find_tool(section, "write_file")

    params = WriteFileParams(file_path="docs/intro.md", content="hello world")
    invoke_tool(write_tool, params, session=session)

    # Verify file was written to filesystem
    fs = _get_filesystem(section)
    assert fs.exists("docs/intro.md")
    result = fs.read("docs/intro.md")
    assert result.content == "hello world"


def test_ls_lists_directories_and_files(
    session_and_bus: tuple[Session, InProcessEventBus],
) -> None:
    session, _bus = session_and_bus
    section = _make_section(session=session)

    _write(session, section, path=("docs", "intro.md"), content="one")
    _write(session, section, path=("docs", "guide", "setup.md"), content="two")

    list_tool = find_tool(section, "ls")
    params = ListDirectoryParams(path="docs")
    result = invoke_tool(list_tool, params, session=session)

    raw_entries = result.value
    assert raw_entries is not None
    entries = cast(tuple[FileInfo, ...], raw_entries)
    assert [entry.path.segments for entry in entries] == [
        ("docs", "guide"),
        ("docs", "intro.md"),
    ]
    guide_entry, intro_entry = entries
    assert isinstance(guide_entry, FileInfo)
    assert guide_entry.kind == "directory"
    assert intro_entry.kind == "file"


def test_read_file_supports_pagination(
    session_and_bus: tuple[Session, InProcessEventBus],
) -> None:
    session, _bus = session_and_bus
    section = _make_section(session=session)
    _write(
        session,
        section,
        path=("notes.md",),
        content="\n".join(f"line {index}" for index in range(1, 8)),
    )

    read_tool = find_tool(section, "read_file")
    params = ReadFileParams(file_path="notes.md", offset=2, limit=3)
    result = invoke_tool(read_tool, params, session=session)

    payload = result.value
    assert isinstance(payload, ReadFileResult)
    assert payload.offset == 2
    assert payload.limit == 3
    assert payload.total_lines == 7
    assert payload.content.splitlines() == [
        "   3 | line 3",
        "   4 | line 4",
        "   5 | line 5",
    ]


def test_read_file_limit_reports_returned_slice(
    session_and_bus: tuple[Session, InProcessEventBus],
) -> None:
    session, _bus = session_and_bus
    section = _make_section(session=session)
    _write(
        session,
        section,
        path=("notes.md",),
        content="\n".join(f"line {index}" for index in range(1, 6)),
    )

    read_tool = find_tool(section, "read_file")
    params = ReadFileParams(file_path="notes.md", offset=0, limit=10)
    result = invoke_tool(read_tool, params, session=session)

    payload = result.value
    assert isinstance(payload, ReadFileResult)
    assert payload.limit == 5
    assert len(payload.content.splitlines()) == 5


def test_edit_file_replaces_occurrences(
    session_and_bus: tuple[Session, InProcessEventBus],
) -> None:
    session, _bus = session_and_bus
    section = _make_section(session=session)
    _write(
        session,
        section,
        path=("src", "main.py"),
        content="print('old')\nprint('old')",
    )

    edit_tool = find_tool(section, "edit_file")
    params = EditFileParams(
        file_path="src/main.py",
        old_string="old",
        new_string="new",
        replace_all=True,
    )
    invoke_tool(edit_tool, params, session=session)

    # Verify edit was applied to filesystem
    fs = _get_filesystem(section)
    result = fs.read("src/main.py")
    assert result.content == "print('new')\nprint('new')"


def test_glob_filters_matches(
    session_and_bus: tuple[Session, InProcessEventBus],
) -> None:
    session, _bus = session_and_bus
    section = _make_section(session=session)
    _write(session, section, path=("docs", "intro.md"), content="a")
    _write(session, section, path=("docs", "guide.md"), content="b")

    glob_tool = find_tool(section, "glob")
    params = GlobParams(pattern="*.md", path="docs")
    result = invoke_tool(glob_tool, params, session=session)

    raw_matches = result.value
    assert raw_matches is not None
    matches = cast(tuple[GlobMatch, ...], raw_matches)
    assert isinstance(matches[0], GlobMatch)
    assert [match.path.segments for match in matches] == [
        ("docs", "guide.md"),
        ("docs", "intro.md"),
    ]


def test_grep_reports_invalid_pattern(
    session_and_bus: tuple[Session, InProcessEventBus],
) -> None:
    session, _bus = session_and_bus
    section = _make_section(session=session)
    grep_tool = find_tool(section, "grep")

    params = GrepParams(pattern="[", path=None, glob=None)
    result = invoke_tool(grep_tool, params, session=session)

    assert result.success is False
    assert result.value is None
    assert "Invalid regular expression" in result.message


def test_rm_removes_single_file(
    session_and_bus: tuple[Session, InProcessEventBus],
) -> None:
    session, _bus = session_and_bus
    section = _make_section(session=session)
    _write(session, section, path=("myfile.txt",), content="content")

    rm_tool = find_tool(section, "rm")
    params = RemoveParams(path="myfile.txt")
    result = invoke_tool(rm_tool, params, session=session)

    # Verify file was deleted
    fs = _get_filesystem(section)
    assert not fs.exists("myfile.txt")
    assert result.success


def test_rm_removes_directory_tree(
    session_and_bus: tuple[Session, InProcessEventBus],
) -> None:
    session, _bus = session_and_bus
    section = _make_section(session=session)
    _write(session, section, path=("src", "pkg", "module.py"), content="print('hi')")
    _write(session, section, path=("src", "pkg", "util.py"), content="print('hi')")

    rm_tool = find_tool(section, "rm")
    params = RemoveParams(path="src/pkg")
    invoke_tool(rm_tool, params, session=session)

    # Verify files were deleted from the filesystem
    fs = _get_filesystem(section)
    assert not fs.exists("src/pkg/module.py")
    assert not fs.exists("src/pkg/util.py")


def test_write_file_rejects_existing_target(
    session_and_bus: tuple[Session, InProcessEventBus],
) -> None:
    session, _bus = session_and_bus
    section = _make_section(session=session)
    _write(session, section, path=("log.txt",), content="initial")

    write_tool = find_tool(section, "write_file")
    params = WriteFileParams(file_path="log.txt", content="again")

    with pytest.raises(ToolValidationError, match="File already exists"):
        invoke_tool(write_tool, params, session=session)


def test_host_mounts_seed_snapshot(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    file_path = docs / "intro.md"
    file_path.write_text("hello", encoding="utf-8")

    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = _make_section(
        session=session,
        mounts=(
            HostMount(
                host_path="docs",
                mount_path=VfsPath(("workspace",)),
                include_glob=("*.md",),
            ),
        ),
        allowed_host_roots=(tmp_path,),
    )

    list_tool = find_tool(section, "ls")
    params = ListDirectoryParams(path="workspace")
    result = invoke_tool(list_tool, params, session=session)

    raw_entries = result.value
    assert raw_entries is not None
    entries = cast(tuple[FileInfo, ...], raw_entries)
    assert entries[0].path.segments == ("workspace", "intro.md")


def test_ls_rejects_file_path(
    session_and_bus: tuple[Session, InProcessEventBus],
) -> None:
    session, _bus = session_and_bus
    section = _make_section(session=session)
    _write(session, section, path=("notes.md",), content="content")

    list_tool = find_tool(section, "ls")
    params = ListDirectoryParams(path="notes.md")
    with pytest.raises(ToolValidationError, match="Cannot list a file path"):
        invoke_tool(list_tool, params, session=session)


def test_ls_ignores_unrelated_paths(
    session_and_bus: tuple[Session, InProcessEventBus],
) -> None:
    session, _bus = session_and_bus
    section = _make_section(session=session)
    _write(session, section, path=("docs", "intro.md"), content="one")
    _write(session, section, path=("logs", "app.log"), content="two")

    list_tool = find_tool(section, "ls")
    params = ListDirectoryParams(path="docs")
    result = invoke_tool(list_tool, params, session=session)
    raw_entries = result.value
    assert raw_entries is not None
    entries = cast(tuple[FileInfo, ...], raw_entries)
    assert [entry.path.segments for entry in entries] == [("docs", "intro.md")]


def test_read_file_negative_offset(
    session_and_bus: tuple[Session, InProcessEventBus],
) -> None:
    session, _bus = session_and_bus
    section = _make_section(session=session)
    _write(session, section, path=("notes.md",), content="hello")

    read_tool = find_tool(section, "read_file")
    params = ReadFileParams(file_path="notes.md", offset=-1)
    with pytest.raises(ToolValidationError, match="offset must be non-negative"):
        invoke_tool(read_tool, params, session=session)


def test_read_file_invalid_limit(
    session_and_bus: tuple[Session, InProcessEventBus],
) -> None:
    session, _bus = session_and_bus
    section = _make_section(session=session)
    _write(session, section, path=("notes.md",), content="hello")

    read_tool = find_tool(section, "read_file")
    params = ReadFileParams(file_path="notes.md", limit=0)
    with pytest.raises(ToolValidationError, match="limit must be a positive integer"):
        invoke_tool(read_tool, params, session=session)


def test_read_file_returns_empty_slice(
    session_and_bus: tuple[Session, InProcessEventBus],
) -> None:
    session, _bus = session_and_bus
    section = _make_section(session=session)
    _write(session, section, path=("notes.md",), content="line one")

    read_tool = find_tool(section, "read_file")
    params = ReadFileParams(file_path="notes.md", offset=5)
    result = invoke_tool(read_tool, params, session=session)
    payload = result.value
    assert isinstance(payload, ReadFileResult)
    assert payload.content == ""
    assert "no lines returned" in result.message


def test_read_file_missing_path(
    session_and_bus: tuple[Session, InProcessEventBus],
) -> None:
    session, _bus = session_and_bus
    section = _make_section(session=session)
    read_tool = find_tool(section, "read_file")
    params = ReadFileParams(file_path="missing.txt")
    with pytest.raises(ToolValidationError, match="File does not exist"):
        invoke_tool(read_tool, params, session=session)


def test_write_file_content_length_limit(
    session_and_bus: tuple[Session, InProcessEventBus],
) -> None:
    session, _bus = session_and_bus
    section = _make_section(session=session)
    write_tool = find_tool(section, "write_file")
    params = WriteFileParams(file_path="docs/big.txt", content="x" * 48_001)
    with pytest.raises(ToolValidationError, match="Content exceeds"):
        invoke_tool(write_tool, params, session=session)


def test_write_file_rejects_empty_path(
    session_and_bus: tuple[Session, InProcessEventBus],
) -> None:
    session, _bus = session_and_bus
    section = _make_section(session=session)
    write_tool = find_tool(section, "write_file")
    params = WriteFileParams(file_path="", content="data")
    with pytest.raises(ToolValidationError, match="file_path must not be empty"):
        invoke_tool(write_tool, params, session=session)


def test_write_file_accepts_leading_slash(
    session_and_bus: tuple[Session, InProcessEventBus],
) -> None:
    session, _bus = session_and_bus
    section = _make_section(session=session)
    write_tool = find_tool(section, "write_file")
    params = WriteFileParams(file_path="/docs/info.txt", content="data")
    invoke_tool(write_tool, params, session=session)

    # Verify file was written with normalized path
    fs = _get_filesystem(section)
    assert fs.exists("docs/info.txt")


def test_write_file_rejects_relative_segments(
    session_and_bus: tuple[Session, InProcessEventBus],
) -> None:
    session, _bus = session_and_bus
    section = _make_section(session=session)
    write_tool = find_tool(section, "write_file")
    params = WriteFileParams(file_path="docs/../info.txt", content="data")
    with pytest.raises(ToolValidationError, match=r"may not include '\.' or '\.\.'"):
        invoke_tool(write_tool, params, session=session)


def test_write_file_rejects_long_segment(
    session_and_bus: tuple[Session, InProcessEventBus],
) -> None:
    session, _bus = session_and_bus
    section = _make_section(session=session)
    write_tool = find_tool(section, "write_file")
    params = WriteFileParams(file_path=f"{'a' * 81}.txt", content="data")
    with pytest.raises(ToolValidationError, match="80 characters or fewer"):
        invoke_tool(write_tool, params, session=session)


def test_write_file_rejects_excessive_depth(
    session_and_bus: tuple[Session, InProcessEventBus],
) -> None:
    session, _bus = session_and_bus
    section = _make_section(session=session)
    write_tool = find_tool(section, "write_file")
    deep_segments = "/".join(f"dir{index}" for index in range(20))
    params = WriteFileParams(file_path=f"{deep_segments}/file.txt", content="data")
    with pytest.raises(ToolValidationError, match="Path depth exceeds"):
        invoke_tool(write_tool, params, session=session)


def test_edit_file_empty_old_string(
    session_and_bus: tuple[Session, InProcessEventBus],
) -> None:
    session, _bus = session_and_bus
    section = _make_section(session=session)
    _write(session, section, path=("src", "file.py"), content="print('hi')")
    edit_tool = find_tool(section, "edit_file")
    params = EditFileParams(
        file_path="src/file.py",
        old_string="",
        new_string="noop",
    )
    with pytest.raises(ToolValidationError, match="must not be empty"):
        invoke_tool(edit_tool, params, session=session)


def test_edit_file_requires_existing_pattern(
    session_and_bus: tuple[Session, InProcessEventBus],
) -> None:
    session, _bus = session_and_bus
    section = _make_section(session=session)
    _write(session, section, path=("src", "file.py"), content="print('hi')")
    edit_tool = find_tool(section, "edit_file")
    params = EditFileParams(
        file_path="src/file.py",
        old_string="missing",
        new_string="found",
    )
    with pytest.raises(ToolValidationError, match="not found"):
        invoke_tool(edit_tool, params, session=session)


def test_edit_file_requires_unique_match(
    session_and_bus: tuple[Session, InProcessEventBus],
) -> None:
    session, _bus = session_and_bus
    section = _make_section(session=session)
    _write(
        session,
        section,
        path=("src", "file.py"),
        content="print('old')\nprint('old')",
    )
    edit_tool = find_tool(section, "edit_file")
    params = EditFileParams(
        file_path="src/file.py",
        old_string="old",
        new_string="new",
        replace_all=False,
    )
    with pytest.raises(ToolValidationError, match="must match exactly once"):
        invoke_tool(edit_tool, params, session=session)


def test_edit_file_single_occurrence_replace(
    session_and_bus: tuple[Session, InProcessEventBus],
) -> None:
    session, _bus = session_and_bus
    section = _make_section(session=session)
    _write(session, section, path=("src", "file.py"), content="value = 'old'")
    edit_tool = find_tool(section, "edit_file")
    params = EditFileParams(
        file_path="src/file.py",
        old_string="old",
        new_string="new",
        replace_all=False,
    )
    invoke_tool(edit_tool, params, session=session)

    # Verify edit was applied to filesystem
    fs = _get_filesystem(section)
    result = fs.read("src/file.py")
    assert result.content == "value = 'new'"


def test_edit_file_replacement_length_guard(
    session_and_bus: tuple[Session, InProcessEventBus],
) -> None:
    session, _bus = session_and_bus
    section = _make_section(session=session)
    _write(session, section, path=("src", "file.py"), content="flag = 1")
    edit_tool = find_tool(section, "edit_file")
    params = EditFileParams(
        file_path="src/file.py",
        old_string="flag",
        new_string="x" * 48_001,
        replace_all=True,
    )
    with pytest.raises(ToolValidationError, match="48,000 characters or fewer"):
        invoke_tool(edit_tool, params, session=session)


def test_glob_requires_pattern(
    session_and_bus: tuple[Session, InProcessEventBus],
) -> None:
    session, _bus = session_and_bus
    section = _make_section(session=session)
    glob_tool = find_tool(section, "glob")
    params = GlobParams(pattern="", path="/")
    with pytest.raises(ToolValidationError, match="Pattern must not be empty"):
        invoke_tool(glob_tool, params, session=session)


def test_glob_filters_with_base_path(
    session_and_bus: tuple[Session, InProcessEventBus],
) -> None:
    session, _bus = session_and_bus
    section = _make_section(session=session)
    _write(session, section, path=("docs", "intro.md"), content="one")
    _write(session, section, path=("docs", "guide.md"), content="two")
    _write(session, section, path=("notes.txt",), content="three")

    glob_tool = find_tool(section, "glob")
    params = GlobParams(pattern="*.md", path="docs")
    result = invoke_tool(glob_tool, params, session=session)
    raw_matches = result.value
    assert raw_matches is not None
    matches = cast(tuple[GrepMatch, ...], raw_matches)
    assert [match.path.segments for match in matches] == [
        ("docs", "guide.md"),
        ("docs", "intro.md"),
    ]


def test_grep_matches_success(
    session_and_bus: tuple[Session, InProcessEventBus],
) -> None:
    session, _bus = session_and_bus
    section = _make_section(session=session)
    _write(
        session,
        section,
        path=("docs", "intro.md"),
        content="alpha\nbeta\ngamma",
    )
    _write(session, section, path=("docs", "skip.tmp"), content="tmp")
    _write(session, section, path=("notes.txt",), content="delta")
    grep_tool = find_tool(section, "grep")
    params = GrepParams(pattern="a", path="docs", glob="*.md")
    result = invoke_tool(grep_tool, params, session=session)
    raw_matches = result.value
    assert raw_matches is not None
    matches = cast(tuple[GlobMatch, ...], raw_matches)
    assert len(matches) == 3
    assert "matches" in result.message


def test_grep_no_matches_message(
    session_and_bus: tuple[Session, InProcessEventBus],
) -> None:
    session, _bus = session_and_bus
    section = _make_section(session=session)
    _write(session, section, path=("docs", "intro.md"), content="alpha")
    grep_tool = find_tool(section, "grep")
    params = GrepParams(pattern="z", path="docs")
    result = invoke_tool(grep_tool, params, session=session)
    assert result.value == ()
    assert "0 matches" in result.message


def test_rm_requires_existing_path(
    session_and_bus: tuple[Session, InProcessEventBus],
) -> None:
    session, _bus = session_and_bus
    section = _make_section(session=session)
    rm_tool = find_tool(section, "rm")
    params = RemoveParams(path="missing")
    with pytest.raises(ToolValidationError, match="No files matched"):
        invoke_tool(rm_tool, params, session=session)


def test_inmemory_filesystem_supports_append() -> None:
    """Test that InMemoryFilesystem supports append mode."""
    fs = InMemoryFilesystem()
    fs.write("log.txt", "line1", mode="create")
    fs.write("log.txt", "\nline2", mode="append")
    result = fs.read("log.txt")
    assert result.content.endswith("line2")
    assert "line1" in result.content


def test_host_mount_requires_allowed_root(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing"
    with pytest.raises(ToolValidationError, match="Allowed host root does not exist"):
        VfsToolsSection(
            session=Session(bus=InProcessEventBus()),
            allowed_host_roots=(missing_root,),
        )


def test_host_mount_rejects_empty_path(tmp_path: Path) -> None:
    with pytest.raises(ToolValidationError, match="must not be empty"):
        VfsToolsSection(
            session=Session(bus=InProcessEventBus()),
            mounts=(HostMount(host_path=""),),
            allowed_host_roots=(tmp_path,),
        )


def test_host_mount_enforces_max_bytes(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    file_path = docs / "intro.md"
    file_path.write_text("hello", encoding="utf-8")
    with pytest.raises(ToolValidationError, match="byte budget"):
        VfsToolsSection(
            session=Session(bus=InProcessEventBus()),
            mounts=(HostMount(host_path="docs", max_bytes=1, include_glob=("*.md",)),),
            allowed_host_roots=(tmp_path,),
        )


def test_host_mount_defaults_to_relative_destination(tmp_path: Path) -> None:
    file_path = tmp_path / "readme.md"
    file_path.write_text("hi", encoding="utf-8")
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = _make_section(
        session=session,
        mounts=(HostMount(host_path="readme.md"),),
        allowed_host_roots=(tmp_path,),
    )
    list_tool = find_tool(section, "ls")
    result = invoke_tool(list_tool, ListDirectoryParams(), session=session)
    raw_entries = result.value
    assert raw_entries is not None
    entries = cast(tuple[FileInfo, ...], raw_entries)
    assert entries[0].path.segments == ("readme.md",)


def test_host_mount_glob_normalization(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "intro.md").write_text("hello", encoding="utf-8")
    (docs / "skip.tmp").write_text("ignore", encoding="utf-8")
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = _make_section(
        session=session,
        mounts=(
            HostMount(
                host_path="docs",
                include_glob=("", "*.md"),
                exclude_glob=("*.tmp",),
            ),
        ),
        allowed_host_roots=(tmp_path,),
    )
    list_tool = find_tool(section, "ls")
    result = invoke_tool(list_tool, ListDirectoryParams(), session=session)
    raw_entries = result.value
    assert raw_entries is not None
    entries = cast(tuple[FileInfo, ...], raw_entries)
    assert entries[0].path.segments == ("intro.md",)


def test_host_mount_exclude_glob_filters_matches(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "keep.txt").write_text("hello", encoding="utf-8")
    (docs / "skip.log").write_text("ignored", encoding="utf-8")
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = _make_section(
        session=session,
        mounts=(
            HostMount(
                host_path="docs",
                exclude_glob=("*.log",),
            ),
        ),
        allowed_host_roots=(tmp_path,),
    )
    list_tool = find_tool(section, "ls")
    result = invoke_tool(list_tool, ListDirectoryParams(), session=session)

    raw_entries = result.value
    assert raw_entries is not None
    entries = cast(tuple[FileInfo, ...], raw_entries)
    assert [entry.path.segments for entry in entries] == [("keep.txt",)]


def test_normalize_string_path_requires_value() -> None:
    with pytest.raises(ToolValidationError, match="test is required"):
        vfs_module._normalize_string_path(None, allow_empty=False, field="test")


def test_normalize_string_path_allows_empty() -> None:
    result = vfs_module._normalize_string_path("", allow_empty=True, field="path")
    assert result.segments == ()


def test_normalize_string_path_rejects_missing_reference() -> None:
    with pytest.raises(ToolValidationError, match="must reference a file or directory"):
        vfs_module._normalize_string_path("/", allow_empty=False, field="path")


def test_normalize_string_path_returns_segments() -> None:
    result = vfs_module._normalize_string_path(
        "docs/readme.md", allow_empty=False, field="file_path"
    )
    assert result.segments == ("docs", "readme.md")


def test_normalize_segments_rejects_absolute() -> None:
    with pytest.raises(ToolValidationError, match="Absolute paths are not allowed"):
        vfs_module._normalize_segments(("/bad",))


def test_normalize_segments_skips_empty_parts() -> None:
    result = vfs_module._normalize_segments(("dir//sub",))
    assert result == ("dir", "sub")


def test_normalize_optional_path_returns_value() -> None:
    result = vfs_module._normalize_optional_path(VfsPath(("docs",)))
    assert result.segments == ("docs",)


def test_normalize_path_returns_segments() -> None:
    result = vfs_module._normalize_path(VfsPath(("docs", "file.txt")))
    assert result.segments == ("docs", "file.txt")


def test_normalize_path_enforces_depth_limit() -> None:
    deep = tuple(str(index) for index in range(17))
    with pytest.raises(ToolValidationError, match="Path depth exceeds"):
        vfs_module._normalize_path(VfsPath(deep))


def test_host_mount_outside_allowed_root(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    with pytest.raises(ToolValidationError, match="outside the allowed roots"):
        VfsToolsSection(
            session=Session(bus=InProcessEventBus()),
            mounts=(HostMount(host_path="../forbidden.txt"),),
            allowed_host_roots=(allowed,),
        )


def test_host_mount_read_error_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    broken = docs / "broken.txt"
    broken.write_text("payload", encoding="utf-8")

    original_read_text = Path.read_text

    def fail_read_text(
        self: Path, encoding: str | None = None, errors: str | None = None
    ) -> str:
        if self == broken:
            raise OSError("boom")
        return original_read_text(self, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", fail_read_text)

    with pytest.raises(ToolValidationError, match="Failed to read mounted file"):
        VfsToolsSection(
            session=Session(bus=InProcessEventBus()),
            mounts=(HostMount(host_path="docs"),),
            allowed_host_roots=(tmp_path,),
        )


def test_resolve_mount_path_returns_candidate(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    file_path = allowed / "data.txt"
    file_path.write_text("payload", encoding="utf-8")

    resolved = vfs_module._resolve_mount_path("data.txt", (allowed,))
    assert resolved == file_path


def test_resolve_mount_path_requires_allowed_roots() -> None:
    with pytest.raises(ToolValidationError, match="No allowed host roots"):
        vfs_module._resolve_mount_path("data.txt", ())


def test_prompt_filesystem_returns_workspace_section_filesystem() -> None:
    """Test that Prompt.filesystem() returns filesystem from workspace section."""
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    template = PromptTemplate(
        ns="test",
        key="test-prompt",
        sections=(section,),
    )
    prompt = Prompt(template)

    fs = prompt.filesystem()

    assert fs is not None
    assert fs is section.filesystem


def test_prompt_filesystem_returns_none_when_no_workspace() -> None:
    """Test that Prompt.filesystem() returns None without workspace section."""
    section = MarkdownSection(
        title="Test",
        template="Test content",
        key="test",
    )
    template = PromptTemplate(
        ns="test",
        key="test-prompt",
        sections=(section,),
    )
    prompt = Prompt(template)

    assert prompt.filesystem() is None


def test_clone_preserves_filesystem_state() -> None:
    """Test that cloning VfsToolsSection preserves filesystem state."""
    bus1 = InProcessEventBus()
    session1 = Session(bus=bus1)
    section1 = VfsToolsSection(session=session1)

    # Write a file to the filesystem
    fs = section1.filesystem
    assert isinstance(fs, InMemoryFilesystem)
    _ = fs.write("myfile.txt", "my content")

    # Clone the section into a new session
    bus2 = InProcessEventBus()
    session2 = Session(bus=bus2)
    section2 = section1.clone(session=session2)

    # Verify the filesystem is preserved (same instance)
    assert section2.filesystem is section1.filesystem

    # Verify the file content is accessible from the cloned section
    result = section2.filesystem.read("myfile.txt")
    assert result.content == "my content"


def test_clone_requires_session() -> None:
    """Test that clone raises error when session is missing."""
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)

    with pytest.raises(TypeError, match="session is required to clone"):
        section.clone()


def test_clone_requires_matching_bus() -> None:
    """Test that clone raises error when bus doesn't match session's bus."""
    bus1 = InProcessEventBus()
    session1 = Session(bus=bus1)
    section = VfsToolsSection(session=session1)

    bus2 = InProcessEventBus()
    session2 = Session(bus=bus2)
    bus3 = InProcessEventBus()  # Different bus than session2

    with pytest.raises(TypeError, match="Provided bus must match"):
        section.clone(session=session2, bus=bus3)


def test_list_directory_result_render_formats_output() -> None:
    """Test that ListDirectoryResult.render() formats output correctly."""
    path = VfsPath(("subdir",))
    result = ListDirectoryResult(
        path=path,
        directories=("child_dir",),
        files=("file1.txt", "file2.txt"),
    )
    rendered = result.render()

    assert "Directory listing for subdir:" in rendered
    assert "Directories:" in rendered
    assert "- child_dir" in rendered
    assert "Files:" in rendered
    assert "- file1.txt" in rendered
    assert "- file2.txt" in rendered


def test_list_directory_result_render_empty_entries() -> None:
    """Test that ListDirectoryResult.render() handles empty entries."""
    path = VfsPath(())
    result = ListDirectoryResult(
        path=path,
        directories=(),
        files=(),
    )
    rendered = result.render()

    assert "Directory listing for /:" in rendered
    assert "Directories: <none>" in rendered
    assert "Files: <none>" in rendered


def test_edit_file_nonexistent_raises_error() -> None:
    """Test that edit_file raises error for non-existent file."""
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = VfsToolsSection(session=session)
    edit_tool = find_tool(section, "edit_file")

    with pytest.raises(ToolValidationError, match="File does not exist"):
        invoke_tool(
            edit_tool,
            EditFileParams(
                file_path="nonexistent.txt",
                old_string="old",
                new_string="new",
            ),
            session=session,
        )
