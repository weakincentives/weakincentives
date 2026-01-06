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

"""Integration tests for filesystem operations across workspace types.

This module tests filesystem tools (ls, read_file, write_file, edit_file,
glob, grep, rm) work correctly across all workspace implementations:
- VfsToolsSection (InMemoryFilesystem)
- PodmanSandboxSection (HostFilesystem)
- Standalone InMemoryFilesystem
- Standalone HostFilesystem
"""

# pyright: reportOptionalCall=false, reportInvalidTypeForm=false

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from tests.tools.helpers import build_tool_context, find_tool
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
    PodmanSandboxConfig,
    PodmanSandboxSection,
    ReadFileParams,
    ReadFileResult,
    RemoveParams,
    VfsPath,
    VfsToolsSection,
    WriteFile,
    WriteFileParams,
)
from weakincentives.contrib.tools.vfs import FilesystemToolHandlers
from weakincentives.filesystem import Filesystem, HostFilesystem
from weakincentives.prompt.tool import ToolContext
from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.session import Session

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def session() -> Session:
    """Create a fresh session for each test."""
    return Session(bus=InProcessDispatcher())


@pytest.fixture
def vfs_section(session: Session) -> VfsToolsSection:
    """Create a VfsToolsSection with InMemoryFilesystem."""
    return VfsToolsSection(session=session)


@pytest.fixture
def host_fs(tmp_path: Path) -> HostFilesystem:
    """Create a HostFilesystem rooted at a temp directory."""
    return HostFilesystem(_root=str(tmp_path))


@pytest.fixture
def memory_fs() -> InMemoryFilesystem:
    """Create a standalone InMemoryFilesystem."""
    return InMemoryFilesystem()


def _make_context(session: Session, filesystem: Filesystem) -> ToolContext:
    """Create a ToolContext with the given filesystem."""
    return build_tool_context(session, filesystem=filesystem)


# ---------------------------------------------------------------------------
# VfsToolsSection (InMemoryFilesystem) Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestVfsToolsSectionFilesystem:
    """Integration tests for VfsToolsSection filesystem operations."""

    def test_write_read_roundtrip(
        self, session: Session, vfs_section: VfsToolsSection
    ) -> None:
        """Write a file and read it back."""
        write_tool = find_tool(vfs_section, "write_file")
        read_tool = find_tool(vfs_section, "read_file")
        ctx = _make_context(session, vfs_section.filesystem)

        # Write
        write_result = write_tool.handler(
            WriteFileParams(file_path="test.txt", content="hello world"),
            context=ctx,
        )
        assert write_result.success
        assert write_result.value is not None
        write_value = cast(WriteFile, write_result.value)
        assert write_value.mode == "create"

        # Read
        read_result = read_tool.handler(
            ReadFileParams(file_path="test.txt"),
            context=ctx,
        )
        assert read_result.success
        assert read_result.value is not None
        read_value = cast(ReadFileResult, read_result.value)
        assert "hello world" in read_value.content

    def test_edit_file(self, session: Session, vfs_section: VfsToolsSection) -> None:
        """Edit an existing file."""
        write_tool = find_tool(vfs_section, "write_file")
        edit_tool = find_tool(vfs_section, "edit_file")
        read_tool = find_tool(vfs_section, "read_file")
        ctx = _make_context(session, vfs_section.filesystem)

        # Write initial content
        write_tool.handler(
            WriteFileParams(file_path="edit_me.txt", content="foo bar baz"),
            context=ctx,
        )

        # Edit
        edit_result = edit_tool.handler(
            EditFileParams(
                file_path="edit_me.txt",
                old_string="bar",
                new_string="BAR",
            ),
            context=ctx,
        )
        assert edit_result.success

        # Verify
        read_result = read_tool.handler(
            ReadFileParams(file_path="edit_me.txt"),
            context=ctx,
        )
        assert read_result.success
        read_value = cast(ReadFileResult, read_result.value)
        assert "foo BAR baz" in read_value.content

    def test_list_directory(
        self, session: Session, vfs_section: VfsToolsSection
    ) -> None:
        """List directory contents."""
        write_tool = find_tool(vfs_section, "write_file")
        ls_tool = find_tool(vfs_section, "ls")
        ctx = _make_context(session, vfs_section.filesystem)

        # Create files
        write_tool.handler(
            WriteFileParams(file_path="src/main.py", content="# main"),
            context=ctx,
        )
        write_tool.handler(
            WriteFileParams(file_path="src/utils.py", content="# utils"),
            context=ctx,
        )

        # List
        ls_result = ls_tool.handler(
            ListDirectoryParams(path="src"),
            context=ctx,
        )
        assert ls_result.success
        assert ls_result.value is not None
        entries = cast(tuple[FileInfo, ...], ls_result.value)
        names = [e.path.segments[-1] for e in entries]
        assert "main.py" in names
        assert "utils.py" in names

    def test_glob_pattern(self, session: Session, vfs_section: VfsToolsSection) -> None:
        """Search files using glob patterns."""
        write_tool = find_tool(vfs_section, "write_file")
        glob_tool = find_tool(vfs_section, "glob")
        ctx = _make_context(session, vfs_section.filesystem)

        # Create files
        write_tool.handler(
            WriteFileParams(file_path="src/app.py", content="# app"),
            context=ctx,
        )
        write_tool.handler(
            WriteFileParams(file_path="src/test.py", content="# test"),
            context=ctx,
        )
        write_tool.handler(
            WriteFileParams(file_path="docs/readme.md", content="# docs"),
            context=ctx,
        )

        # Glob for Python files
        glob_result = glob_tool.handler(
            GlobParams(pattern="**/*.py", path=""),
            context=ctx,
        )
        assert glob_result.success
        matches = cast(tuple[GlobMatch, ...], glob_result.value)
        paths = ["/".join(m.path.segments) for m in matches]
        assert any("app.py" in p for p in paths)
        assert any("test.py" in p for p in paths)
        assert not any("readme.md" in p for p in paths)

    def test_grep_search(self, session: Session, vfs_section: VfsToolsSection) -> None:
        """Search file contents using grep."""
        write_tool = find_tool(vfs_section, "write_file")
        grep_tool = find_tool(vfs_section, "grep")
        ctx = _make_context(session, vfs_section.filesystem)

        # Create files with content
        write_tool.handler(
            WriteFileParams(
                file_path="src/main.py",
                content="def main():\n    print('hello')\n",
            ),
            context=ctx,
        )
        write_tool.handler(
            WriteFileParams(
                file_path="src/utils.py",
                content="def helper():\n    return 42\n",
            ),
            context=ctx,
        )

        # Grep for 'def'
        grep_result = grep_tool.handler(
            GrepParams(pattern="def ", path=""),
            context=ctx,
        )
        assert grep_result.success
        matches = cast(tuple[GrepMatch, ...], grep_result.value)
        assert len(matches) == 2

    def test_remove_file(self, session: Session, vfs_section: VfsToolsSection) -> None:
        """Remove a file."""
        write_tool = find_tool(vfs_section, "write_file")
        rm_tool = find_tool(vfs_section, "rm")
        read_tool = find_tool(vfs_section, "read_file")
        ctx = _make_context(session, vfs_section.filesystem)

        # Create and verify file exists
        write_tool.handler(
            WriteFileParams(file_path="to_delete.txt", content="delete me"),
            context=ctx,
        )
        read_result = read_tool.handler(
            ReadFileParams(file_path="to_delete.txt"),
            context=ctx,
        )
        assert read_result.success

        # Remove
        rm_result = rm_tool.handler(
            RemoveParams(path="to_delete.txt"),
            context=ctx,
        )
        assert rm_result.success
        rm_value = cast(DeleteEntry, rm_result.value)
        assert rm_value.path.segments == ("to_delete.txt",)

        # Verify removed
        from weakincentives.errors import ToolValidationError

        with pytest.raises(ToolValidationError):
            read_tool.handler(
                ReadFileParams(file_path="to_delete.txt"),
                context=ctx,
            )


# ---------------------------------------------------------------------------
# Standalone InMemoryFilesystem Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestInMemoryFilesystemDirect:
    """Integration tests for InMemoryFilesystem used directly."""

    def test_write_read_roundtrip(
        self, session: Session, memory_fs: InMemoryFilesystem
    ) -> None:
        """Write and read using the filesystem directly."""
        handlers = FilesystemToolHandlers()
        ctx = _make_context(session, memory_fs)

        # Write
        write_result = handlers.write_file(
            WriteFileParams(file_path="direct.txt", content="direct write"),
            context=ctx,
        )
        assert write_result.success

        # Read
        read_result = handlers.read_file(
            ReadFileParams(file_path="direct.txt"),
            context=ctx,
        )
        assert read_result.success
        read_value = cast(ReadFileResult, read_result.value)
        assert "direct write" in read_value.content

    def test_nested_directories(
        self, session: Session, memory_fs: InMemoryFilesystem
    ) -> None:
        """Create files in nested directories."""
        handlers = FilesystemToolHandlers()
        ctx = _make_context(session, memory_fs)

        # Write to nested path
        write_result = handlers.write_file(
            WriteFileParams(
                file_path="a/b/c/deep.txt",
                content="deeply nested",
            ),
            context=ctx,
        )
        assert write_result.success

        # List intermediate directory
        ls_result = handlers.list_directory(
            ListDirectoryParams(path="a/b"),
            context=ctx,
        )
        assert ls_result.success
        entries = cast(tuple[FileInfo, ...], ls_result.value)
        assert any(e.path.segments[-1] == "c" for e in entries)


# ---------------------------------------------------------------------------
# Standalone HostFilesystem Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestHostFilesystemDirect:
    """Integration tests for HostFilesystem used directly."""

    def test_write_read_roundtrip(
        self, session: Session, host_fs: HostFilesystem
    ) -> None:
        """Write and read using HostFilesystem."""
        handlers = FilesystemToolHandlers()
        ctx = _make_context(session, host_fs)

        # Write
        write_result = handlers.write_file(
            WriteFileParams(file_path="host_test.txt", content="host content"),
            context=ctx,
        )
        assert write_result.success

        # Read
        read_result = handlers.read_file(
            ReadFileParams(file_path="host_test.txt"),
            context=ctx,
        )
        assert read_result.success
        read_value = cast(ReadFileResult, read_result.value)
        assert "host content" in read_value.content

    def test_glob_on_host(self, session: Session, host_fs: HostFilesystem) -> None:
        """Glob patterns work on HostFilesystem."""
        handlers = FilesystemToolHandlers()
        ctx = _make_context(session, host_fs)

        # Create files
        handlers.write_file(
            WriteFileParams(file_path="code/app.py", content="# app"),
            context=ctx,
        )
        handlers.write_file(
            WriteFileParams(file_path="code/lib.py", content="# lib"),
            context=ctx,
        )
        handlers.write_file(
            WriteFileParams(file_path="data/config.json", content="{}"),
            context=ctx,
        )

        # Glob for Python files
        glob_result = handlers.glob(
            GlobParams(pattern="**/*.py", path=""),
            context=ctx,
        )
        assert glob_result.success
        matches = cast(tuple[GlobMatch, ...], glob_result.value)
        assert len(matches) == 2

    def test_grep_on_host(self, session: Session, host_fs: HostFilesystem) -> None:
        """Grep patterns work on HostFilesystem."""
        handlers = FilesystemToolHandlers()
        ctx = _make_context(session, host_fs)

        # Create files with searchable content
        handlers.write_file(
            WriteFileParams(
                file_path="search/file1.txt",
                content="line one\nfind me here\nline three",
            ),
            context=ctx,
        )
        handlers.write_file(
            WriteFileParams(
                file_path="search/file2.txt",
                content="nothing to see\nfind me too\n",
            ),
            context=ctx,
        )

        # Grep
        grep_result = handlers.grep(
            GrepParams(pattern="find me", path="search"),
            context=ctx,
        )
        assert grep_result.success
        matches = cast(tuple[GrepMatch, ...], grep_result.value)
        assert len(matches) == 2

    def test_edit_on_host(self, session: Session, host_fs: HostFilesystem) -> None:
        """Edit files on HostFilesystem."""
        handlers = FilesystemToolHandlers()
        ctx = _make_context(session, host_fs)

        # Write
        handlers.write_file(
            WriteFileParams(file_path="editable.txt", content="old value"),
            context=ctx,
        )

        # Edit
        edit_result = handlers.edit_file(
            EditFileParams(
                file_path="editable.txt",
                old_string="old value",
                new_string="new value",
            ),
            context=ctx,
        )
        assert edit_result.success

        # Verify
        read_result = handlers.read_file(
            ReadFileParams(file_path="editable.txt"),
            context=ctx,
        )
        read_value = cast(ReadFileResult, read_result.value)
        assert "new value" in read_value.content

    def test_remove_on_host(self, session: Session, host_fs: HostFilesystem) -> None:
        """Remove files on HostFilesystem."""
        handlers = FilesystemToolHandlers()
        ctx = _make_context(session, host_fs)

        # Write
        handlers.write_file(
            WriteFileParams(file_path="removable.txt", content="remove me"),
            context=ctx,
        )

        # Verify exists
        read_result = handlers.read_file(
            ReadFileParams(file_path="removable.txt"),
            context=ctx,
        )
        assert read_result.success

        # Remove
        rm_result = handlers.remove(
            RemoveParams(path="removable.txt"),
            context=ctx,
        )
        assert rm_result.success

        # Verify removed
        from weakincentives.errors import ToolValidationError

        with pytest.raises(ToolValidationError):
            handlers.read_file(
                ReadFileParams(file_path="removable.txt"),
                context=ctx,
            )


# ---------------------------------------------------------------------------
# PodmanSandboxSection Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.podman
class TestPodmanFilesystem:
    """Integration tests for PodmanSandboxSection filesystem operations."""

    @pytest.fixture
    def podman_section(
        self, session: Session, tmp_path: Path
    ) -> PodmanSandboxSection | None:
        """Create a PodmanSandboxSection if podman is available."""
        connection = PodmanSandboxSection.resolve_connection()
        if connection is None:
            return None
        return PodmanSandboxSection(
            session=session,
            config=PodmanSandboxConfig(cache_dir=tmp_path),
        )

    def test_write_read_roundtrip(self, session: Session, tmp_path: Path) -> None:
        """Write and read files through Podman filesystem."""
        connection = PodmanSandboxSection.resolve_connection()
        if connection is None:
            pytest.skip("Podman integration requires a running podman machine.")

        section = PodmanSandboxSection(
            session=session,
            config=PodmanSandboxConfig(cache_dir=tmp_path),
        )
        try:
            write_tool = find_tool(section, "write_file")
            read_tool = find_tool(section, "read_file")
            ctx = _make_context(session, section.filesystem)

            # Write
            write_result = write_tool.handler(
                WriteFileParams(file_path="podman_test.txt", content="podman content"),
                context=ctx,
            )
            assert write_result.success

            # Read
            read_result = read_tool.handler(
                ReadFileParams(file_path="podman_test.txt"),
                context=ctx,
            )
            assert read_result.success
            read_value = cast(ReadFileResult, read_result.value)
            assert "podman content" in read_value.content
        finally:
            section.close()

    def test_edit_file(self, session: Session, tmp_path: Path) -> None:
        """Edit files through Podman filesystem."""
        connection = PodmanSandboxSection.resolve_connection()
        if connection is None:
            pytest.skip("Podman integration requires a running podman machine.")

        section = PodmanSandboxSection(
            session=session,
            config=PodmanSandboxConfig(cache_dir=tmp_path),
        )
        try:
            write_tool = find_tool(section, "write_file")
            edit_tool = find_tool(section, "edit_file")
            read_tool = find_tool(section, "read_file")
            ctx = _make_context(session, section.filesystem)

            # Write
            write_tool.handler(
                WriteFileParams(file_path="edit_test.txt", content="before edit"),
                context=ctx,
            )

            # Edit
            edit_result = edit_tool.handler(
                EditFileParams(
                    file_path="edit_test.txt",
                    old_string="before",
                    new_string="after",
                ),
                context=ctx,
            )
            assert edit_result.success

            # Verify
            read_result = read_tool.handler(
                ReadFileParams(file_path="edit_test.txt"),
                context=ctx,
            )
            read_value = cast(ReadFileResult, read_result.value)
            assert "after edit" in read_value.content
        finally:
            section.close()

    def test_glob_search(self, session: Session, tmp_path: Path) -> None:
        """Glob search through Podman filesystem."""
        connection = PodmanSandboxSection.resolve_connection()
        if connection is None:
            pytest.skip("Podman integration requires a running podman machine.")

        section = PodmanSandboxSection(
            session=session,
            config=PodmanSandboxConfig(cache_dir=tmp_path),
        )
        try:
            write_tool = find_tool(section, "write_file")
            glob_tool = find_tool(section, "glob")
            ctx = _make_context(session, section.filesystem)

            # Create files
            write_tool.handler(
                WriteFileParams(file_path="src/main.py", content="# main"),
                context=ctx,
            )
            write_tool.handler(
                WriteFileParams(file_path="src/utils.py", content="# utils"),
                context=ctx,
            )
            write_tool.handler(
                WriteFileParams(file_path="readme.md", content="# readme"),
                context=ctx,
            )

            # Glob
            glob_result = glob_tool.handler(
                GlobParams(pattern="**/*.py", path=""),
                context=ctx,
            )
            assert glob_result.success
            matches = cast(tuple[GlobMatch, ...], glob_result.value)
            assert len(matches) == 2
        finally:
            section.close()

    def test_grep_search(self, session: Session, tmp_path: Path) -> None:
        """Grep search through Podman filesystem."""
        connection = PodmanSandboxSection.resolve_connection()
        if connection is None:
            pytest.skip("Podman integration requires a running podman machine.")

        section = PodmanSandboxSection(
            session=session,
            config=PodmanSandboxConfig(cache_dir=tmp_path),
        )
        try:
            write_tool = find_tool(section, "write_file")
            grep_tool = find_tool(section, "grep")
            ctx = _make_context(session, section.filesystem)

            # Create files
            write_tool.handler(
                WriteFileParams(
                    file_path="code/app.py",
                    content="def main():\n    pass\n",
                ),
                context=ctx,
            )
            write_tool.handler(
                WriteFileParams(
                    file_path="code/lib.py",
                    content="def helper():\n    pass\n",
                ),
                context=ctx,
            )

            # Grep
            grep_result = grep_tool.handler(
                GrepParams(pattern="def ", path=""),
                context=ctx,
            )
            assert grep_result.success
            matches = cast(tuple[GrepMatch, ...], grep_result.value)
            assert len(matches) == 2
        finally:
            section.close()

    def test_list_directory(self, session: Session, tmp_path: Path) -> None:
        """List directory through Podman filesystem."""
        connection = PodmanSandboxSection.resolve_connection()
        if connection is None:
            pytest.skip("Podman integration requires a running podman machine.")

        section = PodmanSandboxSection(
            session=session,
            config=PodmanSandboxConfig(cache_dir=tmp_path),
        )
        try:
            write_tool = find_tool(section, "write_file")
            ls_tool = find_tool(section, "ls")
            ctx = _make_context(session, section.filesystem)

            # Create files
            write_tool.handler(
                WriteFileParams(file_path="dir/file1.txt", content="one"),
                context=ctx,
            )
            write_tool.handler(
                WriteFileParams(file_path="dir/file2.txt", content="two"),
                context=ctx,
            )

            # List
            ls_result = ls_tool.handler(
                ListDirectoryParams(path="dir"),
                context=ctx,
            )
            assert ls_result.success
            entries = cast(tuple[FileInfo, ...], ls_result.value)
            names = [e.path.segments[-1] for e in entries]
            assert "file1.txt" in names
            assert "file2.txt" in names
        finally:
            section.close()

    def test_remove_file(self, session: Session, tmp_path: Path) -> None:
        """Remove file through Podman filesystem."""
        connection = PodmanSandboxSection.resolve_connection()
        if connection is None:
            pytest.skip("Podman integration requires a running podman machine.")

        section = PodmanSandboxSection(
            session=session,
            config=PodmanSandboxConfig(cache_dir=tmp_path),
        )
        try:
            write_tool = find_tool(section, "write_file")
            rm_tool = find_tool(section, "rm")
            read_tool = find_tool(section, "read_file")
            ctx = _make_context(session, section.filesystem)

            # Write
            write_tool.handler(
                WriteFileParams(file_path="to_remove.txt", content="remove me"),
                context=ctx,
            )

            # Verify exists
            read_result = read_tool.handler(
                ReadFileParams(file_path="to_remove.txt"),
                context=ctx,
            )
            assert read_result.success

            # Remove
            rm_result = rm_tool.handler(
                RemoveParams(path="to_remove.txt"),
                context=ctx,
            )
            assert rm_result.success

            # Verify removed
            from weakincentives.errors import ToolValidationError

            with pytest.raises(ToolValidationError):
                read_tool.handler(
                    ReadFileParams(file_path="to_remove.txt"),
                    context=ctx,
                )
        finally:
            section.close()


# ---------------------------------------------------------------------------
# VFS with Host Mounts Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestVfsWithHostMounts:
    """Integration tests for VfsToolsSection with host mounts."""

    def test_mount_host_directory(self, session: Session, tmp_path: Path) -> None:
        """Mount a host directory and read its contents."""
        # Create files on host
        host_dir = tmp_path / "host_files"
        host_dir.mkdir()
        (host_dir / "existing.txt").write_text("from host", encoding="utf-8")
        (host_dir / "subdir").mkdir()
        (host_dir / "subdir" / "nested.txt").write_text("nested", encoding="utf-8")

        # Create VFS section with mount
        section = VfsToolsSection(
            session=session,
            mounts=(
                HostMount(
                    host_path=str(host_dir),
                    mount_path=VfsPath(("mounted",)),
                ),
            ),
            allowed_host_roots=(str(tmp_path),),
        )

        read_tool = find_tool(section, "read_file")
        ls_tool = find_tool(section, "ls")
        ctx = _make_context(session, section.filesystem)

        # Read mounted file
        read_result = read_tool.handler(
            ReadFileParams(file_path="mounted/existing.txt"),
            context=ctx,
        )
        assert read_result.success
        read_value = cast(ReadFileResult, read_result.value)
        assert "from host" in read_value.content

        # List mounted directory
        ls_result = ls_tool.handler(
            ListDirectoryParams(path="mounted"),
            context=ctx,
        )
        assert ls_result.success
        entries = cast(tuple[FileInfo, ...], ls_result.value)
        names = [e.path.segments[-1] for e in entries]
        assert "existing.txt" in names
        assert "subdir" in names

    def test_write_to_mounted_location(self, session: Session, tmp_path: Path) -> None:
        """Write to VFS doesn't affect host (in-memory only)."""
        # Create host directory
        host_dir = tmp_path / "host_files"
        host_dir.mkdir()

        # Create VFS section with mount
        section = VfsToolsSection(
            session=session,
            mounts=(
                HostMount(
                    host_path=str(host_dir),
                    mount_path=VfsPath(("mounted",)),
                ),
            ),
            allowed_host_roots=(str(tmp_path),),
        )

        write_tool = find_tool(section, "write_file")
        read_tool = find_tool(section, "read_file")
        ctx = _make_context(session, section.filesystem)

        # Write to VFS
        write_result = write_tool.handler(
            WriteFileParams(file_path="mounted/new_file.txt", content="vfs content"),
            context=ctx,
        )
        assert write_result.success

        # Read back from VFS
        read_result = read_tool.handler(
            ReadFileParams(file_path="mounted/new_file.txt"),
            context=ctx,
        )
        assert read_result.success

        # Verify host is NOT modified (VFS is in-memory)
        host_file = host_dir / "new_file.txt"
        assert not host_file.exists()
