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

"""Tests for Codex App Server workspace management."""

from __future__ import annotations

import shutil
import tempfile
import unittest.mock
from pathlib import Path

import pytest

from weakincentives.adapters.codex_app_server.workspace import (
    CodexWorkspaceSection,
    HostMount,
    HostMountPreview,
    WorkspaceBudgetExceededError,
    WorkspaceSecurityError,
    _copy_mount_to_temp,
    _create_workspace,
    _matches_globs,
    _render_workspace_template,
    _resolve_mount_path,
    compute_workspace_fingerprint,
)
from weakincentives.filesystem import HostFilesystem
from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.session import Session


@pytest.fixture
def session() -> Session:
    dispatcher = InProcessDispatcher()
    return Session(dispatcher=dispatcher, tags={"suite": "tests"})


@pytest.fixture
def temp_dir() -> Path:
    """Create a temporary directory for test fixtures."""
    d = Path(tempfile.mkdtemp(prefix="wink-test-"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_source(temp_dir: Path) -> Path:
    """Create a sample source directory with files."""
    src = temp_dir / "source"
    src.mkdir()
    (src / "file1.py").write_text("print('hello')")
    (src / "file2.txt").write_text("readme content")
    sub = src / "subdir"
    sub.mkdir()
    (sub / "nested.py").write_text("# nested")
    return src


class TestMatchesGlobs:
    def test_no_filters_matches_all(self) -> None:
        assert _matches_globs("foo.py", (), ()) is True

    def test_include_matching(self) -> None:
        assert _matches_globs("foo.py", ("*.py",), ()) is True
        assert _matches_globs("foo.txt", ("*.py",), ()) is False

    def test_exclude_matching(self) -> None:
        assert _matches_globs("foo.pyc", (), ("*.pyc",)) is False
        assert _matches_globs("foo.py", (), ("*.pyc",)) is True

    def test_exclude_takes_precedence(self) -> None:
        assert _matches_globs("foo.py", ("*.py",), ("foo.*",)) is False


class TestResolveMountPath:
    def test_existing_path(self, temp_dir: Path) -> None:
        f = temp_dir / "test.txt"
        f.write_text("hello")
        resolved = _resolve_mount_path(str(f), [])
        assert resolved == f.resolve()

    def test_nonexistent_path(self) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            _resolve_mount_path("/nonexistent/path/xyz", [])

    def test_allowed_roots_valid(self, temp_dir: Path) -> None:
        f = temp_dir / "ok.txt"
        f.write_text("data")
        resolved = _resolve_mount_path(str(f), [temp_dir])
        assert resolved == f.resolve()

    def test_allowed_roots_violation(self, temp_dir: Path) -> None:
        f = temp_dir / "ok.txt"
        f.write_text("data")
        other_root = temp_dir / "restricted"
        other_root.mkdir()
        with pytest.raises(WorkspaceSecurityError, match="outside allowed roots"):
            _resolve_mount_path(str(f), [other_root])


class TestCopyMountToTemp:
    def test_single_file(self, temp_dir: Path) -> None:
        src = temp_dir / "src.txt"
        src.write_text("content")
        tgt = temp_dir / "workspace" / "src.txt"
        mount = HostMount(host_path=str(src))

        preview = _copy_mount_to_temp(src, tgt, mount)

        assert tgt.exists()
        assert tgt.read_text() == "content"
        assert preview.is_directory is False
        assert preview.bytes_copied == len("content")
        assert len(preview.entries) == 1

    def test_directory(self, sample_source: Path, temp_dir: Path) -> None:
        tgt = temp_dir / "workspace" / "source"
        tgt.mkdir(parents=True)
        mount = HostMount(host_path=str(sample_source))

        preview = _copy_mount_to_temp(sample_source, tgt, mount)

        assert preview.is_directory is True
        assert len(preview.entries) == 3  # file1.py, file2.txt, subdir/nested.py
        assert preview.bytes_copied > 0

    def test_include_glob_filter(self, sample_source: Path, temp_dir: Path) -> None:
        tgt = temp_dir / "workspace" / "source"
        tgt.mkdir(parents=True)
        mount = HostMount(host_path=str(sample_source), include_glob=("*.py",))

        preview = _copy_mount_to_temp(sample_source, tgt, mount)

        # Only .py files
        assert all(e.endswith(".py") for e in preview.entries)

    def test_exclude_glob_filter(self, sample_source: Path, temp_dir: Path) -> None:
        tgt = temp_dir / "workspace" / "source"
        tgt.mkdir(parents=True)
        mount = HostMount(host_path=str(sample_source), exclude_glob=("*.txt",))

        preview = _copy_mount_to_temp(sample_source, tgt, mount)

        assert all(not e.endswith(".txt") for e in preview.entries)

    def test_max_bytes_single_file(self, temp_dir: Path) -> None:
        src = temp_dir / "big.txt"
        src.write_text("x" * 100)
        tgt = temp_dir / "workspace" / "big.txt"
        mount = HostMount(host_path=str(src), max_bytes=10)

        with pytest.raises(WorkspaceBudgetExceededError, match="byte budget"):
            _copy_mount_to_temp(src, tgt, mount)

    def test_max_bytes_directory(self, sample_source: Path, temp_dir: Path) -> None:
        tgt = temp_dir / "workspace" / "source"
        tgt.mkdir(parents=True)
        mount = HostMount(host_path=str(sample_source), max_bytes=5)

        with pytest.raises(WorkspaceBudgetExceededError, match="byte budget"):
            _copy_mount_to_temp(sample_source, tgt, mount)


class TestCreateWorkspace:
    def test_empty_mounts(self) -> None:
        path, previews = _create_workspace([], allowed_host_roots=[])
        try:
            assert path.exists()
            assert len(previews) == 0
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def test_with_mounts(self, sample_source: Path) -> None:
        mount = HostMount(host_path=str(sample_source))
        path, previews = _create_workspace(
            [mount], allowed_host_roots=[sample_source.parent]
        )
        try:
            assert path.exists()
            assert len(previews) == 1
            assert previews[0].is_directory is True
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def test_custom_mount_path(self, sample_source: Path) -> None:
        mount = HostMount(host_path=str(sample_source), mount_path="my-project")
        path, _previews = _create_workspace(
            [mount], allowed_host_roots=[sample_source.parent]
        )
        try:
            assert (path / "my-project").exists()
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def test_cleanup_on_error(self) -> None:
        """Workspace is cleaned up if mount resolution fails."""
        mount = HostMount(host_path="/nonexistent/path/abc/xyz")
        with pytest.raises(FileNotFoundError):
            _create_workspace([mount], allowed_host_roots=[])


class TestRenderWorkspaceTemplate:
    def test_no_previews(self) -> None:
        result = _render_workspace_template(())
        assert "no host mounts" in result

    def test_with_directory_preview(self) -> None:
        preview = HostMountPreview(
            host_path="/src",
            resolved_host=Path("/src"),
            mount_path="src",
            entries=("file1.py", "file2.py"),
            is_directory=True,
            bytes_copied=100,
        )
        result = _render_workspace_template((preview,))
        assert "src" in result
        assert "directory" in result
        assert "file1.py" in result
        assert "100" in result

    def test_with_file_preview(self) -> None:
        preview = HostMountPreview(
            host_path="/src/main.py",
            resolved_host=Path("/src/main.py"),
            mount_path="main.py",
            entries=("main.py",),
            is_directory=False,
            bytes_copied=42,
        )
        result = _render_workspace_template((preview,))
        assert "file" in result
        assert "main.py" in result

    def test_many_entries_truncated(self) -> None:
        entries = tuple(f"file{i}.py" for i in range(25))
        preview = HostMountPreview(
            host_path="/src",
            resolved_host=Path("/src"),
            mount_path="src",
            entries=entries,
            is_directory=True,
            bytes_copied=1000,
        )
        result = _render_workspace_template((preview,))
        assert "more" in result


class TestComputeWorkspaceFingerprint:
    def test_deterministic(self) -> None:
        mounts = (HostMount(host_path="/src"),)
        fp1 = compute_workspace_fingerprint(mounts)
        fp2 = compute_workspace_fingerprint(mounts)
        assert fp1 == fp2
        assert len(fp1) == 16

    def test_different_mounts_different_fingerprint(self) -> None:
        fp1 = compute_workspace_fingerprint((HostMount(host_path="/a"),))
        fp2 = compute_workspace_fingerprint((HostMount(host_path="/b"),))
        assert fp1 != fp2

    def test_empty_mounts(self) -> None:
        fp = compute_workspace_fingerprint(())
        assert len(fp) == 16


class TestCodexWorkspaceSection:
    def test_no_mounts(self, session: Session) -> None:
        section = CodexWorkspaceSection(session=session)
        try:
            assert section.temp_dir.exists()
            assert section.mount_previews == ()
            assert isinstance(section.filesystem, HostFilesystem)
        finally:
            section.cleanup()

    def test_with_mounts(self, session: Session, sample_source: Path) -> None:
        mount = HostMount(host_path=str(sample_source))
        section = CodexWorkspaceSection(
            session=session,
            mounts=[mount],
            allowed_host_roots=[sample_source.parent],
        )
        try:
            assert section.temp_dir.exists()
            assert len(section.mount_previews) == 1
        finally:
            section.cleanup()

    def test_session_property(self, session: Session) -> None:
        section = CodexWorkspaceSection(session=session)
        try:
            assert section.session is session
        finally:
            section.cleanup()

    def test_fingerprint(self, session: Session) -> None:
        section = CodexWorkspaceSection(session=session)
        try:
            fp = section.workspace_fingerprint
            assert isinstance(fp, str)
            assert len(fp) == 16
        finally:
            section.cleanup()

    def test_resources_provides_filesystem(self, session: Session) -> None:
        section = CodexWorkspaceSection(session=session)
        try:
            registry = section.resources()
            assert registry is not None
        finally:
            section.cleanup()

    def test_cleanup(self, session: Session) -> None:
        section = CodexWorkspaceSection(session=session)
        temp = section.temp_dir
        assert temp.exists()
        section.cleanup()
        assert not temp.exists()

    def test_clone(self, session: Session) -> None:
        section = CodexWorkspaceSection(session=session)
        try:
            new_dispatcher = InProcessDispatcher()
            new_session = Session(dispatcher=new_dispatcher, tags={"suite": "tests"})
            cloned = section.clone(session=new_session)
            assert cloned.session is new_session
            assert cloned.temp_dir == section.temp_dir
        finally:
            section.cleanup()

    def test_clone_requires_session(self, session: Session) -> None:
        section = CodexWorkspaceSection(session=session)
        try:
            with pytest.raises(TypeError, match="session is required"):
                section.clone(session="not a session")
        finally:
            section.cleanup()

    def test_clone_dispatcher_mismatch(self, session: Session) -> None:
        section = CodexWorkspaceSection(session=session)
        try:
            other_dispatcher = InProcessDispatcher()
            new_session = Session(
                dispatcher=InProcessDispatcher(), tags={"suite": "tests"}
            )
            with pytest.raises(TypeError, match="dispatcher must match"):
                section.clone(session=new_session, dispatcher=other_dispatcher)
        finally:
            section.cleanup()

    def test_pre_created_workspace(self, session: Session, temp_dir: Path) -> None:
        """Section can be created with pre-existing workspace data."""
        workspace = temp_dir / "prebuilt"
        workspace.mkdir()
        previews = (
            HostMountPreview(
                host_path="/src",
                resolved_host=Path("/src"),
                mount_path="src",
                entries=(),
                is_directory=True,
                bytes_copied=0,
            ),
        )
        section = CodexWorkspaceSection(
            session=session,
            _temp_dir=workspace,
            _mount_previews=previews,
        )
        assert section.temp_dir == workspace
        assert section.mount_previews == previews

    def test_created_at_timestamp(self, session: Session) -> None:
        section = CodexWorkspaceSection(session=session)
        try:
            assert section.created_at is not None
        finally:
            section.cleanup()

    def test_cleanup_already_cleaned(self, session: Session) -> None:
        """Calling cleanup twice should not raise."""
        section = CodexWorkspaceSection(session=session)
        section.cleanup()
        # Second cleanup should be safe
        section.cleanup()


class TestPathTraversal:
    def test_mount_path_traversal_rejected(self, sample_source: Path) -> None:
        mount = HostMount(host_path=str(sample_source), mount_path="../../escape")
        with pytest.raises(WorkspaceSecurityError, match="escapes workspace"):
            _create_workspace([mount], allowed_host_roots=[sample_source.parent])


class TestSymlinkEscape:
    def test_symlink_escape_skipped(self, temp_dir: Path) -> None:
        """Symlinks pointing outside source are skipped when follow_symlinks=True."""
        src = temp_dir / "source"
        src.mkdir()
        (src / "legit.txt").write_text("ok")

        outside = temp_dir / "outside.txt"
        outside.write_text("secret")
        (src / "escape_link").symlink_to(outside)

        tgt = temp_dir / "workspace" / "source"
        tgt.mkdir(parents=True)
        mount = HostMount(host_path=str(src), follow_symlinks=True)

        preview = _copy_mount_to_temp(src, tgt, mount)

        assert "legit.txt" in preview.entries
        assert "escape_link" not in preview.entries

    def test_copy_preserves_follow_symlinks_false(self, temp_dir: Path) -> None:
        """shutil.copy2 called with follow_symlinks=False for single file."""
        src = temp_dir / "src.txt"
        src.write_text("content")
        tgt = temp_dir / "workspace" / "src.txt"
        mount = HostMount(host_path=str(src), follow_symlinks=False)

        with unittest.mock.patch("shutil.copy2", wraps=shutil.copy2) as mock_copy:
            _copy_mount_to_temp(src, tgt, mount)
            mock_copy.assert_called_once()
            assert mock_copy.call_args[1].get("follow_symlinks") is False

    def test_symlink_file_skipped_when_not_following(self, temp_dir: Path) -> None:
        """Symlink files are skipped for directory mounts when follow_symlinks=False."""
        src = temp_dir / "source"
        src.mkdir()
        (src / "kept.txt").write_text("ok")

        outside = temp_dir / "outside.txt"
        outside.write_text("x" * 100)
        (src / "escape_link").symlink_to(outside)

        tgt = temp_dir / "workspace" / "source"
        tgt.mkdir(parents=True)
        mount = HostMount(host_path=str(src), follow_symlinks=False, max_bytes=2)

        preview = _copy_mount_to_temp(src, tgt, mount)

        assert preview.entries == ("kept.txt",)
        assert preview.bytes_copied == 2
        assert not (tgt / "escape_link").exists()


class TestMaxBytesZero:
    def test_max_bytes_zero_rejects_files(self, temp_dir: Path) -> None:
        """max_bytes=0 should reject any file."""
        src = temp_dir / "tiny.txt"
        src.write_text("a")
        tgt = temp_dir / "workspace" / "tiny.txt"
        mount = HostMount(host_path=str(src), max_bytes=0)

        with pytest.raises(WorkspaceBudgetExceededError, match="byte budget"):
            _copy_mount_to_temp(src, tgt, mount)

    def test_max_bytes_zero_rejects_directory_files(
        self, sample_source: Path, temp_dir: Path
    ) -> None:
        """max_bytes=0 should reject files in a directory walk."""
        tgt = temp_dir / "workspace" / "source"
        tgt.mkdir(parents=True)
        mount = HostMount(host_path=str(sample_source), max_bytes=0)

        with pytest.raises(WorkspaceBudgetExceededError, match="byte budget"):
            _copy_mount_to_temp(sample_source, tgt, mount)
