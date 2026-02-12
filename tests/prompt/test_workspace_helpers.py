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

"""Tests for workspace helper functions and data types."""

from __future__ import annotations

import shutil
import tempfile
import unittest.mock
from pathlib import Path

import pytest

from weakincentives.prompt.workspace import (
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


@pytest.fixture
def temp_dir() -> Path:
    """Create a temporary directory for test fixtures."""
    d = Path(tempfile.mkdtemp(prefix="wink-test-"))
    yield d  # type: ignore[misc]
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


# ---------------------------------------------------------------------------
# HostMount / HostMountPreview
# ---------------------------------------------------------------------------


class TestHostMount:
    def test_defaults(self) -> None:
        mount = HostMount(host_path="/home/user/src")
        assert mount.host_path == "/home/user/src"
        assert mount.mount_path is None
        assert mount.include_glob == ()
        assert mount.exclude_glob == ()
        assert mount.max_bytes is None
        assert mount.follow_symlinks is False

    def test_with_all_options(self) -> None:
        mount = HostMount(
            host_path="/home/user/src",
            mount_path="project/src",
            include_glob=("*.py", "*.txt"),
            exclude_glob=("__pycache__/*",),
            max_bytes=1000000,
            follow_symlinks=True,
        )
        assert mount.host_path == "/home/user/src"
        assert mount.mount_path == "project/src"
        assert mount.include_glob == ("*.py", "*.txt")
        assert mount.exclude_glob == ("__pycache__/*",)
        assert mount.max_bytes == 1000000
        assert mount.follow_symlinks is True


class TestHostMountPreview:
    def test_construction(self) -> None:
        preview = HostMountPreview(
            host_path="src",
            resolved_host=Path("/home/user/project/src"),
            mount_path="src",
            entries=("main.py", "utils.py"),
            is_directory=True,
            bytes_copied=5000,
        )
        assert preview.host_path == "src"
        assert preview.resolved_host == Path("/home/user/project/src")
        assert preview.mount_path == "src"
        assert preview.entries == ("main.py", "utils.py")
        assert preview.is_directory is True
        assert preview.bytes_copied == 5000


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Path traversal / symlink security
# ---------------------------------------------------------------------------


class TestPathTraversal:
    def test_mount_path_traversal_rejected(self, sample_source: Path) -> None:
        mount = HostMount(host_path=str(sample_source), mount_path="../../escape")
        with pytest.raises(WorkspaceSecurityError, match="escapes workspace"):
            _create_workspace([mount], allowed_host_roots=[sample_source.parent])


class TestSymlinkEscape:
    def test_symlink_escape_skipped(self, temp_dir: Path) -> None:
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
        src = temp_dir / "src.txt"
        src.write_text("content")
        tgt = temp_dir / "workspace" / "src.txt"
        mount = HostMount(host_path=str(src), follow_symlinks=False)

        with unittest.mock.patch("shutil.copy2", wraps=shutil.copy2) as mock_copy:
            _copy_mount_to_temp(src, tgt, mount)
            mock_copy.assert_called_once()
            assert mock_copy.call_args[1].get("follow_symlinks") is False

    def test_symlink_file_skipped_when_not_following(self, temp_dir: Path) -> None:
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


class TestSingleFileSymlinkEscape:
    def test_single_file_symlink_rejected_when_not_following(
        self, temp_dir: Path
    ) -> None:
        target_file = temp_dir / "real.txt"
        target_file.write_text("content")
        link = temp_dir / "link.txt"
        link.symlink_to(target_file)

        tgt = temp_dir / "workspace" / "link.txt"
        mount = HostMount(host_path=str(link), follow_symlinks=False)

        with pytest.raises(WorkspaceSecurityError, match="follow_symlinks=False"):
            _copy_mount_to_temp(link, tgt, mount)

    def test_single_file_symlink_escaping_parent_rejected(self, temp_dir: Path) -> None:
        outside = temp_dir / "outside.txt"
        outside.write_text("secret")

        subdir = temp_dir / "subdir"
        subdir.mkdir()
        link = subdir / "escape.txt"
        link.symlink_to(outside)

        tgt = temp_dir / "workspace" / "escape.txt"
        mount = HostMount(host_path=str(link), follow_symlinks=True)

        with pytest.raises(WorkspaceSecurityError, match="escapes parent"):
            _copy_mount_to_temp(link, tgt, mount)

    def test_single_file_symlink_within_parent_allowed(self, temp_dir: Path) -> None:
        real_file = temp_dir / "real.txt"
        real_file.write_text("allowed")
        link = temp_dir / "link.txt"
        link.symlink_to(real_file)

        tgt = temp_dir / "workspace" / "link.txt"
        mount = HostMount(host_path=str(link), follow_symlinks=True)

        preview = _copy_mount_to_temp(link, tgt, mount)
        assert tgt.exists()
        assert preview.bytes_copied == len("allowed")


class TestMaxBytesZero:
    def test_max_bytes_zero_rejects_files(self, temp_dir: Path) -> None:
        src = temp_dir / "tiny.txt"
        src.write_text("a")
        tgt = temp_dir / "workspace" / "tiny.txt"
        mount = HostMount(host_path=str(src), max_bytes=0)

        with pytest.raises(WorkspaceBudgetExceededError, match="byte budget"):
            _copy_mount_to_temp(src, tgt, mount)

    def test_max_bytes_zero_rejects_directory_files(
        self, sample_source: Path, temp_dir: Path
    ) -> None:
        tgt = temp_dir / "workspace" / "source"
        tgt.mkdir(parents=True)
        mount = HostMount(host_path=str(sample_source), max_bytes=0)

        with pytest.raises(WorkspaceBudgetExceededError, match="byte budget"):
            _copy_mount_to_temp(sample_source, tgt, mount)
