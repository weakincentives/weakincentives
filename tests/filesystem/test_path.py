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

"""Tests for shared path normalization utilities."""

from __future__ import annotations

import pytest

from weakincentives.filesystem._path import (
    MAX_PATH_DEPTH,
    MAX_SEGMENT_LENGTH,
    normalize_path_string,
    strip_mount_point,
    validate_path,
)


class TestConstants:
    """Test path constants are properly defined."""

    def test_max_path_depth_is_16(self) -> None:
        assert MAX_PATH_DEPTH == 16

    def test_max_segment_length_is_80(self) -> None:
        assert MAX_SEGMENT_LENGTH == 80


class TestNormalizePathString:
    """Test normalize_path_string function."""

    def test_empty_string_returns_empty(self) -> None:
        assert normalize_path_string("") == ""

    def test_dot_returns_empty(self) -> None:
        assert normalize_path_string(".") == ""

    def test_slash_returns_empty(self) -> None:
        assert normalize_path_string("/") == ""

    def test_simple_path(self) -> None:
        assert normalize_path_string("foo/bar") == "foo/bar"

    def test_strips_leading_slash(self) -> None:
        assert normalize_path_string("/foo/bar") == "foo/bar"

    def test_strips_trailing_slash(self) -> None:
        assert normalize_path_string("foo/bar/") == "foo/bar"

    def test_strips_leading_and_trailing_slashes(self) -> None:
        assert normalize_path_string("/foo/bar/") == "foo/bar"

    def test_strips_whitespace(self) -> None:
        assert normalize_path_string("  foo/bar  ") == "foo/bar"

    def test_removes_empty_segments(self) -> None:
        assert normalize_path_string("foo//bar") == "foo/bar"

    def test_removes_dot_segments(self) -> None:
        assert normalize_path_string("foo/./bar") == "foo/bar"

    def test_processes_dotdot_segments(self) -> None:
        assert normalize_path_string("foo/bar/../baz") == "foo/baz"

    def test_dotdot_at_root_is_ignored(self) -> None:
        assert normalize_path_string("../foo") == "foo"

    def test_multiple_dotdot(self) -> None:
        assert normalize_path_string("a/b/c/../../d") == "a/d"

    def test_dotdot_beyond_root(self) -> None:
        assert normalize_path_string("foo/../../bar") == "bar"

    def test_all_dotdot_returns_empty(self) -> None:
        assert normalize_path_string("foo/../") == ""


class TestValidatePath:
    """Test validate_path function."""

    def test_empty_path_is_valid(self) -> None:
        validate_path("")  # Should not raise

    def test_simple_path_is_valid(self) -> None:
        validate_path("foo/bar")  # Should not raise

    def test_max_depth_path_is_valid(self) -> None:
        path = "/".join(f"s{i}" for i in range(MAX_PATH_DEPTH))
        validate_path(path)  # Should not raise

    def test_exceeds_max_depth_raises(self) -> None:
        path = "/".join(f"s{i}" for i in range(MAX_PATH_DEPTH + 1))
        with pytest.raises(ValueError, match="Path depth exceeds limit"):
            validate_path(path)

    def test_max_segment_length_is_valid(self) -> None:
        path = "a" * MAX_SEGMENT_LENGTH
        validate_path(path)  # Should not raise

    def test_exceeds_max_segment_length_raises(self) -> None:
        path = "a" * (MAX_SEGMENT_LENGTH + 1)
        with pytest.raises(ValueError, match="Path segment exceeds limit"):
            validate_path(path)


class TestStripMountPoint:
    """Test strip_mount_point function."""

    def test_none_mount_point_returns_unchanged(self) -> None:
        assert strip_mount_point("foo/bar", None) == "foo/bar"

    def test_matching_prefix_with_slash(self) -> None:
        assert strip_mount_point("workspace/foo", "workspace") == "foo"

    def test_exact_match_returns_empty(self) -> None:
        assert strip_mount_point("workspace", "workspace") == ""

    def test_no_match_returns_unchanged(self) -> None:
        assert strip_mount_point("other/path", "workspace") == "other/path"

    def test_partial_match_not_at_boundary_returns_unchanged(self) -> None:
        # "workspacefoo" should not match mount point "workspace"
        assert strip_mount_point("workspacefoo/bar", "workspace") == "workspacefoo/bar"

    def test_mount_point_with_leading_slash_stripped(self) -> None:
        assert strip_mount_point("workspace/foo", "/workspace") == "foo"

    def test_nested_mount_point(self) -> None:
        assert strip_mount_point("a/b/foo", "a/b") == "foo"

    def test_path_shorter_than_mount_point(self) -> None:
        assert strip_mount_point("ws", "workspace") == "ws"
