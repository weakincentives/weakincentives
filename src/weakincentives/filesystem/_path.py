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

"""Shared path normalization utilities.

This module provides the canonical path normalization functions used throughout
the codebase. Both the core filesystem types and workspace tools use these
utilities to ensure consistent path handling.

Constants:
    MAX_PATH_DEPTH: Maximum allowed path depth (16 segments)
    MAX_SEGMENT_LENGTH: Maximum allowed segment length (80 characters)

Functions:
    normalize_path_string: Normalize a string path by removing slashes and ".." segments
    validate_path: Validate path constraints (depth and segment length)
    strip_mount_point: Remove a mount point prefix from a path
"""

from __future__ import annotations

from typing import Final

MAX_PATH_DEPTH: Final[int] = 16
MAX_SEGMENT_LENGTH: Final[int] = 80


def normalize_path_string(path: str) -> str:
    """Normalize a path by removing leading/trailing slashes and cleaning segments.

    This function:
    - Converts "/" or "." to empty string ""
    - Strips leading/trailing whitespace and slashes
    - Removes empty segments and "." entries
    - Processes ".." segments by popping from the result stack

    Args:
        path: The path string to normalize.

    Returns:
        Normalized path with segments joined by "/", or empty string for root.

    Examples:
        >>> normalize_path_string("/foo/bar/")
        'foo/bar'
        >>> normalize_path_string("foo/../bar")
        'bar'
        >>> normalize_path_string("/")
        ''
    """
    if not path or path in {".", "/"}:
        return ""
    stripped = path.strip().strip("/")
    segments = [s for s in stripped.split("/") if s and s != "."]
    # Process .. segments
    result: list[str] = []
    for segment in segments:
        if segment == "..":
            if result:
                _ = result.pop()
        else:
            result.append(segment)
    return "/".join(result)


def validate_path(path: str) -> None:
    """Validate path constraints.

    Args:
        path: The normalized path to validate.

    Raises:
        ValueError: If path depth exceeds MAX_PATH_DEPTH or any segment
            exceeds MAX_SEGMENT_LENGTH.
    """
    if not path:
        return
    segments = path.split("/")
    if len(segments) > MAX_PATH_DEPTH:
        msg = f"Path depth exceeds limit of {MAX_PATH_DEPTH} segments."
        raise ValueError(msg)
    for segment in segments:
        if len(segment) > MAX_SEGMENT_LENGTH:
            msg = f"Path segment exceeds limit of {MAX_SEGMENT_LENGTH} characters."
            raise ValueError(msg)


def strip_mount_point(path: str, mount_point: str | None) -> str:
    """Strip mount point prefix from a path.

    This function normalizes paths that may include a virtual mount point prefix
    (e.g., "/workspace/foo" -> "foo" when mount_point="/workspace").

    Args:
        path: The path to process (may or may not include mount point).
        mount_point: Optional mount point prefix to strip.

    Returns:
        Path with mount point prefix removed, or unchanged path if no match.

    Examples:
        >>> strip_mount_point("workspace/foo", "workspace")
        'foo'
        >>> strip_mount_point("workspace", "workspace")
        ''
        >>> strip_mount_point("other/path", "workspace")
        'other/path'
    """
    if mount_point is None:
        return path
    prefix = mount_point.lstrip("/")
    if path.startswith(prefix + "/"):
        return path[len(prefix) + 1 :]
    if path == prefix:
        return ""
    return path


__all__ = [
    "MAX_PATH_DEPTH",
    "MAX_SEGMENT_LENGTH",
    "normalize_path_string",
    "strip_mount_point",
    "validate_path",
]
