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

"""Host mount materialization for the virtual filesystem.

This module handles loading files from the host filesystem into the VFS:

- normalize_host_root: Validate and resolve allowed mount roots
- materialize_host_mounts_to_filesystem: Load all configured mounts
- render_host_mounts_block: Format mount info for prompt display

The mount system provides a way to seed the VFS with files from the host
filesystem at session start, controlled by allowed roots for security.
"""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Iterable, Sequence
from dataclasses import field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

from ...dataclasses import FrozenDataclass
from ...errors import ToolValidationError
from .filesystem import InMemoryFilesystem
from .vfs_types import (
    HostMount,
    HostMountPreview,
    VfsPath,
    ensure_ascii,
    format_path,
    normalize_optional_path,
    normalize_path,
)

_DEFAULT_ENCODING: Final[Literal["utf-8"]] = "utf-8"
MAX_MOUNT_PREVIEW_ENTRIES: Final[int] = 20

_VFS_SECTION_TEMPLATE: Final[str] = (
    "The virtual filesystem starts empty unless host mounts are configured."
    " It is the only filesystem available and will contain files relevant to the task;"
    " use it as scratch space when necessary.\n"
    "1. Remember the snapshot begins empty aside from configured host mounts.\n"
    "2. Explore with `ls` or `glob` before reading or modifying files.\n"
    "3. Fetch file content with `read_file`; pagination keeps responses focused.\n"
    "4. Create files via `write_file` (create-only) and edit them with `edit_file`.\n"
    "5. Remove files or directories recursively with `rm` when they are no longer needed.\n"
    "6. Host mounts are fixed at session start; additional directories cannot be mounted later.\n"
    "7. Avoid mirroring large repositories or binary assets—only UTF-8 text up to 48k characters is accepted.\n"
    "8. Use `grep` to search for patterns across files when the workspace grows."
)


def normalize_host_root(path: os.PathLike[str] | str) -> Path:
    """Validate and resolve an allowed host root path."""
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise ToolValidationError("Allowed host root does not exist.")
    return root


def materialize_host_mounts_to_filesystem(
    fs: InMemoryFilesystem,
    mounts: Sequence[HostMount],
    allowed_roots: Sequence[Path],
) -> tuple[HostMountPreview, ...]:
    """Materialize host mounts directly into the InMemoryFilesystem."""
    if not mounts:
        return ()

    previews: list[HostMountPreview] = []
    for mount in mounts:
        preview = _load_mount_to_filesystem(fs, mount, allowed_roots)
        previews.append(preview)
    return tuple(previews)


def render_host_mounts_block(previews: Sequence[HostMountPreview]) -> str:
    """Render a block of text describing the configured host mounts."""
    if not previews:
        return ""

    lines: list[str] = ["Configured host mounts:"]
    for preview in previews:
        mount_label = format_path(preview.mount_path)
        resolved_label = str(preview.resolved_host)
        lines.append(
            f"- Host `{resolved_label}` (configured as `{preview.host_path}`) mounted at `{mount_label}`."
        )
        contents = _format_mount_entries(preview.entries)
        lines.append(f"  Contents: {contents}")
    return "\n".join(lines)


def render_section_template(previews: Sequence[HostMountPreview]) -> str:
    """Render the VFS section template with optional mount information."""
    block = render_host_mounts_block(previews)
    if not block:
        return _VFS_SECTION_TEMPLATE
    return f"{_VFS_SECTION_TEMPLATE}\n\n{block}"


def _format_mount_entries(entries: Sequence[str]) -> str:
    """Format a list of mount entries for display."""
    if not entries:
        return "<empty>"
    preview = entries[:MAX_MOUNT_PREVIEW_ENTRIES]
    formatted = " ".join(f"`{entry}`" for entry in preview)
    remaining = len(entries) - len(preview)
    if remaining > 0:
        formatted += f" … (+{remaining} more)"
    return formatted


def _load_mount_to_filesystem(
    fs: InMemoryFilesystem,
    mount: HostMount,
    allowed_roots: Sequence[Path],
) -> HostMountPreview:
    """Load a single host mount into the filesystem."""
    host_path = mount.host_path.strip()
    if not host_path:
        raise ToolValidationError("Host mount path must not be empty.")
    ensure_ascii(host_path, "host path")
    resolved_host = resolve_mount_path(host_path, allowed_roots)
    include_patterns = _normalize_globs(mount.include_glob, "include_glob")
    exclude_patterns = _normalize_globs(mount.exclude_glob, "exclude_glob")
    mount_prefix = normalize_optional_path(mount.mount_path)
    preview = HostMountPreview(
        host_path=host_path,
        resolved_host=resolved_host,
        mount_path=mount_prefix,
        entries=_list_mount_entries(resolved_host),
        is_directory=resolved_host.is_dir(),
    )

    context = MountContext(
        resolved_host=resolved_host,
        mount_prefix=mount_prefix,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        timestamp=get_current_time(),
        max_bytes=mount.max_bytes,
    )

    consumed_bytes = 0
    for path in _iter_mount_files(resolved_host, mount.follow_symlinks):
        consumed_bytes = _read_mount_entry_to_filesystem(
            fs=fs, path=path, context=context, consumed_bytes=consumed_bytes
        )

    return preview


@FrozenDataclass()
class MountContext:
    """Context for mount operations."""

    resolved_host: Path = field(
        metadata={"description": "Absolute host path resolved against allowed roots."}
    )
    mount_prefix: VfsPath = field(
        metadata={
            "description": "Normalized VFS path where the host mount will appear."
        }
    )
    include_patterns: tuple[str, ...] = field(
        metadata={
            "description": "Glob patterns that must match mounted files when provided."
        }
    )
    exclude_patterns: tuple[str, ...] = field(
        metadata={
            "description": "Glob patterns that exclude files from the mounted set."
        }
    )
    timestamp: datetime = field(
        metadata={
            "description": "Timestamp applied to mounted files for created/updated metadata."
        }
    )
    max_bytes: int | None = field(
        metadata={
            "description": "Optional byte budget limiting total mounted file size."
        }
    )


def _read_mount_entry_to_filesystem(
    *,
    fs: InMemoryFilesystem,
    path: Path,
    context: MountContext,
    consumed_bytes: int,
) -> int:
    """Read a mount entry and write it to the filesystem. Returns updated consumed_bytes."""
    relative = (
        Path(path.name)
        if context.resolved_host.is_file()
        else path.relative_to(context.resolved_host)
    )
    relative_posix = relative.as_posix()
    if context.include_patterns and not any(
        match_glob(relative_posix, pattern) for pattern in context.include_patterns
    ):
        return consumed_bytes
    if any(match_glob(relative_posix, pattern) for pattern in context.exclude_patterns):
        return consumed_bytes

    try:
        content = path.read_text(encoding=_DEFAULT_ENCODING)
    except UnicodeDecodeError as error:  # pragma: no cover - defensive guard
        raise ToolValidationError("Mounted file must be valid UTF-8.") from error
    except OSError as error:
        raise ToolValidationError(f"Failed to read mounted file {path}.") from error
    size = len(content.encode(_DEFAULT_ENCODING))
    if context.max_bytes is not None and consumed_bytes + size > context.max_bytes:
        raise ToolValidationError("Host mount exceeded the configured byte budget.")
    consumed_bytes += size

    # Build the VFS path
    segments = context.mount_prefix.segments + relative.parts
    normalized_path = normalize_path(VfsPath(segments))
    vfs_path_str = "/".join(normalized_path.segments)

    # Write to filesystem
    _ = fs.write(vfs_path_str, content, mode="overwrite")

    return consumed_bytes


def _list_mount_entries(root: Path) -> tuple[str, ...]:
    """List entries in a host directory for preview."""
    if root.is_file():
        return (root.name,)
    try:
        children = sorted(root.iterdir(), key=lambda path: path.name.lower())
    except OSError as error:  # pragma: no cover - defensive guard
        raise ToolValidationError(f"Failed to inspect host mount {root}.") from error
    labels: list[str] = []
    for child in children:
        suffix = "/" if child.is_dir() else ""
        labels.append(f"{child.name}{suffix}")
    return tuple(labels)


def resolve_mount_path(host_path: str, allowed_roots: Sequence[Path]) -> Path:
    """Resolve a host path against allowed roots."""
    if not allowed_roots:
        raise ToolValidationError("No allowed host roots configured for mounts.")
    for root in allowed_roots:
        candidate = (root / host_path).resolve()
        try:
            _ = candidate.relative_to(root)
        except ValueError:
            continue
        if candidate.exists():
            return candidate
    raise ToolValidationError("Host path is outside the allowed roots or missing.")


def _normalize_globs(patterns: Sequence[str], field: str) -> tuple[str, ...]:
    """Normalize a sequence of glob patterns."""
    normalized: list[str] = []
    for pattern in patterns:
        stripped = pattern.strip()
        if not stripped:
            continue
        ensure_ascii(stripped, field)
        normalized.append(stripped)
    return tuple(normalized)


def match_glob(path: str, pattern: str) -> bool:
    """Match a path against a glob pattern with proper ** support.

    The standard fnmatch module doesn't treat ** as "zero or more directories".
    This function handles that case: **/*.py should match both "foo.py" (zero
    directories) and "bar/baz.py" (one+ directories).
    """
    if "**/" not in pattern:
        return fnmatch.fnmatchcase(path, pattern)

    # Match the full pattern first (handles one+ directories case).
    if fnmatch.fnmatchcase(path, pattern):
        return True

    # Allow **/ to match zero directories by removing one or more occurrences.
    # This preserves any prefix before **/ (e.g., "src/**/test_*.py" -> "src/test_*.py").
    candidates = [pattern]
    seen: set[str] = {pattern}
    while candidates:
        current = candidates.pop()
        start = 0
        while True:
            index = current.find("**/", start)
            if index == -1:
                break
            variant = f"{current[:index]}{current[index + 3 :]}"
            if variant not in seen:
                if fnmatch.fnmatchcase(path, variant):
                    return True
                seen.add(variant)
                candidates.append(variant)
            start = index + 3

    return False


def _iter_mount_files(root: Path, follow_symlinks: bool) -> Iterable[Path]:
    """Iterate over files in a mount root."""
    if root.is_file():
        yield root
        return
    for current, _dirnames, filenames in root.walk(
        follow_symlinks=follow_symlinks,
    ):
        for name in filenames:
            yield current / name


def get_current_time() -> datetime:
    """Get current UTC time truncated to milliseconds."""
    return _truncate_to_milliseconds(datetime.now(UTC))


def _truncate_to_milliseconds(value: datetime) -> datetime:
    """Truncate a datetime to millisecond precision."""
    microsecond = value.microsecond - (value.microsecond % 1000)
    return value.replace(microsecond=microsecond, tzinfo=UTC)


__all__ = [
    "MAX_MOUNT_PREVIEW_ENTRIES",
    "MountContext",
    "get_current_time",
    "match_glob",
    "materialize_host_mounts_to_filesystem",
    "normalize_host_root",
    "render_host_mounts_block",
    "render_section_template",
    "resolve_mount_path",
]
