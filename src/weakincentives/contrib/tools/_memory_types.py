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

"""Internal types for the in-memory filesystem backend."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from weakincentives.filesystem import FileEntry, is_path_under


# ---------------------------------------------------------------------------
# Internal Types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class InMemoryFile:
    """Internal representation of a file in memory.

    Files are stored as raw bytes internally. Text operations encode/decode
    using UTF-8, while byte operations work directly with the stored content.
    """

    content: bytes
    created_at: datetime
    modified_at: datetime


@dataclass(slots=True, frozen=True)
class InMemoryState:
    """Frozen snapshot of in-memory filesystem state."""

    files: Mapping[str, InMemoryFile]
    directories: frozenset[str]


def empty_files_dict() -> dict[str, InMemoryFile]:
    return {}


def empty_directories_set() -> set[str]:
    return set()


def empty_snapshots_dict() -> dict[str, InMemoryState]:
    return {}


# ---------------------------------------------------------------------------
# Helper Functions (formerly methods on InMemoryFilesystem)
# ---------------------------------------------------------------------------


def collect_file_entries(
    files: dict[str, InMemoryFile],
    normalized: str,
    prefix: str,
    seen: set[str],
) -> list[FileEntry]:
    """Collect file entries and implicit directories from files."""
    entries: list[FileEntry] = []
    for file_path in files:
        if not is_path_under(file_path, normalized):
            continue
        relative = file_path[len(prefix) :] if prefix else file_path
        if "/" in relative:
            child_dir = relative.split("/")[0]
            if child_dir not in seen:
                seen.add(child_dir)
                entries.append(
                    FileEntry(
                        name=child_dir,
                        path=f"{prefix}{child_dir}" if prefix else child_dir,
                        is_file=False,
                        is_directory=True,
                    )
                )
        else:
            entries.append(
                FileEntry(
                    name=relative, path=file_path, is_file=True, is_directory=False
                )
            )
    return entries


def collect_explicit_dir_entries(
    directories: set[str],
    normalized: str,
    prefix: str,
    seen: set[str],
) -> list[FileEntry]:
    """Collect explicit directory entries not already seen."""
    entries: list[FileEntry] = []
    for dir_path in directories:
        if not is_path_under(dir_path, normalized) or dir_path == normalized:
            continue
        relative = dir_path[len(prefix) :] if prefix else dir_path
        if "/" not in relative and relative not in seen:
            seen.add(relative)
            entries.append(
                FileEntry(
                    name=relative, path=dir_path, is_file=False, is_directory=True
                )
            )
    return entries
