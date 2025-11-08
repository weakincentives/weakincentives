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

"""Helpers for inspecting prompt override files on disk."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from .local_store import LocalPromptOverridesStore
from .versioning import PromptOverridesError


@dataclass(frozen=True, slots=True)
class OverrideFileMetadata:
    """Summary information about a prompt override file."""

    path: Path
    relative_segments: tuple[str, ...]
    modified_time: float
    content_hash: str
    section_count: int
    tool_count: int


class _InspectableLocalStore(LocalPromptOverridesStore):
    """Expose internal path resolution for read-only inspection helpers."""

    def overrides_dir(self) -> Path:
        return self._filesystem.overrides_dir()


def resolve_overrides_root(
    *,
    root_path: str | Path | None = None,
    overrides_relative_path: str | Path | None = None,
) -> Path:
    """Return the directory that stores local prompt overrides."""

    if overrides_relative_path is None:
        store = _InspectableLocalStore(root_path=root_path)
    else:
        store = _InspectableLocalStore(
            root_path=root_path,
            overrides_relative_path=overrides_relative_path,
        )
    return store.overrides_dir()


def iter_override_files(
    *,
    overrides_root: Path | None = None,
    root_path: str | Path | None = None,
    overrides_relative_path: str | Path | None = None,
) -> Iterator[OverrideFileMetadata]:
    """Yield metadata for every JSON override file on disk."""

    root = (
        overrides_root
        if overrides_root is not None
        else resolve_overrides_root(
            root_path=root_path, overrides_relative_path=overrides_relative_path
        )
    ).resolve()

    if not root.exists():
        return

    for file_path in sorted(root.rglob("*.json")):
        if not file_path.is_file():
            continue
        yield _build_metadata(file_path, root)


def _build_metadata(file_path: Path, overrides_root: Path) -> OverrideFileMetadata:
    try:
        raw = file_path.read_bytes()
    except OSError as error:  # pragma: no cover - exercised in practice.
        raise PromptOverridesError(
            f"Failed to read override file: {file_path}"
        ) from error

    try:
        stat_result = file_path.stat()
    except OSError as error:  # pragma: no cover - exercised in practice.
        raise PromptOverridesError(
            f"Failed to stat override file: {file_path}"
        ) from error

    try:
        payload_obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PromptOverridesError(
            f"Failed to parse prompt override JSON: {file_path}"
        ) from error

    if not isinstance(payload_obj, dict):
        raise PromptOverridesError(
            f"Prompt override payload must be an object: {file_path}"
        )
    payload = cast(dict[str, Any], payload_obj)

    sections_obj = payload.get("sections", {})
    if not isinstance(sections_obj, dict):
        raise PromptOverridesError(
            f"Override file sections must be a mapping: {file_path}"
        )
    sections = cast(dict[str, Any], sections_obj)

    tools_obj = payload.get("tools", {})
    if not isinstance(tools_obj, dict):
        raise PromptOverridesError(
            f"Override file tools must be a mapping: {file_path}"
        )
    tools = cast(dict[str, Any], tools_obj)

    relative_segments = file_path.resolve().relative_to(overrides_root).parts
    content_hash = sha256(raw).hexdigest()

    return OverrideFileMetadata(
        path=file_path.resolve(),
        relative_segments=relative_segments,
        modified_time=stat_result.st_mtime,
        content_hash=content_hash,
        section_count=len(sections),
        tool_count=len(tools),
    )
