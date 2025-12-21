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

"""FastAPI app for exploring session snapshot JSONL files."""

from __future__ import annotations

import json
import re
import threading
import webbrowser
import zipfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Literal, cast
from urllib.parse import unquote

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from markdown_it import MarkdownIt

from ..dataclasses import FrozenDataclass
from ..errors import WinkError
from ..runtime.logging import StructuredLogger, get_logger
from ..runtime.session.snapshots import (
    Snapshot,
    SnapshotPayload,
    SnapshotRestoreError,
    SnapshotSlicePayload,
)
from ..types import JSONValue

# Module-level logger keeps loader warnings consistent with the debug server.
logger: StructuredLogger = get_logger(__name__)

_MARKDOWN_WRAPPER_KEY = "__markdown__"
_MARKDOWN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(^|\n)#{1,6}\s"),
    re.compile(r"(^|\n)[-*+]\s"),
    re.compile(r"(^|\n)\d+\.\s"),
    re.compile(r"`{1,3}[^`]+`{1,3}"),
    re.compile(r"\[.+?\]\(.+?\)"),
    re.compile(r"\n\n"),
    re.compile(r"\*\*[^\s].+?\*\*"),
)
_MIN_MARKDOWN_LENGTH = 16
_markdown = MarkdownIt("commonmark", {"linkify": True})

# pyright: reportUnusedFunction=false


class SnapshotLoadError(WinkError, RuntimeError):
    """Raised when a snapshot cannot be loaded or validated."""


def _looks_like_markdown(text: str) -> bool:
    candidate = text.strip()
    if len(candidate) < _MIN_MARKDOWN_LENGTH:
        return False
    return any(pattern.search(candidate) for pattern in _MARKDOWN_PATTERNS)


def _render_markdown(text: str) -> Mapping[str, str]:
    return {
        "text": text,
        "html": _markdown.render(text),
    }


def _render_markdown_values(value: JSONValue) -> JSONValue:
    if isinstance(value, str):
        if _looks_like_markdown(value):
            return {_MARKDOWN_WRAPPER_KEY: _render_markdown(value)}
        return value

    if isinstance(value, Mapping):
        if _MARKDOWN_WRAPPER_KEY in value:
            return value
        mapping_value = cast(Mapping[str, JSONValue], value)
        normalized: dict[str, JSONValue] = {}
        for key, item in mapping_value.items():
            normalized[str(key)] = _render_markdown_values(item)
        return normalized

    if isinstance(value, list):
        return [_render_markdown_values(item) for item in value]

    return value


def _class_name(type_identifier: str) -> str:
    """Extract the class name from a fully qualified type identifier."""
    return type_identifier.rsplit(".", 1)[-1]


@FrozenDataclass()
class SliceSummary:
    slice_type: str
    item_type: str
    count: int


@FrozenDataclass()
class SnapshotMeta:
    version: str
    created_at: str
    path: str
    session_id: str
    line_number: int
    slices: tuple[SliceSummary, ...]
    tags: Mapping[str, str]
    validation_error: str | None = None


@FrozenDataclass()
class LoadedSnapshot:
    meta: SnapshotMeta
    slices: Mapping[str, SnapshotSlicePayload]
    raw_payload: Mapping[str, JSONValue]
    raw_text: str
    path: Path


SnapshotLoader = Callable[[Path], tuple[LoadedSnapshot, ...]]

# Metadata filename stored inside the ZIP archive
_WINK_METADATA_FILENAME = "_wink_metadata.json"


@FrozenDataclass()
class FilesystemArchiveMetadata:
    """Metadata from a filesystem archive."""

    version: str
    created_at: str
    session_id: str | None
    root_path: str
    file_count: int
    total_bytes: int


@FrozenDataclass()
class FileTreeNode:
    """Node in the filesystem tree."""

    name: str
    type: Literal["file", "directory"]
    path: str | None = None  # Full path for files
    size: int | None = None  # Byte size for files
    children: tuple[FileTreeNode, ...] | None = None  # For directories


@FrozenDataclass()
class FileContent:
    """Content of a file from the archive."""

    path: str
    content: str | None  # None for binary files
    size_bytes: int
    total_lines: int | None = None
    offset: int = 0
    limit: int | None = None
    truncated: bool = False
    encoding: str = "utf-8"
    binary: bool = False
    error: str | None = None


def load_snapshot(snapshot_path: Path) -> tuple[LoadedSnapshot, ...]:
    """Load and validate one or more snapshots from disk."""

    if not snapshot_path.exists():
        msg = f"Snapshot file not found: {snapshot_path}"
        raise SnapshotLoadError(msg)

    try:
        raw_text = snapshot_path.read_text()
    except OSError as error:  # pragma: no cover - filesystem failures are unlikely
        msg = f"Snapshot file cannot be read: {snapshot_path}"
        raise SnapshotLoadError(msg) from error

    entries: list[LoadedSnapshot] = []
    for line_number, line in _extract_snapshot_lines(raw_text):
        entries.append(_load_snapshot_line(line, line_number, snapshot_path))

    if not entries:
        msg = f"Snapshot file contained no entries: {snapshot_path}"
        raise SnapshotLoadError(msg)

    return tuple(entries)


def _extract_snapshot_lines(raw_text: str) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    for index, line in enumerate(raw_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        lines.append((index, stripped))
    return lines


def _slice_lookup(
    slices: tuple[SnapshotSlicePayload, ...],
) -> Mapping[str, SnapshotSlicePayload]:
    return MappingProxyType({entry.slice_type: entry for entry in slices})


def _summaries_from_slices(
    slices: tuple[SnapshotSlicePayload, ...],
) -> tuple[SliceSummary, ...]:
    return tuple(
        SliceSummary(
            slice_type=entry.slice_type,
            item_type=entry.item_type,
            count=len(entry.items),
        )
        for entry in slices
    )


def _load_snapshot_line(
    line: str,
    line_number: int,
    snapshot_path: Path,
) -> LoadedSnapshot:
    try:
        payload = SnapshotPayload.from_json(line)
    except SnapshotRestoreError as error:
        msg = f"Invalid snapshot at line {line_number}: {error}"
        raise SnapshotLoadError(msg) from error

    validation_error: str | None = None
    try:
        _ = Snapshot.from_json(line)
    except SnapshotRestoreError as error:
        validation_error = str(error)
        logger.warning(
            "Snapshot validation failed",
            event="wink.debug.snapshot_error",
            context={
                "path": str(snapshot_path),
                "line_number": line_number,
                "error": validation_error,
            },
        )

    raw_payload = MappingProxyType(
        json.loads(line, object_pairs_hook=dict),
    )

    slices = _slice_lookup(payload.slices)
    summaries = _summaries_from_slices(payload.slices)

    session_id = payload.tags.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        msg = f"Snapshot is missing a session_id tag at line {line_number}"
        raise SnapshotLoadError(msg)

    meta = SnapshotMeta(
        version=payload.version,
        created_at=payload.created_at,
        path=str(snapshot_path),
        session_id=session_id,
        line_number=line_number,
        slices=tuple(summaries),
        tags=payload.tags,
        validation_error=validation_error,
    )

    return LoadedSnapshot(
        meta=meta,
        slices=slices,
        raw_payload=raw_payload,
        raw_text=line,
        path=snapshot_path,
    )


class SnapshotStore:
    """In-memory store for the active snapshot and reload handling."""

    def __init__(
        self,
        path: Path,
        *,
        loader: SnapshotLoader,
        logger: StructuredLogger | None = None,
    ) -> None:
        super().__init__()
        resolved = path.resolve()
        self._root, self._path = self._normalize_path(resolved)
        self._loader = loader
        self._logger = logger or get_logger(__name__)
        self._entries: tuple[LoadedSnapshot, ...] = self._load_entries(self._path)
        self._index = 0

    @property
    def meta(self) -> SnapshotMeta:
        return self._current.meta

    @property
    def raw_payload(self) -> Mapping[str, JSONValue]:
        return self._current.raw_payload

    @property
    def raw_text(self) -> str:
        return self._current.raw_text

    @property
    def path(self) -> Path:
        return self._path

    @property
    def entries(self) -> tuple[LoadedSnapshot, ...]:
        return self._entries

    def list_snapshots(self) -> list[Mapping[str, JSONValue]]:
        snapshots: list[tuple[float, Path]] = []
        for candidate in sorted(self._iter_snapshot_files(self._root)):
            try:
                stats = candidate.stat()
                created_at = max(stats.st_ctime, stats.st_mtime)
            except OSError:
                continue
            snapshots.append((created_at, candidate))

        entries: list[Mapping[str, JSONValue]] = []
        for created_at, candidate in sorted(
            snapshots, key=lambda entry: entry[0], reverse=True
        ):
            created_iso = datetime.fromtimestamp(created_at, tz=UTC).isoformat()
            entries.append(
                {
                    "path": str(candidate),
                    "name": candidate.name,
                    "created_at": created_iso,
                }
            )
        return entries

    def list_entries(self) -> list[Mapping[str, JSONValue]]:
        entries: list[Mapping[str, JSONValue]] = []
        for entry in self._entries:
            meta = entry.meta
            entries.append(
                {
                    "session_id": meta.session_id,
                    "name": f"{meta.session_id} (line {meta.line_number})",
                    "path": meta.path,
                    "line_number": meta.line_number,
                    "created_at": meta.created_at,
                    "tags": dict(meta.tags),
                    "selected": meta.session_id == self.meta.session_id,
                }
            )
        return entries

    def slice_items(self, slice_type: str) -> SnapshotSlicePayload:
        try:
            return self._current.slices[slice_type]
        except KeyError as error:
            raise KeyError(f"Unknown slice type: {slice_type}") from error

    def reload(self) -> SnapshotMeta:
        current_session_id = self.meta.session_id
        self._entries = self._load_entries(self._path)
        try:
            self._index = self._select_index(session_id=current_session_id)
        except SnapshotLoadError:
            self._index = 0
        self._logger.info(
            "Snapshot reloaded",
            event="debug.server.reload",
            context={"path": str(self._path)},
        )
        return self.meta

    def select(
        self, *, session_id: str | None = None, line_number: int | None = None
    ) -> SnapshotMeta:
        self._index = self._select_index(
            session_id=session_id,
            line_number=line_number,
        )
        return self.meta

    def switch(
        self,
        path: Path,
        *,
        session_id: str | None = None,
        line_number: int | None = None,
    ) -> SnapshotMeta:
        resolved = path.resolve()
        root, target = self._normalize_path(resolved)
        if root != self._root:
            msg = f"Snapshot must live under {self._root}"
            raise SnapshotLoadError(msg)

        self._root = root
        self._path = target
        self._entries = self._load_entries(target)
        self._index = self._select_index(
            session_id=session_id,
            line_number=line_number,
        )
        self._logger.info(
            "Snapshot switched",
            event="debug.server.switch",
            context={"path": str(self._path)},
        )
        return self.meta

    def _normalize_path(self, path: Path) -> tuple[Path, Path]:
        if path.is_dir():
            root = path
            candidates = sorted(
                self._iter_snapshot_files(root),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not candidates:
                msg = f"No snapshots found under {root}"
                raise SnapshotLoadError(msg)
            target = candidates[0]
        else:
            root = path.parent
            target = path
        return root, target

    def _select_index(
        self, *, session_id: str | None = None, line_number: int | None = None
    ) -> int:
        if session_id is not None:
            for index, entry in enumerate(self._entries):
                if entry.meta.session_id == session_id:
                    return index
            msg = f"Unknown session_id: {session_id}"
            raise SnapshotLoadError(msg)

        if line_number is not None:
            for index, entry in enumerate(self._entries):
                if entry.meta.line_number == line_number:
                    return index
            msg = f"Unknown line_number: {line_number}"
            raise SnapshotLoadError(msg)

        return 0

    def _load_entries(self, path: Path) -> tuple[LoadedSnapshot, ...]:
        entries = self._loader(path)
        if not entries:
            msg = f"No snapshots found under {path}"
            raise SnapshotLoadError(msg)
        return entries

    @staticmethod
    def _iter_snapshot_files(root: Path) -> list[Path]:
        candidates: list[Path] = []
        for pattern in ("*.jsonl", "*.json"):
            candidates.extend(p for p in root.glob(pattern) if p.is_file())
        return candidates

    @property
    def _current(self) -> LoadedSnapshot:
        return self._entries[self._index]


class FilesystemArchiveStore:
    """In-memory store for filesystem archive contents."""

    def __init__(self, *, logger: StructuredLogger | None = None) -> None:
        super().__init__()
        self._logger = logger or get_logger(__name__)
        self._archive_path: Path | None = None
        self._metadata: FilesystemArchiveMetadata | None = None
        self._file_index: dict[str, int] = {}  # path -> size in bytes
        self._tree: FileTreeNode | None = None

    @property
    def has_archive(self) -> bool:
        """Return True if an archive is currently loaded."""
        return self._archive_path is not None

    @property
    def archive_path(self) -> Path | None:
        """Return the path to the loaded archive."""
        return self._archive_path

    @property
    def metadata(self) -> FilesystemArchiveMetadata | None:
        """Return metadata for the loaded archive."""
        return self._metadata

    @property
    def tree(self) -> FileTreeNode | None:
        """Return the file tree for the loaded archive."""
        return self._tree

    def load_for_snapshot(self, snapshot_path: Path) -> bool:
        """Load the companion archive for a snapshot file.

        Args:
            snapshot_path: Path to the .jsonl snapshot file.

        Returns:
            True if an archive was found and loaded, False otherwise.
        """
        archive_path = self._find_companion_archive(snapshot_path)
        if archive_path is None:
            self._clear()
            self._logger.debug(
                "No companion archive found",
                event="debug.filesystem.archive_not_found",
                context={"expected_path": str(snapshot_path.with_suffix(".fs.zip"))},
            )
            return False

        return self._load_archive(archive_path)

    def read_file(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> FileContent:
        """Read a file from the archive.

        Args:
            path: Path to the file within the archive.
            offset: Line offset (0-indexed).
            limit: Maximum number of lines to return.

        Returns:
            FileContent with the file contents or error information.
        """
        if self._archive_path is None:
            return FileContent(
                path=path,
                content=None,
                size_bytes=0,
                error="No archive loaded",
            )

        if path not in self._file_index:
            return FileContent(
                path=path,
                content=None,
                size_bytes=0,
                error=f"File not found: {path}",
            )

        try:
            with zipfile.ZipFile(self._archive_path, "r") as zf:
                raw_bytes = zf.read(path)
                size_bytes = len(raw_bytes)

                # Try to decode as UTF-8
                try:
                    content = raw_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    self._logger.debug(
                        "Binary file detected",
                        event="debug.filesystem.file_read",
                        context={"path": path, "size_bytes": size_bytes},
                    )
                    return FileContent(
                        path=path,
                        content=None,
                        size_bytes=size_bytes,
                        binary=True,
                        error="Binary file cannot be displayed as text",
                    )

                lines = content.splitlines(keepends=True)
                total_lines = len(lines)

                # Apply pagination
                if offset > 0:
                    lines = lines[offset:]
                truncated = False
                if limit is not None and len(lines) > limit:
                    lines = lines[:limit]
                    truncated = True

                paginated_content = "".join(lines)

                self._logger.debug(
                    "File read from archive",
                    event="debug.filesystem.file_read",
                    context={"path": path, "size_bytes": size_bytes},
                )

                return FileContent(
                    path=path,
                    content=paginated_content,
                    size_bytes=size_bytes,
                    total_lines=total_lines,
                    offset=offset,
                    limit=limit,
                    truncated=truncated,
                )

        except (zipfile.BadZipFile, OSError) as error:
            self._logger.warning(
                "Failed to read file from archive",
                event="debug.filesystem.file_read_error",
                context={"path": path, "error": str(error)},
            )
            return FileContent(
                path=path,
                content=None,
                size_bytes=0,
                error=f"Failed to read file: {error}",
            )

    def read_file_raw(self, path: str) -> bytes | None:
        """Read raw bytes of a file from the archive.

        Args:
            path: Path to the file within the archive.

        Returns:
            Raw bytes of the file, or None if not found or error.
        """
        if self._archive_path is None or path not in self._file_index:
            return None

        try:
            with zipfile.ZipFile(self._archive_path, "r") as zf:
                return zf.read(path)
        except (zipfile.BadZipFile, OSError):
            return None

    def _load_archive(self, archive_path: Path) -> bool:
        """Load an archive file.

        Args:
            archive_path: Path to the .fs.zip archive.

        Returns:
            True if loaded successfully, False otherwise.
        """
        try:
            file_index, metadata, tree = self._parse_archive(archive_path)
        except (zipfile.BadZipFile, OSError) as error:
            self._logger.warning(
                "Failed to load filesystem archive",
                event="debug.filesystem.archive_error",
                context={"path": str(archive_path), "error": str(error)},
            )
            self._clear()
            return False

        self._archive_path = archive_path
        self._file_index = file_index
        self._metadata = metadata
        self._tree = tree

        self._logger.info(
            "Filesystem archive loaded",
            event="debug.filesystem.archive_loaded",
            context={"path": str(archive_path), "file_count": len(file_index)},
        )
        return True

    def _parse_archive(
        self, archive_path: Path
    ) -> tuple[dict[str, int], FilesystemArchiveMetadata | None, FileTreeNode]:
        """Parse an archive file and extract its contents.

        Returns:
            Tuple of (file_index, metadata, tree).
        """
        with zipfile.ZipFile(archive_path, "r") as zf:
            # Build file index
            file_index: dict[str, int] = {}
            for info in zf.infolist():
                if not info.is_dir() and info.filename != _WINK_METADATA_FILENAME:
                    file_index[info.filename] = info.file_size

            # Load metadata if present
            metadata = self._load_archive_metadata(zf, file_index, archive_path)

            # Build file tree
            tree = _build_file_tree(file_index)

        return file_index, metadata, tree

    def _load_archive_metadata(
        self,
        zf: zipfile.ZipFile,
        file_index: dict[str, int],
        archive_path: Path,
    ) -> FilesystemArchiveMetadata | None:
        """Load metadata from an archive if present."""
        if _WINK_METADATA_FILENAME not in zf.namelist():
            return None

        try:
            raw = zf.read(_WINK_METADATA_FILENAME).decode("utf-8")
            meta_dict = json.loads(raw)
            return FilesystemArchiveMetadata(
                version=meta_dict.get("version", "1"),
                created_at=meta_dict.get("created_at", ""),
                session_id=meta_dict.get("session_id"),
                root_path=meta_dict.get("root_path", "/"),
                file_count=meta_dict.get("file_count", len(file_index)),
                total_bytes=meta_dict.get("total_bytes", 0),
            )
        except (json.JSONDecodeError, KeyError) as error:
            self._logger.warning(
                "Failed to parse archive metadata",
                event="debug.filesystem.archive_error",
                context={"path": str(archive_path), "error": str(error)},
            )
            # Continue without metadata
            return FilesystemArchiveMetadata(
                version="1",
                created_at="",
                session_id=None,
                root_path="/",
                file_count=len(file_index),
                total_bytes=sum(file_index.values()),
            )

    def _clear(self) -> None:
        """Clear the current archive state."""
        self._archive_path = None
        self._metadata = None
        self._file_index = {}
        self._tree = None

    @staticmethod
    def _find_companion_archive(snapshot_path: Path) -> Path | None:
        """Find the companion .fs.zip archive for a snapshot file."""
        # Try .fs.zip suffix first
        archive_path = snapshot_path.with_suffix(".fs.zip")
        if archive_path.exists():
            return archive_path

        # Also try replacing .jsonl with .fs.zip (handles non-standard paths)
        if snapshot_path.suffix == ".jsonl":  # pragma: no branch
            stem = snapshot_path.stem
            archive_path = snapshot_path.parent / f"{stem}.fs.zip"
            if archive_path.exists():  # pragma: no cover
                return archive_path

        return None


def _build_file_tree(file_index: dict[str, int]) -> FileTreeNode:
    """Build a hierarchical file tree from the file index."""
    tree_dict = _build_tree_dict(file_index)
    return _convert_tree_dict("/", tree_dict)


# Type alias for tree dictionary nodes
_TreeDict = dict[str, "_TreeDict | dict[str, object]"]


def _build_tree_dict(file_index: dict[str, int]) -> dict[str, object]:
    """Build a nested dictionary structure from file paths."""
    tree_dict: dict[str, object] = {}

    for path, size in sorted(file_index.items()):
        parts = path.split("/")
        current = tree_dict
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                # Leaf node (file)
                current[part] = {"__file__": True, "__size__": size, "__path__": path}
            elif part not in current:
                current[part] = {}
                current = cast("dict[str, object]", current[part])
            else:
                next_level = current[part]
                if (
                    isinstance(next_level, dict) and "__file__" not in next_level
                ):  # pragma: no branch
                    current = cast("dict[str, object]", next_level)

    return tree_dict


def _convert_tree_dict(name: str, node: Mapping[str, object]) -> FileTreeNode:
    """Convert a tree dictionary node to a FileTreeNode."""
    if "__file__" in node:
        return FileTreeNode(
            name=name,
            type="file",
            path=cast(str, node.get("__path__")),
            size=cast(int, node.get("__size__")),
        )

    children = tuple(
        _convert_tree_dict(str(child_name), cast("Mapping[str, object]", child_node))
        for child_name, child_node in sorted(node.items())
    )
    return FileTreeNode(name=name, type="directory", children=children)


def _meta_response(meta: SnapshotMeta) -> Mapping[str, JSONValue]:
    return {
        "version": meta.version,
        "created_at": meta.created_at,
        "path": meta.path,
        "session_id": meta.session_id,
        "line_number": meta.line_number,
        "tags": dict(meta.tags),
        "validation_error": meta.validation_error,
        "slices": [
            {
                "slice_type": entry.slice_type,
                "item_type": entry.item_type,
                "display_name": _class_name(entry.slice_type),
                "item_display_name": _class_name(entry.item_type),
                "count": entry.count,
            }
            for entry in meta.slices
        ],
    }


class _DebugAppHandlers:
    def __init__(
        self,
        *,
        store: SnapshotStore,
        filesystem_store: FilesystemArchiveStore,
        logger: StructuredLogger,
        static_dir: Path,
    ) -> None:
        super().__init__()
        self._store = store
        self._filesystem_store = filesystem_store
        self._logger = logger
        self._static_dir = static_dir
        # Load filesystem archive for current snapshot
        _ = self._filesystem_store.load_for_snapshot(store.path)

    def index(self) -> str:
        index_path = self._static_dir / "index.html"
        return index_path.read_text()

    def get_meta(self) -> Mapping[str, JSONValue]:
        return _meta_response(self._store.meta)

    def list_entries(self) -> list[Mapping[str, JSONValue]]:
        return self._store.list_entries()

    def get_slice(
        self,
        encoded_slice_type: str,
        *,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int | None, Query(ge=0)] = None,
    ) -> Mapping[str, JSONValue]:
        slice_items = self._slice_items(encoded_slice_type)
        items = self._paginate_items(
            list(slice_items.items), offset=offset, limit=limit
        )
        rendered_items = [_render_markdown_values(item) for item in items]
        return {
            "slice_type": slice_items.slice_type,
            "item_type": slice_items.item_type,
            "display_name": _class_name(slice_items.slice_type),
            "item_display_name": _class_name(slice_items.item_type),
            "items": rendered_items,
        }

    def get_raw(self) -> JSONResponse:
        return JSONResponse(json.loads(self._store.raw_text))

    def reload(self) -> Mapping[str, JSONValue]:
        return _meta_response(self._execute_reload())

    def list_snapshots(self) -> list[Mapping[str, JSONValue]]:
        return self._store.list_snapshots()

    def select(self, payload: dict[str, JSONValue]) -> Mapping[str, JSONValue]:
        session_id, line_number = self._parse_select_payload(payload)
        meta = self._execute_snapshot_command(
            lambda: self._store.select(session_id=session_id, line_number=line_number)
        )
        return _meta_response(meta)

    def switch(self, payload: dict[str, JSONValue]) -> Mapping[str, JSONValue]:
        path, session_id, line_number = self._parse_switch_payload(payload)
        meta = self._execute_snapshot_command(
            lambda: self._store.switch(
                path,
                session_id=session_id,
                line_number=line_number,
            )
        )
        # Reload filesystem archive for new snapshot
        _ = self._filesystem_store.load_for_snapshot(self._store.path)
        return _meta_response(meta)

    def _slice_items(self, encoded_slice_type: str) -> SnapshotSlicePayload:
        slice_type = unquote(encoded_slice_type)
        try:
            return self._store.slice_items(slice_type)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @staticmethod
    def _paginate_items(
        items: list[Mapping[str, JSONValue]], *, offset: int, limit: int | None
    ) -> list[Mapping[str, JSONValue]]:
        if offset:
            items = items[offset:]
        if limit is not None:
            items = items[:limit]
        return items

    def _execute_reload(self) -> SnapshotMeta:
        try:
            return self._store.reload()
        except SnapshotLoadError as error:
            self._logger.warning(
                "Snapshot reload failed",
                event="debug.server.reload_failed",
                context={"path": self._store.meta.path, "error": str(error)},
            )
            raise self._translate_snapshot_error(error) from error

    @staticmethod
    def _translate_snapshot_error(error: SnapshotLoadError) -> HTTPException:
        return HTTPException(status_code=400, detail=str(error))

    def _execute_snapshot_command(
        self, command: Callable[[], SnapshotMeta]
    ) -> SnapshotMeta:
        try:
            return command()
        except SnapshotLoadError as error:
            raise self._translate_snapshot_error(error) from error

    @staticmethod
    def _parse_select_payload(
        payload: Mapping[str, JSONValue],
    ) -> tuple[str | None, int | None]:
        session_value = payload.get("session_id")
        line_value = payload.get("line_number")

        if session_value is None and line_value is None:
            raise HTTPException(
                status_code=400,
                detail="session_id or line_number is required",
            )

        _validate_optional_session_id(session_value)
        _validate_optional_line_number(line_value)
        return cast(str | None, session_value), cast(int | None, line_value)

    # --- Filesystem Explorer Handlers ---

    def get_filesystem_tree(self) -> Mapping[str, JSONValue]:
        """Return the filesystem tree for the current snapshot."""
        fs = self._filesystem_store
        if not fs.has_archive:
            return {
                "has_archive": False,
                "archive_path": None,
                "metadata": None,
                "tree": None,
            }

        return {
            "has_archive": True,
            "archive_path": str(fs.archive_path) if fs.archive_path else None,
            "metadata": _filesystem_metadata_response(fs.metadata),
            "tree": _file_tree_response(fs.tree),
        }

    def get_filesystem_file(
        self,
        encoded_path: str,
        *,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int | None, Query(ge=0)] = None,
    ) -> Mapping[str, JSONValue]:
        """Return the content of a file from the filesystem archive."""
        file_path = unquote(encoded_path)
        content = self._filesystem_store.read_file(
            file_path, offset=offset, limit=limit
        )
        return _file_content_response(content)

    def download_filesystem_file(self, encoded_path: str) -> Response:
        """Return raw file content for download."""
        file_path = unquote(encoded_path)
        raw_bytes = self._filesystem_store.read_file_raw(file_path)
        if raw_bytes is None:
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

        # Determine content type based on file extension
        filename = file_path.rsplit("/", 1)[-1]
        content_type = _guess_content_type(filename)

        return Response(
            content=raw_bytes,
            media_type=content_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @staticmethod
    def _parse_switch_payload(
        payload: Mapping[str, JSONValue],
    ) -> tuple[Path, str | None, int | None]:
        path_value = payload.get("path")
        if not isinstance(path_value, str):
            raise HTTPException(status_code=400, detail="path is required")

        session_value = payload.get("session_id")
        _validate_optional_session_id(session_value)

        line_value = payload.get("line_number")
        _validate_optional_line_number(line_value)

        return (
            Path(path_value),
            cast(str | None, session_value),
            cast(int | None, line_value),
        )


def _filesystem_metadata_response(
    metadata: FilesystemArchiveMetadata | None,
) -> Mapping[str, JSONValue] | None:
    """Convert FilesystemArchiveMetadata to JSON response."""
    if metadata is None:
        return None
    return {
        "version": metadata.version,
        "created_at": metadata.created_at,
        "session_id": metadata.session_id,
        "root_path": metadata.root_path,
        "file_count": metadata.file_count,
        "total_bytes": metadata.total_bytes,
    }


def _file_tree_response(tree: FileTreeNode | None) -> Mapping[str, JSONValue] | None:
    """Convert FileTreeNode to JSON response."""
    if tree is None:  # pragma: no cover
        return None

    def convert(node: FileTreeNode) -> Mapping[str, JSONValue]:
        result: dict[str, JSONValue] = {
            "name": node.name,
            "type": node.type,
        }
        if node.path is not None:
            result["path"] = node.path
        if node.size is not None:
            result["size"] = node.size
        if node.children is not None:
            result["children"] = [convert(child) for child in node.children]
        return result

    return convert(tree)


def _file_content_response(content: FileContent) -> Mapping[str, JSONValue]:
    """Convert FileContent to JSON response."""
    result: dict[str, JSONValue] = {
        "path": content.path,
        "content": content.content,
        "size_bytes": content.size_bytes,
        "binary": content.binary,
    }
    if content.total_lines is not None:
        result["total_lines"] = content.total_lines
    if content.offset:
        result["offset"] = content.offset
    if content.limit is not None:
        result["limit"] = content.limit
    if content.truncated:
        result["truncated"] = content.truncated
    if content.error is not None:
        result["error"] = content.error
    return result


def _guess_content_type(filename: str) -> str:
    """Guess content type based on file extension."""
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    content_types: dict[str, str] = {
        "py": "text/x-python",
        "js": "text/javascript",
        "ts": "text/typescript",
        "json": "application/json",
        "yaml": "application/x-yaml",
        "yml": "application/x-yaml",
        "md": "text/markdown",
        "txt": "text/plain",
        "html": "text/html",
        "css": "text/css",
        "xml": "application/xml",
        "toml": "application/toml",
        "sh": "text/x-shellscript",
        "rs": "text/x-rust",
        "go": "text/x-go",
        "java": "text/x-java",
        "c": "text/x-c",
        "cpp": "text/x-c++",
        "h": "text/x-c",
        "hpp": "text/x-c++",
    }
    return content_types.get(extension, "application/octet-stream")


def _validate_optional_session_id(value: JSONValue | None) -> None:
    if value is not None and not isinstance(value, str):
        raise HTTPException(status_code=400, detail="session_id must be a string")


def _validate_optional_line_number(value: JSONValue | None) -> None:
    if value is not None and not isinstance(value, int):
        raise HTTPException(status_code=400, detail="line_number must be an integer")


def build_debug_app(store: SnapshotStore, logger: StructuredLogger) -> FastAPI:
    """Construct the FastAPI application for inspecting snapshots."""

    static_dir = Path(str(files(__package__).joinpath("static")))
    filesystem_store = FilesystemArchiveStore(logger=logger)
    handlers = _DebugAppHandlers(
        store=store,
        filesystem_store=filesystem_store,
        logger=logger,
        static_dir=static_dir,
    )

    app = FastAPI(title="wink snapshot debug server")
    app.state.snapshot_store = store
    app.state.filesystem_store = filesystem_store
    app.state.logger = logger
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    _ = app.get("/", response_class=HTMLResponse)(handlers.index)
    _ = app.get("/api/meta")(handlers.get_meta)
    _ = app.get("/api/entries")(handlers.list_entries)
    _ = app.get("/api/slices/{encoded_slice_type}")(handlers.get_slice)
    _ = app.get("/api/raw")(handlers.get_raw)
    _ = app.post("/api/reload")(handlers.reload)
    _ = app.get("/api/snapshots")(handlers.list_snapshots)
    _ = app.post("/api/select")(handlers.select)
    _ = app.post("/api/switch")(handlers.switch)

    # Filesystem explorer routes
    _ = app.get("/api/filesystem/tree")(handlers.get_filesystem_tree)
    _ = app.get("/api/filesystem/file/{encoded_path:path}")(
        handlers.get_filesystem_file
    )
    _ = app.get("/api/filesystem/download/{encoded_path:path}")(
        handlers.download_filesystem_file
    )

    return app


def run_debug_server(
    app: FastAPI,
    *,
    host: str,
    port: int,
    open_browser: bool,
    logger: StructuredLogger,
) -> int:
    """Run the uvicorn server for the supplied FastAPI app."""

    url = f"http://{host}:{port}/"
    if open_browser:
        threading.Timer(0.2, _open_browser, args=(url, logger)).start()

    logger.info(
        "Starting wink debug server",
        event="debug.server.start",
        context={"url": url},
    )

    try:
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_config=None,
        )
        server = uvicorn.Server(config)
        server.run()
    except Exception as error:  # pragma: no cover - exercised in wink tests
        logger.exception(
            "Failed to start wink debug server",
            event="debug.server.error",
            context={"url": url, "error": repr(error)},
        )
        return 3
    return 0


def _open_browser(url: str, logger: StructuredLogger) -> None:
    try:
        _ = webbrowser.open(url)
    except Exception as error:  # pragma: no cover - best effort logging
        logger.warning(
            "Unable to open browser",
            event="debug.server.browser",
            context={"url": url, "error": repr(error)},
        )
