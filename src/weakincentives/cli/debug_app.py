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

"""Static site generator and server for exploring session snapshot JSONL files."""

from __future__ import annotations

import contextlib
import http.server
import json
import re
import shutil
import socketserver
import tempfile
import threading
import webbrowser
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from functools import partial
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar, cast, override
from urllib.parse import quote

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


# ---------------------------------------------------------------------------
# Static Site Generation
# ---------------------------------------------------------------------------


def _encode_filename(name: str) -> str:
    """URL-encode a filename for use in paths."""
    return quote(name, safe="")


def _meta_to_json(meta: SnapshotMeta, file: str) -> Mapping[str, JSONValue]:
    """Convert SnapshotMeta to JSON-serializable dict for static output."""
    return {
        "version": meta.version,
        "created_at": meta.created_at,
        "file": file,
        "session_id": meta.session_id,
        "line_number": meta.line_number,
        "tags": dict(meta.tags),
        "validation_error": meta.validation_error,
        "slices": [
            {
                "slice_type": entry.slice_type,
                "item_type": entry.item_type,
                "count": entry.count,
            }
            for entry in meta.slices
        ],
    }


def _write_json(path: Path, data: JSONValue) -> None:
    """Write JSON data to a file atomically."""
    _ = path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    try:
        _ = tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        _ = tmp_path.rename(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


@FrozenDataclass()
class _SnapshotResult:
    """Result from processing a single snapshot file."""

    manifest_entry: Mapping[str, JSONValue]
    default_snapshot: str | None
    default_entry: int | None


def _setup_output_dirs(output_dir: Path, base_path: str) -> Path:
    """Set up output directory structure and copy static assets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    static_out = output_dir / "static"
    data_out = output_dir / "data"
    static_out.mkdir(exist_ok=True)
    data_out.mkdir(exist_ok=True)

    source_static = Path(str(files(__package__).joinpath("static")))
    for asset in ("style.css", "app.js"):
        src = source_static / asset
        if src.exists():
            _ = shutil.copy2(src, static_out / asset)

    _write_index_html(output_dir, source_static, base_path)
    return data_out


def _process_entry(
    entry: LoadedSnapshot,
    file_name: str,
    entry_dir: Path,
) -> Mapping[str, JSONValue]:
    """Process a single entry and write its data files."""
    meta = entry.meta
    _write_json(entry_dir / "meta.json", _meta_to_json(meta, file_name))
    _write_json(entry_dir / "raw.json", dict(entry.raw_payload))

    slices_dir = entry_dir / "slices"
    for slice_type, slice_payload in entry.slices.items():
        encoded_slice = _encode_filename(slice_type)
        rendered_items = [_render_markdown_values(item) for item in slice_payload.items]
        slice_data: Mapping[str, JSONValue] = {
            "slice_type": slice_payload.slice_type,
            "item_type": slice_payload.item_type,
            "items": rendered_items,
        }
        _write_json(slices_dir / f"{encoded_slice}.json", slice_data)

    return {
        "session_id": meta.session_id,
        "name": f"{meta.session_id} (line {meta.line_number})",
        "file": file_name,
        "line_number": meta.line_number,
        "created_at": meta.created_at,
        "tags": dict(meta.tags),
    }


def _process_snapshot_file(
    snapshot_file: Path,
    data_out: Path,
    logger: StructuredLogger,
) -> _SnapshotResult | None:
    """Process a single snapshot file and return its manifest entry."""
    try:
        entries = load_snapshot(snapshot_file)
    except SnapshotLoadError as error:
        logger.warning(
            "Skipping invalid snapshot file",
            event="wink.debug.snapshot_error",
            context={"path": str(snapshot_file), "error": str(error)},
        )
        return None

    file_name = snapshot_file.name
    encoded_file = _encode_filename(file_name)
    file_data_dir = data_out / "snapshots" / encoded_file

    logger.info(
        "Processing snapshot file",
        event="debug.generate.snapshot",
        context={"file": file_name, "entry_count": len(entries)},
    )

    entries_list: list[Mapping[str, JSONValue]] = []
    entry_line_numbers: list[int] = []

    for entry in entries:
        entry_dir = file_data_dir / "entries" / str(entry.meta.line_number)
        entries_list.append(_process_entry(entry, file_name, entry_dir))
        entry_line_numbers.append(entry.meta.line_number)

    _write_json(file_data_dir / "entries.json", entries_list)

    manifest_entry: Mapping[str, JSONValue] = {
        "file": file_name,
        "path": f"snapshots/{encoded_file}",
        "entry_count": len(entries),
        "entries": entry_line_numbers,
    }

    default_snap = file_name if entries else None
    default_ent = entries[0].meta.line_number if entries else None

    return _SnapshotResult(
        manifest_entry=manifest_entry,
        default_snapshot=default_snap,
        default_entry=default_ent,
    )


def generate_static_site(
    snapshot_path: Path,
    output_dir: Path,
    *,
    base_path: str = "/",
    logger: StructuredLogger | None = None,
) -> None:
    """Generate a static site from snapshot files.

    Args:
        snapshot_path: Path to a JSONL snapshot file or directory.
        output_dir: Directory to write static files to.
        base_path: URL path prefix for deployment.
        logger: Optional logger for progress messages.
    """
    log = logger or get_logger(__name__)

    log.info(
        "Starting static site generation",
        event="debug.generate.start",
        context={"output": str(output_dir), "base_path": base_path},
    )

    resolved = snapshot_path.resolve()
    snapshot_files = (
        _iter_snapshot_files(resolved)
        if resolved.is_dir()
        else [resolved]
    )

    if not snapshot_files:
        msg = f"No snapshot files found at {snapshot_path}"
        raise SnapshotLoadError(msg)

    data_out = _setup_output_dirs(output_dir, base_path)

    manifest_snapshots: list[Mapping[str, JSONValue]] = []
    default_snapshot: str | None = None
    default_entry: int | None = None

    for snapshot_file in sorted(
        snapshot_files, key=lambda p: p.stat().st_mtime, reverse=True
    ):
        result = _process_snapshot_file(snapshot_file, data_out, log)
        if result is None:
            continue

        manifest_snapshots.append(result.manifest_entry)
        if default_snapshot is None:
            default_snapshot = result.default_snapshot
            default_entry = result.default_entry

    manifest: Mapping[str, JSONValue] = {
        "version": "1",
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "base_path": base_path,
        "snapshots": manifest_snapshots,
        "default_snapshot": default_snapshot,
        "default_entry": default_entry,
    }
    _write_json(data_out / "manifest.json", manifest)

    log.info(
        "Static site generation complete",
        event="debug.generate.complete",
        context={"output": str(output_dir), "file_count": len(manifest_snapshots)},
    )


def _write_index_html(output_dir: Path, source_static: Path, base_path: str) -> None:
    """Write index.html with base path injected."""
    source_html = source_static / "index.html"
    html_content = source_html.read_text()

    # Ensure base_path ends with /
    if not base_path.endswith("/"):
        base_path += "/"

    # Inject base tag after <head> for non-root deployments
    if base_path != "/" and "<base" not in html_content:
        base_tag = f'<base href="{base_path}">'
        html_content = html_content.replace("<head>", f"<head>\n    {base_tag}", 1)

    _ = (output_dir / "index.html").write_text(html_content)


def _iter_snapshot_files(root: Path) -> list[Path]:
    """Find all snapshot files in a directory."""
    candidates: list[Path] = []
    for pattern in ("*.jsonl", "*.json"):
        candidates.extend(p for p in root.glob(pattern) if p.is_file())
    return candidates


# ---------------------------------------------------------------------------
# Debug Server (serves generated static files)
# ---------------------------------------------------------------------------


class _ReloadHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler that supports /api/reload to regenerate static files."""

    # Class variables set before handler instantiation
    snapshot_path: ClassVar[Path]
    output_dir: ClassVar[Path]
    base_path: ClassVar[str]
    logger: ClassVar[StructuredLogger]

    def __init__(self, *args: object, **kwargs: object) -> None:
        # SimpleHTTPRequestHandler expects directory as keyword arg
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    def do_POST(self) -> None:
        """Handle POST requests for /api/reload."""
        if self.path == "/api/reload":
            self._handle_reload()
        else:
            self.send_error(404, "Not Found")

    def _handle_reload(self) -> None:
        """Regenerate static files and return success."""
        try:
            generate_static_site(
                self.snapshot_path,
                self.output_dir,
                base_path=self.base_path,
                logger=self.logger,
            )
            self.logger.info(
                "Snapshot reloaded",
                event="debug.server.reload",
                context={"temp_dir": str(self.output_dir)},
            )
            response = {
                "success": True,
                "generated_at": datetime.now(tz=UTC).isoformat(),
            }
            self._send_json_response(200, response)
        except SnapshotLoadError as error:
            self._send_json_response(400, {"success": False, "error": str(error)})

    def _send_json_response(self, code: int, data: Mapping[str, JSONValue]) -> None:
        """Send a JSON response."""
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        _ = self.wfile.write(body)

    @override
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Suppress default logging (overrides parent method)."""
        del format, args  # Unused


def run_debug_server(
    snapshot_path: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
    logger: StructuredLogger | None = None,
) -> int:
    """Run a debug server that generates and serves static files.

    Args:
        snapshot_path: Path to snapshot file or directory.
        host: Host interface to bind to.
        port: Port to bind to.
        open_browser: Whether to open browser automatically.
        logger: Optional logger instance.

    Returns:
        Exit code (0 for success, 3 for server error).
    """
    log = logger or get_logger(__name__)

    # Create temporary directory for generated files
    temp_dir = Path(tempfile.mkdtemp(prefix="wink-debug-"))

    try:
        # Generate initial static site
        generate_static_site(
            snapshot_path,
            temp_dir,
            base_path="/",
            logger=log,
        )

        url = f"http://{host}:{port}/"
        log.info(
            "Starting wink debug server",
            event="debug.server.start",
            context={"url": url, "temp_dir": str(temp_dir)},
        )

        # Create handler class with our config
        handler_class = partial(
            _ReloadHandler,
            directory=str(temp_dir),
        )
        # Attach config to class (accessed via self.__class__)
        handler_class.func.snapshot_path = snapshot_path  # type: ignore[attr-defined]
        handler_class.func.output_dir = temp_dir  # type: ignore[attr-defined]
        handler_class.func.base_path = "/"  # type: ignore[attr-defined]
        handler_class.func.logger = log  # type: ignore[attr-defined]

        if open_browser:
            threading.Timer(0.2, _open_browser, args=(url, log)).start()

        # Use TCPServer with address reuse
        socketserver.TCPServer.allow_reuse_address = True
        with (
            socketserver.TCPServer((host, port), handler_class) as httpd,
            contextlib.suppress(KeyboardInterrupt),
        ):
            httpd.serve_forever()

    except Exception as error:  # pragma: no cover
        log.exception(
            "Failed to start wink debug server",
            event="debug.server.error",
            context={"error": repr(error)},
        )
        return 3

    else:
        return 0

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def _open_browser(url: str, logger: StructuredLogger) -> None:
    """Open URL in default browser."""
    try:
        _ = webbrowser.open(url)
    except Exception as error:  # pragma: no cover
        logger.warning(
            "Unable to open browser",
            event="debug.server.browser",
            context={"url": url, "error": repr(error)},
        )
