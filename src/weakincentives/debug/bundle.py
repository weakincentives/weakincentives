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

"""Debug bundle for capturing MainLoop execution state.

A debug bundle is a self-contained zip archive capturing everything needed to
understand, reproduce, and debug a MainLoop execution. Bundles unify session
state, logs, filesystem snapshots, configuration, and metrics into a single
portable artifact.

Example::

    with BundleWriter(target="./debug/", bundle_id=run_id) as writer:
        writer.write_session_before(session)
        writer.write_request_input(request)
        with writer.capture_logs():
            response = adapter.evaluate(prompt, session=session)
        writer.write_session_after(session)
        writer.write_request_output(response)
        writer.write_config(config)
        writer.write_run_context(run_context)
        writer.write_metrics(metrics)
    # Bundle finalized on exit: README generated, checksums computed, zip created

"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
import zipfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Self, override
from uuid import UUID, uuid4

from ..dataclasses import FrozenDataclass
from ..errors import WinkError
from ..serde import dump
from ..types import JSONValue

if TYPE_CHECKING:
    from ..filesystem import Filesystem
    from ..runtime.run_context import RunContext
    from ..runtime.session import Session
    from .environment import EnvironmentCapture

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal log collector (inlined from _log_collector.py)
# ---------------------------------------------------------------------------


class _LogCollectorHandler(logging.Handler):
    """Handler that captures log records and writes them to a file as JSONL."""

    def __init__(
        self,
        file_handle: IO[str],
        level: int = logging.DEBUG,
    ) -> None:
        super().__init__(level=level)
        self._file = file_handle

    @override
    def emit(self, record: logging.LogRecord) -> None:
        """Convert log record to JSON and write to file."""
        try:
            json_line = self._record_to_json(record)
            _ = self._file.write(json_line + "\n")
            self._file.flush()
        except (OSError, TypeError, ValueError):  # pragma: no cover
            self.handleError(record)

    @staticmethod
    def _record_to_json(record: logging.LogRecord) -> str:
        """Convert a logging.LogRecord to a JSON string."""
        from typing import cast

        event = getattr(record, "event", "")
        raw_context = getattr(record, "context", {})

        context_dict: dict[str, JSONValue] = {}
        if isinstance(raw_context, Mapping):  # pragma: no branch
            source = cast("Mapping[str, JSONValue]", raw_context)
            context_dict = dict(source)

        data: dict[str, JSONValue] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": event if isinstance(event, str) else "",
            "message": record.getMessage(),
            "context": context_dict,
        }
        return json.dumps(data, ensure_ascii=False)


@contextmanager
def collect_all_logs(
    target: str | Path,
    *,
    level: int = logging.DEBUG,
) -> Iterator[Path]:
    """Capture all log records and write them to a file.

    Internal helper for BundleWriter.capture_logs(). Attaches a handler to the
    root logger and writes captured records as JSONL.

    Args:
        target: Path to the output file.
        level: Minimum log level to capture.

    Yields:
        Path to the log file.
    """
    path = Path(target).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    with path.open("a", encoding="utf-8") as file_handle:
        handler = _LogCollectorHandler(file_handle, level=level)
        root_logger.addHandler(handler)
        try:
            yield path
        finally:
            root_logger.removeHandler(handler)
            handler.close()


# ---------------------------------------------------------------------------
# Bundle constants and types
# ---------------------------------------------------------------------------

BUNDLE_FORMAT_VERSION = "1.1.0"
BUNDLE_ROOT_DIR = "debug_bundle"


class BundleError(WinkError, RuntimeError):
    """Base error for bundle operations."""


class BundleValidationError(BundleError):
    """Raised when bundle validation fails."""


@FrozenDataclass()
class BundleConfig:
    """Configuration for bundle creation.

    Attributes:
        target: Output directory for bundles. None disables bundling.
        max_file_size: Skip files larger than this (bytes). Default 10MB.
        max_total_size: Maximum filesystem capture size (bytes). Default 50MB.
        compression: Zip compression method.
    """

    target: Path | None = None
    max_file_size: int = 10_000_000  # 10MB
    max_total_size: int = 52_428_800  # 50MB
    compression: str = "deflate"

    @classmethod
    def __pre_init__(
        cls,
        *,
        target: Path | str | None = None,
        max_file_size: int = 10_000_000,
        max_total_size: int = 52_428_800,
        compression: str = "deflate",
    ) -> Mapping[str, object]:
        """Normalize inputs before construction."""
        normalized_target = Path(target) if isinstance(target, str) else target
        return {
            "target": normalized_target,
            "max_file_size": max_file_size,
            "max_total_size": max_total_size,
            "compression": compression,
        }

    @property
    def enabled(self) -> bool:
        """Return True if bundling is enabled (target is set)."""
        return self.target is not None


@FrozenDataclass()
class RedactionInfo:
    """Redaction metadata for manifest."""

    enabled: bool = False
    ruleset: str = ""
    patterns: tuple[str, ...] = ()


@FrozenDataclass()
class LimitsApplied:
    """Limits applied during capture."""

    filesystem_truncated: bool = False
    logs_truncated: bool = False
    tool_outputs_truncated: bool = False


@FrozenDataclass()
class CaptureInfo:
    """Capture metadata for manifest."""

    mode: str
    trigger: str
    limits_applied: LimitsApplied = field(default_factory=LimitsApplied)
    redaction: RedactionInfo = field(default_factory=RedactionInfo)


@FrozenDataclass()
class PromptInfo:
    """Prompt metadata for manifest."""

    ns: str = ""
    key: str = ""
    adapter: str = ""


@FrozenDataclass()
class RequestInfo:
    """Request metadata for manifest."""

    request_id: str
    session_id: str | None = None
    status: str = "success"
    started_at: str = ""
    ended_at: str = ""


@FrozenDataclass()
class IntegrityInfo:
    """Integrity metadata for manifest."""

    algorithm: str = "sha256"
    checksums: Mapping[str, str] = field(default_factory=lambda: dict[str, str]())


@FrozenDataclass()
class BuildInfo:
    """Build metadata for manifest."""

    version: str = ""
    commit: str = ""


@FrozenDataclass()
class TokenUsage:
    """Token usage summary."""

    input: int = 0
    output: int = 0
    cached: int = 0


@FrozenDataclass()
class BundleSummary:
    """Execution summary for manifest."""

    duration_ms: int = 0
    status: str = "success"
    error_count: int = 0
    prompt_count: int = 0
    tool_call_count: int = 0
    provider_call_count: int = 0
    token_usage: TokenUsage = field(default_factory=TokenUsage)


@FrozenDataclass()
class IndexPointer:
    """Pointer to an index file for an artifact."""

    path: str


@FrozenDataclass()
class TimeRange:
    """Time range for temporal artifacts like logs."""

    start: str = ""
    end: str = ""


@FrozenDataclass()
class LogCounts:
    """Count metadata for log artifacts.

    The `records` field is required to distinguish from FilesystemCounts
    during serde union parsing.
    """

    records: int  # Required - no default
    levels: Mapping[str, int] = field(default_factory=lambda: dict[str, int]())


@FrozenDataclass()
class FilesystemCapture:
    """Capture parameters for filesystem artifacts."""

    max_file_size: int = 10_000_000
    max_total_size: int = 52_428_800
    excluded_patterns: tuple[str, ...] = ()


@FrozenDataclass()
class FilesystemCounts:
    """Count metadata for filesystem artifacts.

    The `files_captured` field is required to distinguish from LogCounts
    during serde union parsing.
    """

    files_captured: int  # Required - no default
    files_skipped: int = 0
    total_bytes_captured: int = 0


@FrozenDataclass()
class SchemaInfo:
    """Schema information for structured artifacts."""

    type: str = ""
    version: str = ""


@FrozenDataclass()
class ArtifactInfo:
    """Metadata for a single artifact in the bundle.

    The artifact is identified by a logical ID (e.g., 'logs', 'session_after',
    'filesystem') rather than its file path. This allows agents to understand
    the artifact's purpose without parsing file paths.
    """

    path: str
    kind: str  # 'json', 'jsonl', 'text', 'directory'
    content_type: str = ""
    size_bytes: int = 0
    sha256: str = ""
    index: IndexPointer | None = None
    time_range: TimeRange | None = None
    counts: FilesystemCounts | LogCounts | None = None
    schema: SchemaInfo | None = None
    capture: FilesystemCapture | None = None


@FrozenDataclass()
class BundleManifest:
    """Bundle manifest containing metadata and integrity checksums.

    Schema::

        {
          "format_version": "1.1.0",
          "bundle_id": "uuid",
          "created_at": "2024-01-15T10:30:00+00:00",
          "request": { ... },
          "capture": { ... },
          "summary": { ... },
          "artifacts": { "logs": { ... }, "session_after": { ... }, ... },
          "prompt": { ... },
          "files": ["manifest.json", ...],
          "integrity": { ... },
          "build": { ... }
        }
    """

    format_version: str = BUNDLE_FORMAT_VERSION
    bundle_id: str = ""
    created_at: str = ""
    request: RequestInfo = field(default_factory=lambda: RequestInfo(request_id=""))
    capture: CaptureInfo = field(
        default_factory=lambda: CaptureInfo(mode="full", trigger="config")
    )
    summary: BundleSummary = field(default_factory=BundleSummary)
    artifacts: Mapping[str, ArtifactInfo] = field(
        default_factory=lambda: dict[str, ArtifactInfo]()
    )
    prompt: PromptInfo = field(default_factory=PromptInfo)
    files: tuple[str, ...] = ()
    integrity: IntegrityInfo = field(default_factory=IntegrityInfo)
    build: BuildInfo = field(default_factory=BuildInfo)

    def to_json(self) -> str:
        """Serialize manifest to JSON string."""
        return json.dumps(dump(self), indent=2)

    @classmethod
    def from_json(cls, raw: str) -> BundleManifest:
        """Deserialize manifest from JSON string."""
        from ..serde import parse

        data: JSONValue = json.loads(raw)
        if not isinstance(data, Mapping):
            raise BundleValidationError("Manifest must be a JSON object")
        return parse(cls, data)


def _generate_readme(manifest: BundleManifest) -> str:
    """Generate human-readable README for bundle navigation."""
    lines = [
        "Debug Bundle",
        "=" * 60,
        "",
        f"Bundle ID: {manifest.bundle_id}",
        f"Created: {manifest.created_at}",
        f"Status: {manifest.request.status}",
        "",
        "Contents",
        "-" * 60,
        "",
        "manifest.json       Bundle metadata and integrity checksums",
        "README.txt          This file",
        "",
        "request/",
        "  input.json        MainLoop request",
        "  output.json       MainLoop response",
        "",
        "session/",
        "  before.jsonl      Session state before execution",
        "  after.jsonl       Session state after execution",
        "",
        "logs/",
        "  app.jsonl         Log records during execution",
        "",
        "environment/        Reproducibility envelope",
        "  system.json       OS, kernel, arch, CPU, memory",
        "  python.json       Python version, executable, venv info",
        "  packages.txt      Installed packages (pip freeze)",
        "  env_vars.json     Environment variables (filtered/redacted)",
        "  git.json          Repo root, commit, branch, remotes",
        "  git.diff          Uncommitted changes (if any)",
        "  command.txt       argv, working dir, entrypoint",
        "  container.json    Container runtime info (if applicable)",
        "",
        "config.json         MainLoop and adapter configuration",
        "run_context.json    Execution context (IDs, tracing)",
        "metrics.json        Token usage, timing, budget state",
        "",
        "Optional files (if present):",
        "  prompt_overrides.json   Visibility overrides",
        "  error.json              Error details (if failed)",
        "  eval.json               Eval metadata (EvalLoop only)",
        "  filesystem/             Workspace snapshot",
        "",
        "=" * 60,
        f"Format version: {manifest.format_version}",
    ]
    return "\n".join(lines)


def _compute_checksum(content: bytes) -> str:
    """Compute SHA-256 checksum of content."""
    return hashlib.sha256(content).hexdigest()


def _serialize_object(obj: object) -> JSONValue:
    """Serialize object to JSON-compatible form.

    Tries serde.dump() for dataclasses, falls back to the object itself
    for dicts and other JSON-serializable types.
    """
    from dataclasses import is_dataclass

    if is_dataclass(obj) and not isinstance(obj, type):
        return dump(obj)
    # For plain dicts and other JSON-serializable objects, return as-is
    # Cast is safe since caller ensures JSON-serializability via json.dumps
    from typing import cast

    return cast(JSONValue, obj)


def _get_compression_type(compression: str) -> int:
    """Map compression string to zipfile constant."""
    if compression == "deflate":
        return zipfile.ZIP_DEFLATED
    if compression == "stored":
        return zipfile.ZIP_STORED
    if compression == "bzip2":
        return zipfile.ZIP_BZIP2
    if compression == "lzma":
        return zipfile.ZIP_LZMA
    return zipfile.ZIP_DEFLATED


class BundleWriter:
    """Context manager for streaming bundle creation.

    Creates a debug bundle atomically - either the full bundle is written
    or nothing is written. Artifacts are written to a temporary directory
    during the context, then finalized to a zip archive on exit.

    Example::

        with BundleWriter(target="./debug/", bundle_id=run_id) as writer:
            writer.write_request_input(request)
            with writer.capture_logs():
                # ... execution ...
                pass
            writer.write_request_output(response)
        # Bundle is now at writer.path
    """

    _target: Path
    _bundle_id: UUID
    _config: BundleConfig
    _temp_dir: Path | None
    _started_at: datetime
    _ended_at: datetime | None
    _files: list[str]
    _checksums: dict[str, str]
    _sizes: dict[str, int]  # Track artifact sizes for manifest
    _request_id: UUID | None
    _session_id: UUID | None
    _status: str
    _prompt_info: PromptInfo
    _trigger: str
    _limits_applied: LimitsApplied
    _finalized: bool
    _path: Path | None
    _log_collector_path: Path | None
    # Filesystem capture stats
    _fs_files_captured: int
    _fs_files_skipped: int
    _fs_total_bytes: int

    def __init__(
        self,
        target: Path | str,
        *,
        bundle_id: UUID | None = None,
        config: BundleConfig | None = None,
        trigger: str = "config",
    ) -> None:
        """Initialize bundle writer.

        Args:
            target: Output directory for the bundle zip file.
            bundle_id: Unique identifier for this bundle. Auto-generated if None.
            config: Bundle configuration. Uses defaults if None.
            trigger: What triggered bundle creation (config, env, request).
        """
        super().__init__()
        self._target = Path(target)
        self._bundle_id = bundle_id if bundle_id is not None else uuid4()
        self._config = config if config is not None else BundleConfig()
        self._temp_dir = None
        self._started_at = datetime.now(UTC)
        self._ended_at = None
        self._files = []
        self._checksums = {}
        self._sizes = {}
        self._request_id = None
        self._session_id = None
        self._status = "success"
        self._prompt_info = PromptInfo()
        self._trigger = trigger
        self._limits_applied = LimitsApplied()
        self._finalized = False
        self._path = None
        self._log_collector_path = None
        # Filesystem capture stats
        self._fs_files_captured = 0
        self._fs_files_skipped = 0
        self._fs_total_bytes = 0

    @property
    def bundle_id(self) -> UUID:
        """Return the bundle ID."""
        return self._bundle_id

    @property
    def path(self) -> Path | None:
        """Return path to the created bundle, or None if not finalized."""
        return self._path

    def __enter__(self) -> Self:
        """Enter context and create temporary directory."""
        self._temp_dir = Path(tempfile.mkdtemp(prefix="debug_bundle_"))
        # Create directory structure
        (self._temp_dir / BUNDLE_ROOT_DIR).mkdir()
        (self._temp_dir / BUNDLE_ROOT_DIR / "request").mkdir()
        (self._temp_dir / BUNDLE_ROOT_DIR / "session").mkdir()
        (self._temp_dir / BUNDLE_ROOT_DIR / "logs").mkdir()
        (self._temp_dir / BUNDLE_ROOT_DIR / "environment").mkdir()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context, finalize bundle, and clean up."""
        try:
            if exc_type is not None:
                # Record error status
                self._status = "error"
                if exc_val is not None:  # pragma: no branch
                    self._write_error_from_exception(exc_val)

            self._finalize()
        except Exception:
            _logger.exception(
                "Failed to finalize debug bundle",
                extra={"bundle_id": str(self._bundle_id)},
            )
        finally:
            # Clean up temp directory
            if (
                self._temp_dir is not None and self._temp_dir.exists()
            ):  # pragma: no branch
                shutil.rmtree(self._temp_dir, ignore_errors=True)

    def _write_artifact(self, rel_path: str, content: bytes | str) -> None:
        """Write an artifact to the bundle."""
        if self._temp_dir is None:
            raise BundleError("BundleWriter not entered")

        full_path = self._temp_dir / BUNDLE_ROOT_DIR / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        content_bytes = content.encode("utf-8") if isinstance(content, str) else content

        _ = full_path.write_bytes(content_bytes)
        self._files.append(rel_path)
        self._checksums[rel_path] = _compute_checksum(content_bytes)
        self._sizes[rel_path] = len(content_bytes)

    def write_request_input(self, request: object) -> None:
        """Write the MainLoop request input."""
        try:
            content = json.dumps(_serialize_object(request), indent=2)
            self._write_artifact("request/input.json", content)
        except Exception:
            _logger.exception(
                "Failed to write request input",
                extra={"bundle_id": str(self._bundle_id)},
            )

    def write_request_output(self, response: object) -> None:
        """Write the MainLoop response output."""
        try:
            content = json.dumps(_serialize_object(response), indent=2)
            self._write_artifact("request/output.json", content)
        except Exception:
            _logger.exception(
                "Failed to write request output",
                extra={"bundle_id": str(self._bundle_id)},
            )

    def write_session_before(self, session: Session) -> None:
        """Write session state before execution."""
        try:
            self._session_id = session.session_id
            snapshot = session.snapshot(include_all=True)
            if snapshot.slices:
                content = snapshot.to_json() + "\n"
                self._write_artifact("session/before.jsonl", content)
        except Exception:
            _logger.exception(
                "Failed to write session before",
                extra={"bundle_id": str(self._bundle_id)},
            )

    def write_session_after(self, session: Session) -> None:
        """Write session state after execution."""
        try:
            self._session_id = session.session_id
            snapshot = session.snapshot(include_all=True)
            if snapshot.slices:
                content = snapshot.to_json() + "\n"
                self._write_artifact("session/after.jsonl", content)
        except Exception:
            _logger.exception(
                "Failed to write session after",
                extra={"bundle_id": str(self._bundle_id)},
            )

    @contextmanager
    def capture_logs(self) -> Iterator[None]:
        """Context manager to capture logs during execution."""
        if self._temp_dir is None:
            yield
            return

        log_path = self._temp_dir / BUNDLE_ROOT_DIR / "logs" / "app.jsonl"
        self._log_collector_path = log_path

        try:
            with collect_all_logs(log_path, level=logging.DEBUG):
                yield
        except Exception:
            _logger.exception(
                "Error during log capture",
                extra={"bundle_id": str(self._bundle_id)},
            )
            raise  # Re-raise to allow bundle finalization to capture error status

    def write_config(self, config: object) -> None:
        """Write MainLoop and adapter configuration."""
        try:
            content = json.dumps(_serialize_object(config), indent=2)
            self._write_artifact("config.json", content)
        except Exception:
            _logger.exception(
                "Failed to write config",
                extra={"bundle_id": str(self._bundle_id)},
            )

    def write_run_context(self, run_context: RunContext) -> None:
        """Write execution context."""
        try:
            self._request_id = run_context.request_id
            self._session_id = run_context.session_id
            content = json.dumps(_serialize_object(run_context), indent=2)
            self._write_artifact("run_context.json", content)
        except Exception:
            _logger.exception(
                "Failed to write run context",
                extra={"bundle_id": str(self._bundle_id)},
            )

    def write_metrics(self, metrics: object) -> None:
        """Write timing phases, token consumption, and budget state."""
        try:
            content = json.dumps(_serialize_object(metrics), indent=2)
            self._write_artifact("metrics.json", content)
        except Exception:
            _logger.exception(
                "Failed to write metrics",
                extra={"bundle_id": str(self._bundle_id)},
            )

    def write_prompt_overrides(self, overrides: object) -> None:
        """Write visibility overrides accumulated during execution."""
        try:
            content = json.dumps(_serialize_object(overrides), indent=2)
            self._write_artifact("prompt_overrides.json", content)
        except Exception:
            _logger.exception(
                "Failed to write prompt overrides",
                extra={"bundle_id": str(self._bundle_id)},
            )

    def write_error(self, error_info: Mapping[str, Any]) -> None:
        """Write error details."""
        try:
            self._status = "error"
            content = json.dumps(error_info, indent=2)
            self._write_artifact("error.json", content)
        except Exception:
            _logger.exception(
                "Failed to write error",
                extra={"bundle_id": str(self._bundle_id)},
            )

    def _write_error_from_exception(self, exc: BaseException) -> None:
        """Write error details from an exception."""
        import traceback

        error_info: dict[str, Any] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__),
        }
        self.write_error(error_info)

    def write_eval(self, eval_info: Mapping[str, Any]) -> None:
        """Write eval metadata for EvalLoop bundles."""
        try:
            content = json.dumps(eval_info, indent=2)
            self._write_artifact("eval.json", content)
        except Exception:
            _logger.exception(
                "Failed to write eval info",
                extra={"bundle_id": str(self._bundle_id)},
            )

    def write_environment(
        self,
        env: EnvironmentCapture | None = None,
        *,
        include_packages: bool = True,
        include_git_diff: bool = True,
    ) -> None:
        """Write reproducibility envelope (environment capture).

        Captures system, Python, packages, env vars, git state, command info,
        and container info into the environment/ directory.

        Args:
            env: Pre-captured environment, or None to capture now.
            include_packages: Whether to capture installed packages (slower).
            include_git_diff: Whether to capture git diff (may be large).
        """
        try:
            from .environment import capture_environment

            if env is None:
                env = capture_environment(
                    include_packages=include_packages,
                    include_git_diff=include_git_diff,
                )

            # Write system.json
            system_data = _serialize_object(env.system)
            self._write_artifact(
                "environment/system.json", json.dumps(system_data, indent=2)
            )

            # Write python.json
            python_data = _serialize_object(env.python)
            self._write_artifact(
                "environment/python.json", json.dumps(python_data, indent=2)
            )

            # Write packages.txt
            if env.packages:
                self._write_artifact("environment/packages.txt", env.packages)

            # Write env_vars.json
            self._write_artifact(
                "environment/env_vars.json", json.dumps(env.env_vars, indent=2)
            )

            # Write git.json (if in a git repo)
            if env.git is not None:
                git_data = _serialize_object(env.git)
                self._write_artifact(
                    "environment/git.json", json.dumps(git_data, indent=2)
                )

            # Write git.diff (if available and non-empty)
            if env.git_diff:
                self._write_artifact("environment/git.diff", env.git_diff)

            # Write command.txt
            command_lines = [
                f"Working Directory: {env.command.working_dir}",
                f"Executable: {env.command.executable}",
                f"Entrypoint: {env.command.entrypoint}",
                f"Arguments: {' '.join(env.command.argv)}",
            ]
            self._write_artifact("environment/command.txt", "\n".join(command_lines))

            # Write container.json (if containerized)
            if env.container is not None:  # pragma: no cover
                container_data = _serialize_object(env.container)
                self._write_artifact(
                    "environment/container.json", json.dumps(container_data, indent=2)
                )

        except Exception:
            _logger.exception(
                "Failed to write environment",
                extra={"bundle_id": str(self._bundle_id)},
            )

    def write_filesystem(
        self,
        fs: Filesystem,
        *,
        root_path: str = ".",
    ) -> None:
        """Write workspace filesystem snapshot."""
        try:
            self._archive_filesystem(fs, root_path)
        except Exception:
            _logger.exception(
                "Failed to write filesystem",
                extra={"bundle_id": str(self._bundle_id)},
            )

    def _archive_filesystem(self, fs: Filesystem, root_path: str) -> None:
        """Archive filesystem contents to bundle."""
        from dataclasses import replace as dc_replace

        if self._temp_dir is None:  # pragma: no cover
            return

        fs_dir = self._temp_dir / BUNDLE_ROOT_DIR / "filesystem"
        fs_dir.mkdir(parents=True, exist_ok=True)

        total_size = 0
        truncated = False
        files_captured = 0
        files_skipped = 0

        files = self._collect_files(fs, root_path)
        for file_path in files:
            if total_size >= self._config.max_total_size:
                truncated = True
                files_skipped += len(files) - files_captured - files_skipped
                break

            try:
                stat = fs.stat(file_path)
                if stat.size_bytes > self._config.max_file_size:
                    files_skipped += 1
                    continue

                result = fs.read_bytes(file_path)
                content = result.content

                # Normalize path for bundle
                rel_path = file_path.lstrip("./")
                dest_path = fs_dir / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                _ = dest_path.write_bytes(content)

                bundle_rel = f"filesystem/{rel_path}"
                self._files.append(bundle_rel)
                self._checksums[bundle_rel] = _compute_checksum(content)
                self._sizes[bundle_rel] = len(content)

                total_size += len(content)
                files_captured += 1

            except (FileNotFoundError, PermissionError, IsADirectoryError):
                files_skipped += 1
                continue

        # Update filesystem stats
        self._fs_files_captured = files_captured
        self._fs_files_skipped = files_skipped
        self._fs_total_bytes = total_size
        self._limits_applied = dc_replace(
            self._limits_applied, filesystem_truncated=truncated
        )

    def _collect_files(self, fs: Filesystem, path: str) -> list[str]:
        """Recursively collect all file paths from filesystem."""
        files: list[str] = []
        try:
            entries = fs.list(path)
        except (FileNotFoundError, NotADirectoryError):
            return files

        for entry in entries:
            if entry.is_file:
                files.append(entry.path)
            elif entry.is_directory:  # pragma: no branch
                files.extend(self._collect_files(fs, entry.path))
        return files

    def set_prompt_info(
        self, *, ns: str = "", key: str = "", adapter: str = ""
    ) -> None:
        """Set prompt metadata for manifest."""
        self._prompt_info = PromptInfo(ns=ns, key=key, adapter=adapter)

    def _finalize(self) -> None:
        """Finalize bundle: generate README, compute manifest, create zip."""
        from dataclasses import replace as dc_replace

        if self._finalized or self._temp_dir is None:
            return

        self._ended_at = datetime.now(UTC)
        log_stats = self._finalize_logs()
        ctx = self._build_artifact_context(log_stats)

        # Compute duration and build summary
        delta = self._ended_at - self._started_at
        summary = BundleSummary(
            duration_ms=int(delta.total_seconds() * 1000),
            status=self._status,
            error_count=1 if self._status == "error" else 0,
        )

        # Build manifest
        manifest = self._build_manifest(summary, _build_artifacts_map(ctx))

        # Write README and update manifest
        self._write_artifact("README.txt", _generate_readme(manifest))
        manifest = dc_replace(
            manifest,
            artifacts=_build_artifacts_map(self._build_artifact_context(log_stats)),
            files=tuple(sorted(self._files)),
            integrity=IntegrityInfo(algorithm="sha256", checksums=self._checksums),
        )

        # Write manifest and create zip
        self._write_manifest_file(manifest)
        self._path = self._create_zip_archive()
        self._finalized = True

        _logger.info(
            "Debug bundle created",
            extra={
                "bundle_id": str(self._bundle_id),
                "bundle_path": str(self._path),
                "file_count": len(self._files),
            },
        )

    def _build_artifact_context(
        self, log_stats: tuple[int, dict[str, int], str, str]
    ) -> _ArtifactBuildContext:
        """Build the artifact context for the artifacts map."""
        return _ArtifactBuildContext(
            files=tuple(self._files),
            sizes=self._sizes,
            checksums=self._checksums,
            config=self._config,
            fs_stats=(
                self._fs_files_captured,
                self._fs_files_skipped,
                self._fs_total_bytes,
            ),
            log_stats=log_stats,
        )

    def _build_manifest(
        self,
        summary: BundleSummary,
        artifacts: dict[str, ArtifactInfo],
    ) -> BundleManifest:
        """Build the bundle manifest."""
        return BundleManifest(
            format_version=BUNDLE_FORMAT_VERSION,
            bundle_id=str(self._bundle_id),
            created_at=self._started_at.isoformat(),
            request=RequestInfo(
                request_id=str(self._request_id) if self._request_id else "",
                session_id=str(self._session_id) if self._session_id else None,
                status=self._status,
                started_at=self._started_at.isoformat(),
                ended_at=self._ended_at.isoformat() if self._ended_at else "",
            ),
            capture=CaptureInfo(
                mode="full",
                trigger=self._trigger,
                limits_applied=self._limits_applied,
            ),
            summary=summary,
            artifacts=artifacts,
            prompt=self._prompt_info,
            files=tuple(sorted(self._files)),
            integrity=IntegrityInfo(algorithm="sha256", checksums=self._checksums),
            build=BuildInfo(),
        )

    def _write_manifest_file(self, manifest: BundleManifest) -> None:
        """Write manifest JSON to the bundle."""
        if self._temp_dir is None:  # pragma: no cover
            return
        manifest_path = self._temp_dir / BUNDLE_ROOT_DIR / "manifest.json"
        _ = manifest_path.write_text(manifest.to_json(), encoding="utf-8")
        self._files.append("manifest.json")

    def _create_zip_archive(self) -> Path:
        """Create the zip archive atomically."""
        if self._temp_dir is None:  # pragma: no cover
            raise BundleError("BundleWriter not entered")

        self._target.mkdir(parents=True, exist_ok=True)
        zip_name = f"{self._bundle_id}_{self._started_at.strftime('%Y%m%d_%H%M%S')}.zip"
        zip_path = self._target / zip_name
        tmp_path = self._target / f"{zip_name}.tmp"

        compression = _get_compression_type(self._config.compression)
        try:
            with zipfile.ZipFile(tmp_path, "w", compression) as zf:
                bundle_dir = self._temp_dir / BUNDLE_ROOT_DIR
                for root, _, files in bundle_dir.walk():
                    for file in files:
                        file_path = root / file
                        arcname = (
                            f"{BUNDLE_ROOT_DIR}/{file_path.relative_to(bundle_dir)}"
                        )
                        zf.write(file_path, arcname)
            _ = tmp_path.replace(zip_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        return zip_path

    def _finalize_logs(self) -> tuple[int, dict[str, int], str, str]:
        """Finalize log capture and return stats.

        Returns:
            Tuple of (record_count, level_counts, start_time, end_time).
        """
        if self._log_collector_path is None or not self._log_collector_path.exists():
            return 0, {}, "", ""

        rel_path = "logs/app.jsonl"
        if rel_path in self._files:  # pragma: no cover
            return 0, {}, "", ""

        content = self._log_collector_path.read_bytes()
        self._files.append(rel_path)
        self._checksums[rel_path] = _compute_checksum(content)
        self._sizes[rel_path] = len(content)

        return _parse_log_stats(content.decode("utf-8"))


def _parse_log_stats(content: str) -> tuple[int, dict[str, int], str, str]:
    """Parse log content and extract statistics.

    Returns:
        Tuple of (record_count, level_counts, start_time, end_time).
    """
    record_count = 0
    level_counts: dict[str, int] = {}
    start_time = ""
    end_time = ""

    for line in content.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue

        record_count += 1
        level = record.get("level", "UNKNOWN")
        level_counts[level] = level_counts.get(level, 0) + 1

        ts = record.get("timestamp", "")
        if ts:
            if not start_time or ts < start_time:
                start_time = ts
            if not end_time or ts > end_time:
                end_time = ts

    return record_count, level_counts, start_time, end_time


@dataclass(frozen=True, slots=True)
class _ArtifactBuildContext:
    """Context for building artifact metadata."""

    files: tuple[str, ...]
    sizes: Mapping[str, int]
    checksums: Mapping[str, str]
    config: BundleConfig
    fs_stats: tuple[int, int, int]  # (files_captured, files_skipped, total_bytes)
    log_stats: tuple[int, Mapping[str, int], str, str]  # (count, levels, start, end)


def _build_artifacts_map(ctx: _ArtifactBuildContext) -> dict[str, ArtifactInfo]:
    """Build the artifacts map with logical IDs."""
    artifacts: dict[str, ArtifactInfo] = {}
    record_count, level_counts, log_start, log_end = ctx.log_stats
    fs_files_captured, fs_files_skipped, fs_total_bytes = ctx.fs_stats

    # Map file paths to logical IDs and content types
    artifact_mappings: list[tuple[str, str, str, str]] = [
        ("request/input.json", "request_input", "json", "application/json"),
        ("request/output.json", "request_output", "json", "application/json"),
        ("session/before.jsonl", "session_before", "jsonl", "application/x-ndjson"),
        ("session/after.jsonl", "session_after", "jsonl", "application/x-ndjson"),
        ("logs/app.jsonl", "logs", "jsonl", "application/x-ndjson"),
        ("config.json", "config", "json", "application/json"),
        ("run_context.json", "run_context", "json", "application/json"),
        ("metrics.json", "metrics", "json", "application/json"),
        ("prompt_overrides.json", "prompt_overrides", "json", "application/json"),
        ("error.json", "error", "json", "application/json"),
        ("eval.json", "eval", "json", "application/json"),
        ("README.txt", "readme", "text", "text/plain"),
    ]

    for path, logical_id, kind, content_type in artifact_mappings:
        if path in ctx.files:
            params = _ArtifactParams(
                path=path,
                kind=kind,
                content_type=content_type,
                logical_id=logical_id,
                size_bytes=ctx.sizes.get(path, 0),
                sha256=ctx.checksums.get(path, ""),
            )
            info = _make_artifact_info(
                params,
                (record_count, level_counts, log_start, log_end),
            )
            artifacts[logical_id] = info

    # Handle filesystem directory artifact
    fs_files = [f for f in ctx.files if f.startswith("filesystem/")]
    if fs_files:
        fs_total = sum(ctx.sizes.get(f, 0) for f in fs_files)
        artifacts["filesystem"] = ArtifactInfo(
            path="filesystem/",
            kind="directory",
            content_type="",
            size_bytes=fs_total,
            sha256="",
            capture=FilesystemCapture(
                max_file_size=ctx.config.max_file_size,
                max_total_size=ctx.config.max_total_size,
            ),
            counts=FilesystemCounts(
                files_captured=fs_files_captured,
                files_skipped=fs_files_skipped,
                total_bytes_captured=fs_total_bytes,
            ),
        )

    # Handle environment directory
    env_files = [f for f in ctx.files if f.startswith("environment/")]
    if env_files:
        env_total = sum(ctx.sizes.get(f, 0) for f in env_files)
        artifacts["environment"] = ArtifactInfo(
            path="environment/",
            kind="directory",
            content_type="",
            size_bytes=env_total,
            sha256="",
        )

    return artifacts


@dataclass(frozen=True, slots=True)
class _ArtifactParams:
    """Parameters for creating an artifact."""

    path: str
    kind: str
    content_type: str
    logical_id: str
    size_bytes: int
    sha256: str


def _make_artifact_info(
    params: _ArtifactParams,
    log_stats: tuple[int, Mapping[str, int], str, str],
) -> ArtifactInfo:
    """Create an ArtifactInfo with appropriate metadata based on logical ID."""
    record_count, level_counts, log_start, log_end = log_stats

    # Add log-specific metadata
    if params.logical_id == "logs" and record_count > 0:
        return ArtifactInfo(
            path=params.path,
            kind=params.kind,
            content_type=params.content_type,
            size_bytes=params.size_bytes,
            sha256=params.sha256,
            time_range=TimeRange(start=log_start, end=log_end),
            counts=LogCounts(records=record_count, levels=level_counts),
        )

    # Add session-specific schema info
    if params.logical_id in {"session_before", "session_after"}:
        return ArtifactInfo(
            path=params.path,
            kind=params.kind,
            content_type=params.content_type,
            size_bytes=params.size_bytes,
            sha256=params.sha256,
            schema=SchemaInfo(type="Snapshot", version="1.0.0"),
        )

    return ArtifactInfo(
        path=params.path,
        kind=params.kind,
        content_type=params.content_type,
        size_bytes=params.size_bytes,
        sha256=params.sha256,
    )


class DebugBundle:
    """Load and inspect existing debug bundles.

    Example::

        bundle = DebugBundle.load("./debug/bundle.zip")
        print(bundle.manifest)
        print(bundle.metrics)
        print(bundle.session_after)
    """

    _zip_path: Path
    _manifest: BundleManifest
    _zip_file: zipfile.ZipFile | None

    def __init__(self, zip_path: Path, manifest: BundleManifest) -> None:
        """Initialize bundle from path and manifest."""
        super().__init__()
        self._zip_path = zip_path
        self._manifest = manifest
        self._zip_file = None

    @property
    def path(self) -> Path:
        """Return path to the bundle zip file."""
        return self._zip_path

    @property
    def manifest(self) -> BundleManifest:
        """Return the bundle manifest."""
        return self._manifest

    @classmethod
    def load(cls, path: Path | str) -> DebugBundle:
        """Load a debug bundle from a zip file.

        Args:
            path: Path to the bundle zip file.

        Returns:
            DebugBundle instance.

        Raises:
            BundleValidationError: If the bundle is invalid.
        """
        zip_path = Path(path)
        if not zip_path.exists():
            raise BundleValidationError(f"Bundle not found: {zip_path}")

        if not zipfile.is_zipfile(zip_path):
            raise BundleValidationError(f"Not a valid zip file: {zip_path}")

        with zipfile.ZipFile(zip_path, "r") as zf:
            manifest_path = f"{BUNDLE_ROOT_DIR}/manifest.json"
            if manifest_path not in zf.namelist():
                raise BundleValidationError("Bundle missing manifest.json")

            manifest_content = zf.read(manifest_path).decode("utf-8")
            manifest = BundleManifest.from_json(manifest_content)

        return cls(zip_path, manifest)

    def read_file(self, rel_path: str) -> bytes:
        """Read an artifact file from the bundle.

        Args:
            rel_path: Relative path within the bundle (e.g. "request/input.json").

        Returns:
            File content as bytes.

        Raises:
            BundleValidationError: If the file is not found in the bundle.
        """
        with zipfile.ZipFile(self._zip_path, "r") as zf:
            full_path = f"{BUNDLE_ROOT_DIR}/{rel_path}"
            if full_path not in zf.namelist():
                raise BundleValidationError(f"Artifact not found: {rel_path}")
            return zf.read(full_path)

    def _read_artifact(self, rel_path: str) -> bytes:
        """Read an artifact from the bundle (internal alias)."""
        return self.read_file(rel_path)

    def _read_json(self, rel_path: str) -> JSONValue:
        """Read and parse a JSON artifact."""
        content = self._read_artifact(rel_path)
        return json.loads(content.decode("utf-8"))

    @property
    def request_input(self) -> JSONValue:
        """Return the request input."""
        return self._read_json("request/input.json")

    @property
    def request_output(self) -> JSONValue:
        """Return the request output."""
        return self._read_json("request/output.json")

    @property
    def session_before(self) -> str | None:
        """Return session state before execution, or None if not captured."""
        try:
            content = self._read_artifact("session/before.jsonl")
            return content.decode("utf-8")
        except BundleValidationError:
            return None

    @property
    def session_after(self) -> str | None:
        """Return session state after execution."""
        try:
            content = self._read_artifact("session/after.jsonl")
            return content.decode("utf-8")
        except BundleValidationError:
            return None

    @property
    def logs(self) -> str | None:
        """Return log records, or None if not captured."""
        try:
            content = self._read_artifact("logs/app.jsonl")
            return content.decode("utf-8")
        except BundleValidationError:
            return None

    @property
    def config(self) -> JSONValue | None:
        """Return configuration, or None if not present."""
        try:
            return self._read_json("config.json")
        except BundleValidationError:
            return None

    @property
    def run_context(self) -> JSONValue | None:
        """Return run context, or None if not present."""
        try:
            return self._read_json("run_context.json")
        except BundleValidationError:
            return None

    @property
    def metrics(self) -> JSONValue | None:
        """Return metrics, or None if not present."""
        try:
            return self._read_json("metrics.json")
        except BundleValidationError:
            return None

    @property
    def prompt_overrides(self) -> JSONValue | None:
        """Return prompt overrides, or None if not present."""
        try:
            return self._read_json("prompt_overrides.json")
        except BundleValidationError:
            return None

    @property
    def error(self) -> JSONValue | None:
        """Return error details, or None if not present."""
        try:
            return self._read_json("error.json")
        except BundleValidationError:
            return None

    @property
    def eval(self) -> JSONValue | None:
        """Return eval metadata, or None if not present."""
        try:
            return self._read_json("eval.json")
        except BundleValidationError:
            return None

    @property
    def environment(self) -> dict[str, JSONValue | str | None] | None:
        """Return environment capture (reproducibility envelope), or None if not present.

        Returns a dict with keys: system, python, packages, env_vars, git,
        git_diff, command, container. Values are None if the specific artifact
        was not captured.
        """
        # Check if any environment files exist
        has_env = any(f.startswith("environment/") for f in self._manifest.files)
        if not has_env:
            return None

        result: dict[str, JSONValue | str | None] = {}

        # JSON files
        for key, path in [
            ("system", "environment/system.json"),
            ("python", "environment/python.json"),
            ("env_vars", "environment/env_vars.json"),
            ("git", "environment/git.json"),
            ("container", "environment/container.json"),
        ]:
            try:
                result[key] = self._read_json(path)
            except BundleValidationError:
                result[key] = None

        # Text files
        for key, path in [
            ("packages", "environment/packages.txt"),
            ("git_diff", "environment/git.diff"),
            ("command", "environment/command.txt"),
        ]:
            try:
                content = self._read_artifact(path)
                result[key] = content.decode("utf-8")
            except BundleValidationError:
                result[key] = None

        return result

    def list_files(self) -> list[str]:
        """Return list of files in the bundle."""
        with zipfile.ZipFile(self._zip_path, "r") as zf:
            prefix = f"{BUNDLE_ROOT_DIR}/"
            return [
                name[len(prefix) :]
                for name in zf.namelist()
                if name.startswith(prefix) and not name.endswith("/")
            ]

    def extract(self, target: Path | str) -> Path:
        """Extract bundle to a directory.

        Args:
            target: Target directory for extraction.

        Returns:
            Path to the extracted bundle root.
        """
        target_path = Path(target)
        target_path.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(self._zip_path, "r") as zf:
            zf.extractall(target_path)

        return target_path / BUNDLE_ROOT_DIR

    def verify_integrity(self) -> bool:
        """Verify bundle integrity using checksums.

        Returns:
            True if all checksums match, False otherwise.
        """
        for rel_path, expected in self._manifest.integrity.checksums.items():
            try:
                content = self._read_artifact(rel_path)
                actual = _compute_checksum(content)
                if actual != expected:
                    _logger.warning(
                        "Checksum mismatch",
                        extra={
                            "path": rel_path,
                            "expected": expected,
                            "actual": actual,
                        },
                    )
                    return False
            except BundleValidationError:
                _logger.warning(
                    "Missing artifact during integrity check",
                    extra={"path": rel_path},
                )
                return False
        return True


__all__ = [
    "BUNDLE_FORMAT_VERSION",
    "ArtifactInfo",
    "BundleConfig",
    "BundleError",
    "BundleManifest",
    "BundleSummary",
    "BundleValidationError",
    "BundleWriter",
    "CaptureInfo",
    "DebugBundle",
    "FilesystemCapture",
    "FilesystemCounts",
    "IndexPointer",
    "LimitsApplied",
    "LogCounts",
    "RedactionInfo",
    "SchemaInfo",
    "TimeRange",
    "TokenUsage",
]
