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

"""Debug bundle for capturing AgentLoop execution state.

A debug bundle is a self-contained zip archive capturing everything needed to
understand, reproduce, and debug a AgentLoop execution. Bundles unify session
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
from dataclasses import field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Protocol, Self, override, runtime_checkable
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

BUNDLE_FORMAT_VERSION = "1.0.0"
BUNDLE_ROOT_DIR = "debug_bundle"


class BundleError(WinkError, RuntimeError):
    """Base error for bundle operations."""


class BundleValidationError(BundleError):
    """Raised when bundle validation fails."""


@FrozenDataclass()
class BundleRetentionPolicy:
    """Policy for cleaning up old debug bundles in the target directory.

    Retention is applied after each bundle is successfully created. If multiple
    limits are configured, all are enforced (most restrictive wins). Bundle age
    is determined from the ``created_at`` field in the manifest.

    Attributes:
        max_bundles: Keep at most N bundles (oldest deleted first).
        max_age_seconds: Delete bundles older than this (in seconds).
        max_total_bytes: Keep total size under limit (oldest deleted first).

    Example::

        policy = BundleRetentionPolicy(
            max_bundles=10,           # Keep last 10 bundles
            max_age_seconds=86400,    # Delete bundles older than 24 hours
        )
    """

    max_bundles: int | None = None
    max_age_seconds: int | None = None
    max_total_bytes: int | None = None


@runtime_checkable
class BundleStorageHandler(Protocol):
    """Protocol for copying bundles to external storage after creation.

    The storage handler is called after retention policy is applied, so only
    bundles that survive cleanup are passed to the handler.

    Errors are logged but do not propagate (non-blocking). The local bundle
    remains regardless of storage success.

    Example::

        @dataclass
        class S3StorageHandler:
            bucket: str
            prefix: str = "debug-bundles/"

            def store_bundle(
                self, bundle_path: Path, manifest: BundleManifest
            ) -> None:
                key = f"{self.prefix}{manifest.bundle_id}.zip"
                s3_client.upload_file(str(bundle_path), self.bucket, key)
    """

    def store_bundle(
        self,
        bundle_path: Path,
        manifest: BundleManifest,
    ) -> None:
        """Copy/upload the bundle to external storage.

        Args:
            bundle_path: Local path to the created bundle ZIP.
            manifest: Bundle metadata (id, timestamp, checksums, etc.).
        """
        ...


@FrozenDataclass()
class BundleConfig:
    """Configuration for bundle creation.

    Attributes:
        target: Output directory for bundles. None disables bundling.
        max_file_size: Skip files larger than this (bytes). Default 10MB.
        max_total_size: Maximum filesystem capture size (bytes). Default 50MB.
        compression: Zip compression method.
        retention: Policy for cleaning up old bundles. None disables cleanup.
        storage_handler: Handler for copying bundles to external storage.
    """

    target: Path | None = None
    max_file_size: int = 10_000_000  # 10MB
    max_total_size: int = 52_428_800  # 50MB
    compression: str = "deflate"
    retention: BundleRetentionPolicy | None = None
    storage_handler: BundleStorageHandler | None = None

    @classmethod
    def __pre_init__(  # noqa: PLR0913
        cls,
        *,
        target: Path | str | None = None,
        max_file_size: int = 10_000_000,
        max_total_size: int = 52_428_800,
        compression: str = "deflate",
        retention: BundleRetentionPolicy | None = None,
        storage_handler: BundleStorageHandler | None = None,
    ) -> Mapping[str, object]:
        """Normalize inputs before construction."""
        normalized_target = Path(target) if isinstance(target, str) else target
        return {
            "target": normalized_target,
            "max_file_size": max_file_size,
            "max_total_size": max_total_size,
            "compression": compression,
            "retention": retention,
            "storage_handler": storage_handler,
        }

    @property
    def enabled(self) -> bool:
        """Return True if bundling is enabled (target is set)."""
        return self.target is not None


@FrozenDataclass()
class CaptureInfo:
    """Capture metadata for manifest."""

    mode: str
    trigger: str
    limits_applied: Mapping[str, bool] = field(
        default_factory=lambda: {"filesystem_truncated": False}
    )


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
class BundleManifest:
    """Bundle manifest containing metadata and integrity checksums.

    Schema::

        {
          "format_version": "1.0.0",
          "bundle_id": "uuid",
          "created_at": "2024-01-15T10:30:00+00:00",
          "request": { ... },
          "capture": { ... },
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
        "  input.json        AgentLoop request",
        "  output.json       AgentLoop response",
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
        "config.json         AgentLoop and adapter configuration",
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
    _request_id: UUID | None
    _session_id: UUID | None
    _status: str
    _prompt_info: PromptInfo
    _trigger: str
    _limits_applied: dict[str, bool]
    _finalized: bool
    _path: Path | None
    _log_collector_path: Path | None

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
        self._request_id = None
        self._session_id = None
        self._status = "success"
        self._prompt_info = PromptInfo()
        self._trigger = trigger
        self._limits_applied = {"filesystem_truncated": False}
        self._finalized = False
        self._path = None
        self._log_collector_path = None

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

    def write_request_input(self, request: object) -> None:
        """Write the AgentLoop request input."""
        try:
            content = json.dumps(_serialize_object(request), indent=2)
            self._write_artifact("request/input.json", content)
        except Exception:
            _logger.exception(
                "Failed to write request input",
                extra={"bundle_id": str(self._bundle_id)},
            )

    def write_request_output(self, response: object) -> None:
        """Write the AgentLoop response output."""
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
        """Write AgentLoop and adapter configuration."""
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

    def write_metadata(self, name: str, data: Mapping[str, Any]) -> None:
        """Write arbitrary metadata to the bundle.

        Creates a JSON file with the given name containing the provided data.
        This is a generic mechanism for adding domain-specific metadata to
        bundles without coupling the bundle layer to those domains.

        Args:
            name: The metadata type name (e.g., "eval", "metrics"). Will be
                written to {name}.json in the bundle.
            data: The metadata to serialize as JSON.
        """
        try:
            content = json.dumps(data, indent=2)
            self._write_artifact(f"{name}.json", content)
        except Exception:
            _logger.exception(
                "Failed to write metadata",
                extra={"bundle_id": str(self._bundle_id), "metadata_name": name},
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
        if self._temp_dir is None:  # pragma: no cover
            return

        fs_dir = self._temp_dir / BUNDLE_ROOT_DIR / "filesystem"
        fs_dir.mkdir(parents=True, exist_ok=True)

        total_size = 0
        truncated = False

        files = self._collect_files(fs, root_path)
        for file_path in files:
            if total_size >= self._config.max_total_size:
                truncated = True
                break

            try:
                stat = fs.stat(file_path)
                if stat.size_bytes > self._config.max_file_size:
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

                total_size += len(content)

            except (FileNotFoundError, PermissionError, IsADirectoryError):
                continue

        self._limits_applied["filesystem_truncated"] = truncated

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

    def _apply_retention_policy(self) -> None:
        """Apply retention policy to clean up old bundles.

        Called after a new bundle is successfully created. Scans the target
        directory for existing bundles and deletes those that exceed the
        configured limits.
        """
        retention = self._config.retention
        if retention is None:
            return

        try:
            self._enforce_retention(retention)
        except Exception:
            _logger.warning(
                "Failed to apply retention policy",
                extra={"bundle_id": str(self._bundle_id)},
                exc_info=True,
            )

    def _enforce_retention(self, retention: BundleRetentionPolicy) -> None:
        """Enforce retention policy on bundles in target directory."""
        bundles = self._collect_existing_bundles()
        to_delete: set[Path] = set()

        self._apply_age_limit(retention, bundles, to_delete)
        self._apply_count_limit(retention, bundles, to_delete)
        self._apply_size_limit(retention, bundles, to_delete)
        self._delete_marked_bundles(to_delete)

    def _get_retention_search_root(self) -> Path:
        """Get the root directory for retention policy bundle search.

        Returns config.target if set (for EvalLoop's nested structure),
        otherwise falls back to the writer's target directory.
        """
        if self._config.target is not None:
            return self._config.target
        return self._target

    def _collect_existing_bundles(self) -> list[tuple[Path, datetime, int]]:
        """Collect metadata for existing bundles in the target directory.

        Uses recursive glob to find bundles in nested directories, supporting
        EvalLoop's ``{target}/{request_id}/{bundle}.zip`` structure.
        """
        search_root = self._get_retention_search_root()
        bundles: list[tuple[Path, datetime, int]] = []
        # Resolve path once for efficient comparison
        self_path_resolved = self._path.resolve() if self._path else None

        for bundle_path in search_root.glob("**/*.zip"):
            try:
                # Skip the bundle we just created - check path first for efficiency
                # (avoids loading the just-created bundle unnecessarily)
                if (
                    self_path_resolved is not None
                    and bundle_path.resolve() == self_path_resolved
                ):
                    continue

                bundle = DebugBundle.load(bundle_path)
                created_at = datetime.fromisoformat(bundle.manifest.created_at)
                # Normalize to UTC for consistent sorting across timezone-naive
                # and timezone-aware timestamps
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=UTC)
                size = bundle_path.stat().st_size
                bundles.append((bundle_path, created_at, size))
            except (BundleValidationError, ValueError, OSError):
                continue
        bundles.sort(key=lambda x: x[1])
        return bundles

    @staticmethod
    def _apply_age_limit(
        retention: BundleRetentionPolicy,
        bundles: list[tuple[Path, datetime, int]],
        to_delete: set[Path],
    ) -> None:
        """Mark bundles older than max_age_seconds for deletion."""
        if retention.max_age_seconds is None:
            return
        now = datetime.now(UTC)
        for bundle_path, created_at, _ in bundles:
            # Timestamps are already normalized to UTC in _collect_existing_bundles
            if (now - created_at).total_seconds() > retention.max_age_seconds:
                to_delete.add(bundle_path)

    @staticmethod
    def _apply_count_limit(
        retention: BundleRetentionPolicy,
        bundles: list[tuple[Path, datetime, int]],
        to_delete: set[Path],
    ) -> None:
        """Mark oldest bundles for deletion to stay under max_bundles limit."""
        if retention.max_bundles is None:
            return
        total_count = len(bundles) + 1  # +1 for newly created bundle
        excess = total_count - retention.max_bundles
        if excess > 0:
            for bundle_path, _, _ in bundles[:excess]:
                to_delete.add(bundle_path)

    def _apply_size_limit(
        self,
        retention: BundleRetentionPolicy,
        bundles: list[tuple[Path, datetime, int]],
        to_delete: set[Path],
    ) -> None:
        """Mark oldest bundles for deletion to stay under max_total_bytes limit."""
        if retention.max_total_bytes is None:
            return

        new_bundle_size = self._path.stat().st_size if self._path else 0

        # Calculate total size including new bundle and all existing bundles not yet marked
        total_size = new_bundle_size + sum(
            size for path, _, size in bundles if path not in to_delete
        )

        # Delete oldest bundles until under limit (bundles already sorted oldest-first)
        for bundle_path, _, size in bundles:
            if bundle_path in to_delete:
                continue
            if total_size <= retention.max_total_bytes:
                break
            to_delete.add(bundle_path)
            total_size -= size

    @staticmethod
    def _delete_marked_bundles(to_delete: set[Path]) -> None:
        """Delete bundles marked for removal."""
        for bundle_path in to_delete:
            try:
                bundle_path.unlink()
                _logger.debug(
                    "Deleted old bundle",
                    extra={"bundle_path": str(bundle_path)},
                )
            except OSError:
                _logger.warning(
                    "Failed to delete old bundle",
                    extra={"bundle_path": str(bundle_path)},
                    exc_info=True,
                )

    def _invoke_storage_handler(self, manifest: BundleManifest) -> None:
        """Invoke storage handler to copy bundle to external storage.

        Called after retention policy is applied. Errors are logged but
        do not propagate.
        """
        handler = self._config.storage_handler
        if handler is None or self._path is None:
            return

        try:
            handler.store_bundle(self._path, manifest)
            _logger.debug(
                "Bundle stored to external storage",
                extra={
                    "bundle_id": str(self._bundle_id),
                    "bundle_path": str(self._path),
                },
            )
        except Exception:
            _logger.warning(
                "Failed to store bundle to external storage",
                extra={
                    "bundle_id": str(self._bundle_id),
                    "bundle_path": str(self._path),
                },
                exc_info=True,
            )

    def _finalize(self) -> None:
        """Finalize bundle: generate README, compute manifest, create zip."""
        if self._finalized or self._temp_dir is None:
            return

        self._ended_at = datetime.now(UTC)

        # Add logs to file list if captured
        if self._log_collector_path is not None and self._log_collector_path.exists():
            rel_path = "logs/app.jsonl"
            if rel_path not in self._files:  # pragma: no branch
                content = self._log_collector_path.read_bytes()
                self._files.append(rel_path)
                self._checksums[rel_path] = _compute_checksum(content)

        # Build manifest
        manifest = BundleManifest(
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
            prompt=self._prompt_info,
            files=tuple(sorted(self._files)),
            integrity=IntegrityInfo(
                algorithm="sha256",
                checksums=self._checksums,
            ),
            build=BuildInfo(),
        )

        # Write README
        readme_content = _generate_readme(manifest)
        self._write_artifact("README.txt", readme_content)

        # Update manifest with README (use replace since files list changed)
        from dataclasses import replace as dc_replace

        manifest = dc_replace(
            manifest,
            files=tuple(sorted(self._files)),
            integrity=IntegrityInfo(
                algorithm="sha256",
                checksums=self._checksums,
            ),
        )

        # Write manifest (excluded from checksums to avoid circular dependency)
        manifest_content = manifest.to_json()
        manifest_path = self._temp_dir / BUNDLE_ROOT_DIR / "manifest.json"
        _ = manifest_path.write_text(manifest_content, encoding="utf-8")
        self._files.append("manifest.json")

        # Create zip archive atomically (write to temp, then rename)
        self._target.mkdir(parents=True, exist_ok=True)
        timestamp = self._started_at.strftime("%Y%m%d_%H%M%S")
        zip_name = f"{self._bundle_id}_{timestamp}.zip"
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
                            BUNDLE_ROOT_DIR
                            + "/"
                            + str(file_path.relative_to(bundle_dir))
                        )
                        zf.write(file_path, arcname)
            # Atomic rename: either fully created or not present
            _ = tmp_path.replace(zip_path)
        finally:
            # Clean up temp file if it still exists (e.g., rename failed)
            if tmp_path.exists():
                tmp_path.unlink()

        self._path = zip_path
        self._finalized = True

        _logger.info(
            "Debug bundle created",
            extra={
                "bundle_id": str(self._bundle_id),
                "bundle_path": str(zip_path),
                "file_count": len(self._files),
            },
        )

        # Post-creation lifecycle: retention then storage
        self._apply_retention_policy()
        self._invoke_storage_handler(manifest)


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
    "BundleConfig",
    "BundleError",
    "BundleManifest",
    "BundleRetentionPolicy",
    "BundleStorageHandler",
    "BundleValidationError",
    "BundleWriter",
    "DebugBundle",
]
