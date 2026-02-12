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
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import field
from datetime import UTC, datetime
from pathlib import Path
from typing import (
    IO,
    TYPE_CHECKING,
    Protocol,
    override,
    runtime_checkable,
)

from ..dataclasses import FrozenDataclass
from ..errors import WinkError
from ..serde import dump
from ..types import JSONValue

if TYPE_CHECKING:
    from ._bundle_reader import DebugBundle as DebugBundle
    from ._bundle_writer import BundleWriter as BundleWriter

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


def generate_readme(manifest: BundleManifest) -> str:
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


def compute_checksum(content: bytes) -> str:
    """Compute SHA-256 checksum of content."""
    return hashlib.sha256(content).hexdigest()


def __getattr__(name: str) -> object:
    """Lazy re-exports to avoid circular imports."""
    if name == "DebugBundle":
        from ._bundle_reader import DebugBundle

        return DebugBundle
    if name == "BundleWriter":
        from ._bundle_writer import BundleWriter

        return BundleWriter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BUNDLE_FORMAT_VERSION",
    "BUNDLE_ROOT_DIR",
    "BuildInfo",
    "BundleConfig",
    "BundleError",
    "BundleManifest",
    "BundleRetentionPolicy",
    "BundleStorageHandler",
    "BundleValidationError",
    "BundleWriter",
    "CaptureInfo",
    "DebugBundle",
    "IntegrityInfo",
    "PromptInfo",
    "RequestInfo",
    "collect_all_logs",
    "compute_checksum",
    "generate_readme",
]
