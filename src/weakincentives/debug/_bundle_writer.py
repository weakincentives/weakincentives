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

# pyright: reportImportCycles=false
"""Bundle writer for streaming debug bundle creation."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import zipfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self
from uuid import UUID, uuid4

from ..clock import SYSTEM_CLOCK
from ..serde import dump
from ..types import JSONValue
from ._bundle_retention import apply_retention_policy, invoke_storage_handler
from .bundle import (
    BUNDLE_FORMAT_VERSION,
    BUNDLE_ROOT_DIR,
    BuildInfo,
    BundleConfig,
    BundleError,
    BundleManifest,
    CaptureInfo,
    IntegrityInfo,
    PromptInfo,
    RequestInfo,
    collect_all_logs,
    compute_checksum,
    generate_readme,
)

if TYPE_CHECKING:
    from ..filesystem import Filesystem
    from ..runtime.run_context import RunContext
    from ..runtime.session import Session
    from .environment import EnvironmentCapture

_logger = logging.getLogger(__name__)


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


def _resolve_build_info() -> BuildInfo:
    """Resolve build metadata best-effort.

    Returns package version from ``importlib.metadata`` and the git
    commit SHA of the *weakincentives* package source tree (not the
    process cwd, which may be a different repository).  Falls back to
    empty strings on any failure.
    """
    import subprocess  # nosec B404 - needed for git introspection

    version = ""
    commit = ""
    try:
        from importlib.metadata import version as pkg_version

        version = pkg_version("weakincentives")
    except Exception:  # pragma: no cover
        pass  # nosec B110 - best-effort, non-critical metadata
    try:
        # Resolve from the package's own repo, not the process cwd.
        pkg_dir = str(Path(__file__).resolve().parent)
        result = subprocess.run(  # nosec B603 B607 - trusted git command
            ["git", "-C", pkg_dir, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            commit = result.stdout.strip()
    except Exception:  # pragma: no cover
        pass  # nosec B110 - best-effort, non-critical metadata
    return BuildInfo(version=version, commit=commit)


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
        self._started_at = SYSTEM_CLOCK.utcnow()
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
        if rel_path not in self._checksums:
            self._files.append(rel_path)
        self._checksums[rel_path] = compute_checksum(content_bytes)

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
                self._checksums[bundle_rel] = compute_checksum(content)

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

    def _extract_transcript(self, log_content: bytes) -> None:
        """Extract transcript entries from captured logs into transcript.jsonl.

        Scans the app.jsonl content for records with ``event == "transcript.entry"``
        and writes them to a separate ``transcript.jsonl`` artifact.

        Args:
            log_content: Raw bytes of the app.jsonl log file.
        """
        transcript_lines: list[str] = []
        for raw_line in log_content.decode("utf-8", errors="replace").splitlines():
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
                if record.get("event") == "transcript.entry":
                    transcript_lines.append(raw_line)
            except (json.JSONDecodeError, TypeError):
                continue
        if transcript_lines:
            transcript_text = "\n".join(transcript_lines) + "\n"
            self._write_artifact("transcript.jsonl", transcript_text)

    def _finalize(self) -> None:
        """Finalize bundle: generate README, compute manifest, create zip."""
        if self._finalized or self._temp_dir is None:
            return

        self._ended_at = SYSTEM_CLOCK.utcnow()

        # Add logs to file list if captured
        if self._log_collector_path is not None and self._log_collector_path.exists():
            rel_path = "logs/app.jsonl"
            if rel_path not in self._files:  # pragma: no branch
                content = self._log_collector_path.read_bytes()
                self._files.append(rel_path)
                self._checksums[rel_path] = compute_checksum(content)

                # Extract transcript entries into a separate transcript.jsonl
                self._extract_transcript(content)

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
            build=_resolve_build_info(),
        )

        # Write README
        readme_content = generate_readme(manifest)
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
        apply_retention_policy(
            config=self._config,
            target=self._target,
            bundle_id=self._bundle_id,
            exclude_path=zip_path,
            bundle_path=self._path,
        )
        invoke_storage_handler(
            config=self._config,
            bundle_id=self._bundle_id,
            bundle_path=self._path,
            manifest=manifest,
        )
