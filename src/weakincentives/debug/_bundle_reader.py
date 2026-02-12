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
"""Debug bundle reader for loading and inspecting existing bundles."""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ..types import JSONValue
from .bundle import BundleValidationError

if TYPE_CHECKING:
    from .bundle import BundleManifest

BUNDLE_ROOT_DIR = "debug_bundle"


_logger = logging.getLogger(__name__)


class DebugBundle:  # noqa: PLR0904 - read-only property accessors, not complex logic
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
        from .bundle import BundleManifest, BundleValidationError

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
        from .bundle import BundleValidationError

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
    def transcript(self) -> list[dict[str, Any]]:
        """Return transcript entries, or empty list if not present.

        Each entry is a log record dict with ``event == "transcript.entry"``
        and a ``context`` dict containing the common transcript envelope.
        """
        try:
            content = self._read_artifact("transcript.jsonl")
        except BundleValidationError:
            return []
        text = content.decode("utf-8")
        entries: list[dict[str, Any]] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                parsed: object = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            else:
                if isinstance(parsed, dict):
                    entries.append(cast("dict[str, Any]", parsed))
        return entries

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
        from .bundle import BundleValidationError, compute_checksum

        for rel_path, expected in self._manifest.integrity.checksums.items():
            try:
                content = self._read_artifact(rel_path)
                actual = compute_checksum(content)
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
