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

from __future__ import annotations

import json
import re

# Bandit false positive: git invocation uses explicit arguments for root
# discovery.
import subprocess  # nosec B404
import tempfile
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any

from .versioning import PromptOverridesError

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

_IDENTIFIER_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,63}$"


class OverrideFilesystem:
    """Handle filesystem interactions for prompt overrides."""

    def __init__(
        self,
        *,
        explicit_root: Path | None,
        overrides_relative_path: Path,
    ) -> None:
        super().__init__()
        self._explicit_root = explicit_root
        self._overrides_relative_path = overrides_relative_path
        self._root_lock = RLock()
        self._root: Path | None = None
        self._path_locks: dict[Path, RLock] = {}
        self._path_locks_lock = RLock()

    def resolve_root(self) -> Path:
        """Resolve the repository root for overrides operations."""

        with self._root_lock:
            if self._root is not None:
                return self._root
            if self._explicit_root is not None:
                self._root = self._explicit_root
                return self._root

            git_root = self._git_toplevel()
            if git_root is not None:
                self._root = git_root
                return self._root

            traversal_root = self._walk_to_git_root()
            if traversal_root is None:
                raise PromptOverridesError(
                    "Failed to locate repository root. Provide root_path explicitly."
                )
            self._root = traversal_root
            return self._root

    def overrides_dir(self) -> Path:
        return self.resolve_root() / self._overrides_relative_path

    def override_file_path(
        self,
        *,
        ns: str,
        prompt_key: str,
        tag: str,
    ) -> Path:
        segments = self._split_namespace(ns)
        prompt_component = self.validate_identifier(prompt_key, "prompt key")
        tag_component = self.validate_identifier(tag, "tag")
        directory = self.overrides_dir().joinpath(*segments, prompt_component)
        return directory / f"{tag_component}.json"

    @staticmethod
    def validate_identifier(value: str, label: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise PromptOverridesError(
                f"{label.capitalize()} must be a non-empty string."
            )
        pattern = _IDENTIFIER_PATTERN
        if not re.fullmatch(pattern, stripped):
            raise PromptOverridesError(
                f"{label.capitalize()} must match pattern {pattern}."
            )
        return stripped

    @contextmanager
    def locked_override_path(self, file_path: Path) -> Iterator[None]:
        lock = self._get_override_path_lock(file_path)
        with lock:
            yield

    @staticmethod
    def atomic_write(file_path: Path, payload: Mapping[str, Any]) -> None:
        directory = file_path.parent
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", dir=directory, delete=False, encoding="utf-8"
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            _ = handle.write("\n")
            temp_name = Path(handle.name)
        _ = Path(temp_name).replace(file_path)

    def _get_override_path_lock(self, file_path: Path) -> RLock:
        with self._path_locks_lock:
            lock = self._path_locks.get(file_path)
            if lock is None:
                lock = RLock()
                self._path_locks[file_path] = lock
            return lock

    def _split_namespace(self, ns: str) -> tuple[str, ...]:
        stripped = ns.strip()
        if not stripped:
            raise PromptOverridesError("Namespace must be a non-empty string.")
        segments = tuple(part.strip() for part in stripped.split("/") if part.strip())
        if not segments:
            raise PromptOverridesError("Namespace must contain at least one segment.")
        return tuple(
            self.validate_identifier(segment, "namespace segment")
            for segment in segments
        )

    @staticmethod
    def _git_toplevel() -> Path | None:
        try:
            # Bandit false positive: git invocation uses explicit arguments.
            result = subprocess.run(  # nosec B603 B607
                ["git", "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        path = result.stdout.strip()
        if not path:
            return None
        return Path(path).resolve()

    @staticmethod
    def _walk_to_git_root() -> Path | None:
        current = Path.cwd().resolve()
        for candidate in (current, *current.parents):
            git_dir = candidate / ".git"
            if git_dir.exists():
                return candidate
        return None


__all__ = ["OverrideFilesystem"]
