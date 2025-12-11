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

"""Temporary workspace management for Claude Agent SDK sessions."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path


class ClaudeAgentWorkspace:
    """Stage host files into a temporary directory for SDK use."""

    def __init__(self, *, mounts: Sequence[str] | None = None) -> None:  # pyright: ignore[reportMissingSuperCall]
        self._tempdir = tempfile.TemporaryDirectory(prefix="wink-claude-sdk-")
        self.root = Path(self._tempdir.name)
        for mount in mounts or ():
            self._stage(Path(mount))

    def _stage(self, mount: Path) -> None:
        destination = self.root / mount.name
        if mount.is_dir():
            _ = shutil.copytree(mount, destination)
        elif mount.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            _ = shutil.copy2(mount, destination)

    def cleanup(self) -> None:
        """Remove the staged workspace."""

        self._tempdir.cleanup()

    def __enter__(self) -> ClaudeAgentWorkspace:
        return self

    def __exit__(self, *exc: object) -> None:  # pragma: no cover - trivial
        self.cleanup()
