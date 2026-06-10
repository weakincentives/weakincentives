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

"""Sandbox providers: materialize a :class:`SandboxConfig` into a sandbox.

``SandboxProvider.open`` is the factory seam between declared intent and a
live environment. :class:`LocalSandboxProvider` materializes configs on
the host: it copies mounts into a fresh temp directory (symlink and
byte-budget guards included), roots a ``Filesystem(HostBackend)`` and a
:class:`~weakincentives.sandbox.LocalShell` there, runs setup commands,
and hands ownership to the returned
:class:`~weakincentives.sandbox.LocalSandbox`. Open fails closed: any
error during materialization removes the partially-built directory.
"""

from __future__ import annotations

import shlex
import shutil
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..filesystem import Filesystem, HostBackend
from ._config import SandboxConfig
from ._mounts import materialize_mounts
from ._sandbox import LocalSandbox, Sandbox, SandboxError
from ._shell import LocalShell

__all__ = [
    "LocalSandboxProvider",
    "SandboxProvider",
    "SandboxSetupError",
]


class SandboxSetupError(SandboxError):
    """Raised when a sandbox setup command fails during open."""


_STDERR_TAIL_BYTES = 2048


@runtime_checkable
class SandboxProvider(Protocol):
    """Materializes sandbox intent into a live environment."""

    def open(self, config: SandboxConfig) -> Sandbox:
        """Open a sandbox satisfying ``config``.

        The caller owns the returned sandbox and must ``close()`` it.
        """
        ...


class LocalSandboxProvider:
    """Open sandboxes on the local host from declarative configs."""

    def __init__(self, *, temp_dir_prefix: str = "wink-sandbox-") -> None:
        super().__init__()
        self._temp_dir_prefix = temp_dir_prefix

    def open(self, config: SandboxConfig) -> LocalSandbox:
        """Materialize ``config`` into a :class:`LocalSandbox`.

        Mounts are validated against ``config.allowed_host_roots`` and
        copied with the same guards the workspace section applies. Setup
        commands run through the sandbox shell, in order, before the
        sandbox is returned.

        Raises:
            SandboxSetupError: A setup command was empty or exited
                non-zero; the partially-built sandbox is removed.
            WorkspaceSecurityError: A mount violates security constraints.
            WorkspaceBudgetExceededError: A mount exceeds its byte budget.
            FileNotFoundError: A mount's host path does not exist.
        """
        allowed_roots = tuple(Path(r) for r in config.allowed_host_roots)
        root, _previews = materialize_mounts(
            config.mounts,
            allowed_host_roots=allowed_roots,
            temp_dir_prefix=self._temp_dir_prefix,
        )
        try:
            filesystem = Filesystem(HostBackend(root, read_only=config.read_only))
            shell = LocalShell(root, env=config.env)
            for command in config.setup:
                self._run_setup_command(shell, command)
            return LocalSandbox(
                root=root,
                filesystem=filesystem,
                shell=shell,  # nosec B604 - Shell facet object, not a shell=True flag
                egress=config.egress,
            )
        except Exception:
            shutil.rmtree(root, ignore_errors=True)
            raise

    @staticmethod
    def _run_setup_command(shell: LocalShell, command: str) -> None:
        """Run one setup command; raise :class:`SandboxSetupError` on failure."""
        argv = shlex.split(command)
        if not argv:
            msg = f"Setup command is empty: {command!r}"
            raise SandboxSetupError(msg)
        result = shell.run(argv)
        if not result.ok:
            stderr_tail = result.stderr[-_STDERR_TAIL_BYTES:].decode(
                "utf-8", errors="replace"
            )
            msg = (
                f"Setup command failed with exit code {result.exit_code}: "
                f"{command!r}\n{stderr_tail}"
            )
            raise SandboxSetupError(msg)
