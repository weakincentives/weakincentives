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

"""Sandbox providers: materialize a :class:`WorkspaceConfig` into a sandbox.

``SandboxProvider.open`` is the factory seam between declared intent and a
live environment. :class:`LocalSandboxProvider` materializes configs on
the host: it copies mounts into a fresh temp directory (symlink and
byte-budget guards included), roots a ``Filesystem(HostBackend)`` and a
:class:`~weakincentives.sandbox.LocalShell` there, runs setup commands,
and hands ownership to the returned
:class:`~weakincentives.sandbox.LocalSandbox`.
:class:`RemoteSandboxProvider` materializes the same intent through one
:class:`~weakincentives.sandbox.SandboxTransport`: mounts stage locally
under the same guards and upload over the transport, the egress policy
seeds the environment's enforcement point, and setup commands run
through the remote shell. Both providers fail closed: any error during
materialization tears down the partially-built environment.
"""

from __future__ import annotations

import os
import shlex
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..filesystem import Filesystem, HostBackend
from ._config import WorkspaceConfig
from ._mounts import materialize_mounts
from ._remote import RemoteBackend, RemoteSandbox, RemoteShell
from ._sandbox import LocalSandbox, Sandbox, SandboxError
from ._shell import LocalShell, Shell
from ._transport import SandboxTransport

__all__ = [
    "LocalSandboxProvider",
    "RemoteSandboxProvider",
    "SandboxProvider",
    "SandboxSetupError",
]


class SandboxSetupError(SandboxError):
    """Raised when a sandbox setup command fails during open."""


_STDERR_TAIL_BYTES = 2048


@runtime_checkable
class SandboxProvider(Protocol):
    """Materializes sandbox intent into a live environment."""

    def open(self, config: WorkspaceConfig) -> Sandbox:
        """Open a sandbox satisfying ``config``.

        The caller owns the returned sandbox and must ``close()`` it.
        """
        ...


def _run_setup_command(shell: Shell, command: str) -> None:
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


class LocalSandboxProvider:
    """Open sandboxes on the local host from declarative configs."""

    def __init__(self, *, temp_dir_prefix: str = "wink-sandbox-") -> None:
        super().__init__()
        self._temp_dir_prefix = temp_dir_prefix

    def open(self, config: WorkspaceConfig) -> LocalSandbox:
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
                _run_setup_command(shell, command)
            return LocalSandbox(
                root=root,
                filesystem=filesystem,
                shell=shell,  # nosec B604 - Shell facet object, not a shell=True flag
                egress=config.egress,
            )
        except Exception:
            shutil.rmtree(root, ignore_errors=True)
            raise


def _upload_tree(transport: SandboxTransport, staging_root: Path) -> None:
    """Replicate a staged mount tree into the environment over the waist.

    Directories are created (including empty ones) and file bytes are
    written through the transport's primitives — no side channel.
    """
    for dirpath, dirnames, filenames in os.walk(staging_root):
        dirnames.sort()
        directory = Path(dirpath)
        relative = directory.relative_to(staging_root)
        if relative != Path():
            transport.mkdir(relative.as_posix())
        for name in sorted(filenames):
            data = (directory / name).read_bytes()
            transport.write((relative / name).as_posix(), data, mode="overwrite")


class RemoteSandboxProvider:
    """Open sandboxes in a remote environment behind one transport.

    ``connect`` yields a fresh :class:`SandboxTransport` per open — the
    one connection the returned sandbox's facets share and ``close()``
    tears down. Mounts stage locally (same allowed-root, symlink, and
    byte-budget guards as the local provider) and upload through the
    transport; the staging directory is always removed.
    """

    def __init__(
        self,
        connect: Callable[[], SandboxTransport],
        *,
        temp_dir_prefix: str = "wink-sandbox-staging-",
    ) -> None:
        super().__init__()
        self._connect = connect
        self._temp_dir_prefix = temp_dir_prefix

    def open(self, config: WorkspaceConfig) -> RemoteSandbox:
        """Materialize ``config`` into a :class:`RemoteSandbox`.

        Fails closed: any error after the transport connects closes the
        transport before propagating.

        Raises:
            SandboxSetupError: A setup command was empty or exited
                non-zero.
            TransportFault: The transport failed while materializing.
            WorkspaceSecurityError: A mount violates security constraints.
            WorkspaceBudgetExceededError: A mount exceeds its byte budget.
            FileNotFoundError: A mount's host path does not exist.
        """
        allowed_roots = tuple(Path(r) for r in config.allowed_host_roots)
        staging_root, _previews = materialize_mounts(
            config.mounts,
            allowed_host_roots=allowed_roots,
            temp_dir_prefix=self._temp_dir_prefix,
        )
        try:
            transport = self._connect()
        except Exception:
            shutil.rmtree(staging_root, ignore_errors=True)
            raise
        try:
            _upload_tree(transport, Path(staging_root))
            transport.configure_egress(config.egress)
            filesystem = Filesystem(
                RemoteBackend(transport, read_only=config.read_only)
            )
            shell = RemoteShell(transport, env=config.env)
            for command in config.setup:
                _run_setup_command(shell, command)
            return RemoteSandbox(
                transport=transport,
                filesystem=filesystem,
                shell=shell,  # nosec B604 - Shell facet object, not a shell=True flag
                egress=config.egress,
            )
        except Exception:
            transport.close()
            raise
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)
