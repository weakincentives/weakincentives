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

"""Remote sandbox facets: thin clients of a :class:`SandboxTransport`.

``RemoteBackend`` implements the filesystem backend protocol and
``RemoteShell`` the shell facet over one transport; ``RemoteSandbox``
composes them and owns the transport's lifecycle — ``close()`` tears the
transport down. There is no streaming or text logic here: all ergonomics
live in the :class:`~weakincentives.filesystem.Filesystem` facade, and
all wire concerns live in the transport.

Transport faults are translated back into each facet's exception
contract (:func:`~weakincentives.sandbox.exception_for_fault`);
connectivity faults surface as ``RuntimeError``.
"""

from __future__ import annotations

import threading
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import PurePosixPath
from typing import override
from uuid import uuid4

from ..filesystem import (
    FileEntry,
    FileStat,
    Filesystem,
    GlobMatch,
    GrepMatch,
    SnapshotRef,
    WriteMode,
)
from ._config import EgressPolicy
from ._sandbox import CredentialBinding, SandboxClosedError, validate_bindings
from ._shell import DEFAULT_SHELL_TIMEOUT_S, CommandResult
from ._transport import SandboxTransport, TransportFault, exception_for_fault

__all__ = [
    "RemoteBackend",
    "RemoteSandbox",
    "RemoteShell",
]


@contextmanager
def _mapped_faults() -> Generator[None]:
    """Translate transport faults into the facet exception contract."""
    try:
        yield
    except TransportFault as fault:
        raise exception_for_fault(fault) from fault


class RemoteBackend:
    """Filesystem backend primitives over a sandbox transport.

    A thin client: every primitive is one transport call, including
    ``glob`` and ``grep`` (executed server-side by contract). Transport
    faults map to the backend protocol's exception contract;
    connectivity faults map to ``RuntimeError``.
    """

    def __init__(
        self,
        transport: SandboxTransport,
        *,
        read_only: bool = False,
    ) -> None:
        super().__init__()
        self._transport = transport
        self._read_only = read_only

    @property
    def root(self) -> str:
        """Remote environment root, reported by the transport."""
        return self._transport.root

    @property
    def read_only(self) -> bool:
        """True if write operations are disabled (enforced by the facade)."""
        return self._read_only

    def stat(self, path: str) -> FileStat:
        """Get metadata for a path."""
        with _mapped_faults():
            return self._transport.stat(path)

    def list(self, path: str) -> Sequence[FileEntry]:
        """List entries of an existing directory, sorted by name."""
        with _mapped_faults():
            return self._transport.list(path)

    def glob(self, pattern: str, *, path: str) -> Sequence[GlobMatch]:
        """Match files under a base directory (server-side)."""
        with _mapped_faults():
            return self._transport.glob(pattern, path=path)

    def grep(
        self,
        pattern: str,
        *,
        path: str,
        glob: str | None,
        max_matches: int,
    ) -> Sequence[GrepMatch]:
        """Regex-search file contents (server-side)."""
        with _mapped_faults():
            return self._transport.grep(
                pattern, path=path, glob=glob, max_matches=max_matches
            )

    def read_range(self, path: str, *, offset: int, length: int | None) -> bytes:
        """Read up to ``length`` bytes starting at ``offset``."""
        with _mapped_faults():
            return self._transport.read_range(path, offset=offset, length=length)

    def write(self, path: str, data: bytes, *, mode: WriteMode) -> None:
        """Write bytes to a file, creating parent directories as needed."""
        with _mapped_faults():
            self._transport.write(path, data, mode=mode)

    def delete(self, path: str, *, recursive: bool) -> None:
        """Delete a file, or a directory when empty or ``recursive=True``."""
        with _mapped_faults():
            self._transport.delete(path, recursive=recursive)

    def mkdir(self, path: str) -> None:
        """Create a directory and any missing parents (idempotent)."""
        with _mapped_faults():
            self._transport.mkdir(path)

    def rename(self, src: str, dst: str) -> None:
        """Move a file or directory (with its contents) to a new path."""
        with _mapped_faults():
            self._transport.rename(src, dst)

    def snapshot(self, *, tag: str | None = None) -> SnapshotRef:
        """Capture remote environment state as an opaque snapshot ref."""
        with _mapped_faults():
            return self._transport.snapshot(tag=tag)

    def restore(self, ref: SnapshotRef) -> None:
        """Restore remote environment state captured by :meth:`snapshot`."""
        with _mapped_faults():
            self._transport.restore(ref)


class RemoteShell:
    """Shell facet over a sandbox transport.

    Argument-shape validation (empty ``argv``, absolute ``cwd``,
    non-positive ``timeout_s``) happens client-side so the contract is
    uniform across transports; ``cwd`` existence and escape checks are
    the environment's to make and come back as transport faults. The
    constructor ``env`` is overlaid on the *remote* base environment by
    the transport's exec implementation.
    """

    def __init__(
        self,
        transport: SandboxTransport,
        *,
        env: Mapping[str, str] | None = None,
        default_timeout_s: float = DEFAULT_SHELL_TIMEOUT_S,
    ) -> None:
        super().__init__()
        if default_timeout_s <= 0:
            msg = f"default_timeout_s must be positive, got {default_timeout_s}"
            raise ValueError(msg)
        self._transport = transport
        self._env: dict[str, str] = dict(env) if env is not None else {}
        self._default_timeout_s = default_timeout_s

    @property
    def root(self) -> str:
        """Directory commands run under (the remote environment root)."""
        return self._transport.root

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        stdin: bytes | None = None,
        timeout_s: float | None = None,
    ) -> CommandResult:
        """Execute ``argv`` in the remote environment.

        See :meth:`~weakincentives.sandbox.Shell.run` for the full
        contract; launch failures keep their shell-conventional exit
        codes inside the returned :class:`CommandResult`.
        """
        if not argv:
            msg = "argv must not be empty"
            raise ValueError(msg)
        if timeout_s is not None and timeout_s <= 0:
            msg = f"timeout_s must be positive, got {timeout_s}"
            raise ValueError(msg)
        if cwd is not None and PurePosixPath(cwd).is_absolute():
            msg = f"cwd must be relative to the sandbox root, got {cwd!r}"
            raise ValueError(msg)
        overlay: dict[str, str] = dict(self._env)
        if env is not None:
            overlay.update(env)
        effective_timeout = (
            timeout_s if timeout_s is not None else self._default_timeout_s
        )
        with _mapped_faults():
            return self._transport.exec(
                list(argv),
                cwd=cwd,
                env=overlay or None,
                stdin=stdin,
                timeout_s=effective_timeout,
            )


class RemoteSandbox:
    """Sandbox whose facets live on the other side of one transport.

    Composes ``Filesystem(RemoteBackend)`` and :class:`RemoteShell` over
    the same transport; ``close()`` tears the transport down and is
    idempotent. The control plane forwards to the transport — egress and
    credential changes reach the environment's enforcement point — and
    only credential *names* are retained client-side.
    """

    def __init__(
        self,
        *,
        transport: SandboxTransport,
        filesystem: Filesystem,
        shell: RemoteShell,
        egress: EgressPolicy | None = None,
        sandbox_id: str | None = None,
    ) -> None:
        super().__init__()
        self._id = sandbox_id if sandbox_id is not None else uuid4().hex
        self._transport = transport
        self._filesystem = filesystem
        self._shell = shell
        self._egress = egress if egress is not None else EgressPolicy()
        self._credential_names: frozenset[str] = frozenset()
        self._closed = False
        self._close_lock = threading.Lock()

    @override
    def __repr__(self) -> str:
        return (
            f"RemoteSandbox(id={self._id!r}, root={self._transport.root!r}, "
            f"closed={self._closed})"
        )

    def _ensure_open(self) -> None:
        if self._closed:
            msg = f"Sandbox {self._id} is closed"
            raise SandboxClosedError(msg)

    @property
    def id(self) -> str:
        """Stable identifier for this sandbox instance."""
        return self._id

    @property
    def root(self) -> str:
        """Remote environment root; equals ``filesystem.root``."""
        return self._transport.root

    @property
    def filesystem(self) -> Filesystem:
        """Filesystem facet rooted at :attr:`root`."""
        self._ensure_open()
        return self._filesystem

    @property
    def shell(self) -> RemoteShell:
        """Command-execution facet rooted at :attr:`root`."""
        self._ensure_open()
        return self._shell

    @property
    def egress(self) -> EgressPolicy:
        """Outbound network policy currently in effect."""
        return self._egress

    @property
    def credential_names(self) -> frozenset[str]:
        """Names of currently bound credentials (never the material)."""
        return self._credential_names

    def configure_egress(self, policy: EgressPolicy) -> None:
        """Replace the egress policy live at the environment's egress sidecar."""
        self._ensure_open()
        with _mapped_faults():
            self._transport.configure_egress(policy)
        self._egress = policy

    def configure_credentials(self, bindings: Sequence[CredentialBinding]) -> None:
        """Replace the full set of credential bindings.

        Material transits the transport's control channel to the
        environment's enforcement point; only the names are retained
        here.

        Raises:
            ValueError: Two bindings share a name.
        """
        self._ensure_open()
        names = frozenset(validate_bindings(bindings))
        with _mapped_faults():
            self._transport.configure_credentials(bindings)
        self._credential_names = names

    def snapshot(self, *, tag: str | None = None) -> SnapshotRef:
        """Capture the environment state via the filesystem backend."""
        self._ensure_open()
        return self._filesystem.snapshot(tag=tag)

    def restore(self, ref: SnapshotRef) -> None:
        """Restore environment state captured by :meth:`snapshot`."""
        self._ensure_open()
        self._filesystem.restore(ref)

    def close(self) -> None:
        """Tear down the transport. Idempotent."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self._credential_names = frozenset()
        self._transport.close()
