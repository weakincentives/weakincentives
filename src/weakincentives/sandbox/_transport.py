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

"""Sandbox transport: the remote environment's narrow waist.

``SandboxTransport`` carries every sandbox effect over one connection —
one method per filesystem primitive, one exec surface, and the
egress/credential control plane. Local and remote sandboxes differ only
in transport: the remote facets (:mod:`._remote`) are thin clients of
this protocol, and a transport implementation (loopback, SSH, container
RPC) is the only place wire concerns live.

``glob`` and ``grep`` are transport methods so they execute **server
side** — a transport must never fan a recursive directory walk out over
the wire.

Fault contract
--------------

Transport methods signal failures by raising :class:`TransportFault`
with a portable :data:`TransportFaultCode`. The remote facets translate
faults back into the protocol exception contract of the surface they
implement (:func:`exception_for_fault`); transport implementations
translate native errors into faults (:func:`fault_for_exception`).
Connectivity faults — the transport itself is broken, not the operation
— map to ``RuntimeError``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Final, Literal, Protocol, runtime_checkable

from ..errors import SnapshotRestoreError, WinkError
from ..filesystem import (
    FileEntry,
    FileStat,
    GlobMatch,
    GrepMatch,
    SnapshotRef,
    WriteMode,
)
from ._config import EgressPolicy
from ._sandbox import CredentialBinding
from ._shell import CommandResult

__all__ = [
    "TRANSPORT_FAULT_CODES",
    "SandboxTransport",
    "TransportFault",
    "TransportFaultCode",
    "exception_for_fault",
    "fault_for_exception",
]


TransportFaultCode = Literal[
    "not-found",
    "is-a-directory",
    "not-a-directory",
    "exists",
    "permission",
    "invalid",
    "io",
    "snapshot",
    "connectivity",
]
"""Portable fault codes a transport may raise.

Each code maps to exactly one exception type of the facet protocols
(see :func:`exception_for_fault`); ``connectivity`` means the transport
itself failed and maps to ``RuntimeError``.
"""

TRANSPORT_FAULT_CODES: Final[tuple[TransportFaultCode, ...]] = (
    "not-found",
    "is-a-directory",
    "not-a-directory",
    "exists",
    "permission",
    "invalid",
    "io",
    "snapshot",
    "connectivity",
)
"""All :data:`TransportFaultCode` values, for exhaustive tests."""


class TransportFault(WinkError, RuntimeError):
    """A sandbox-transport failure with a portable fault code.

    Raised by :class:`SandboxTransport` implementations; the remote
    facets translate faults into the exception contract of the surface
    they implement via :func:`exception_for_fault`.
    """

    def __init__(self, code: TransportFaultCode, message: str) -> None:
        super().__init__(message)
        self.code: TransportFaultCode = code
        self.message = message


_EXCEPTION_FOR_CODE: Final[Mapping[TransportFaultCode, Callable[[str], Exception]]] = {
    "not-found": FileNotFoundError,
    "is-a-directory": IsADirectoryError,
    "not-a-directory": NotADirectoryError,
    "exists": FileExistsError,
    "permission": PermissionError,
    "invalid": ValueError,
    "io": OSError,
    "snapshot": SnapshotRestoreError,
    "connectivity": RuntimeError,
}

_CODE_FOR_EXCEPTION: Final[
    tuple[tuple[type[BaseException], TransportFaultCode], ...]
] = (
    # Most specific first: every entry except the trailing OSError is an
    # OSError subclass (SnapshotRestoreError excepted).
    (FileNotFoundError, "not-found"),
    (IsADirectoryError, "is-a-directory"),
    (NotADirectoryError, "not-a-directory"),
    (FileExistsError, "exists"),
    (PermissionError, "permission"),
    (ConnectionError, "connectivity"),
    (TimeoutError, "connectivity"),
    (SnapshotRestoreError, "snapshot"),
    (OSError, "io"),
)


def exception_for_fault(fault: TransportFault) -> Exception:
    """Map a transport fault to the facet protocols' exception contract.

    Connectivity faults map to ``RuntimeError``: the transport is
    broken, not the operation.
    """
    return _EXCEPTION_FOR_CODE[fault.code](fault.message)


def fault_for_exception(error: BaseException) -> TransportFault:
    """Map a native exception to a transport fault (server side).

    The inverse of :func:`exception_for_fault`, used where a transport
    executes primitives against concrete backends and must report
    failures portably. Existing faults pass through unchanged;
    connection and timeout errors are connectivity faults; any other
    ``OSError`` is an ``io`` fault and any other exception ``invalid``.
    """
    if isinstance(error, TransportFault):
        return error
    message = str(error)
    for exception_type, code in _CODE_FOR_EXCEPTION:
        if isinstance(error, exception_type):
            return TransportFault(code, message)
    return TransportFault("invalid", message)


@runtime_checkable
class SandboxTransport(Protocol):
    """One connection carrying every sandbox effect to the environment.

    Filesystem methods mirror
    :class:`~weakincentives.filesystem.FilesystemBackend` (paths are
    facade-normalized relative strings; ``""`` is the root); ``exec``
    mirrors :meth:`~weakincentives.sandbox.Shell.run` with a concrete
    timeout; ``configure_egress``/``configure_credentials`` reach the
    environment's enforcement point (the egress proxy, where one
    exists). All methods raise :class:`TransportFault` on failure.

    ``close`` tears down the connection and whatever environment
    ownership it carries; it is idempotent, and every other method
    raises a ``connectivity`` fault afterwards.
    """

    @property
    def root(self) -> str:
        """Absolute environment root on the remote side."""
        ...

    def stat(self, path: str) -> FileStat:
        """Get metadata for a path."""
        ...

    def list(self, path: str) -> Sequence[FileEntry]:
        """List entries of an existing directory, sorted by name."""
        ...

    def read_range(self, path: str, *, offset: int, length: int | None) -> bytes:
        """Read up to ``length`` bytes starting at ``offset``."""
        ...

    def write(self, path: str, data: bytes, *, mode: WriteMode) -> None:
        """Write bytes to a file, creating parent directories as needed."""
        ...

    def glob(self, pattern: str, *, path: str) -> Sequence[GlobMatch]:
        """Match files under a base directory, executed server-side."""
        ...

    def grep(
        self,
        pattern: str,
        *,
        path: str,
        glob: str | None,
        max_matches: int,
    ) -> Sequence[GrepMatch]:
        """Regex-search file contents, executed server-side."""
        ...

    def delete(self, path: str, *, recursive: bool) -> None:
        """Delete a file, or a directory when empty or ``recursive=True``."""
        ...

    def mkdir(self, path: str) -> None:
        """Create a directory and any missing parents (idempotent)."""
        ...

    def rename(self, src: str, dst: str) -> None:
        """Move a file or directory (with its contents) from ``src`` to ``dst``."""
        ...

    def snapshot(self, *, tag: str | None) -> SnapshotRef:
        """Capture environment state as an opaque snapshot ref."""
        ...

    def restore(self, ref: SnapshotRef) -> None:
        """Restore environment state captured by :meth:`snapshot`."""
        ...

    def exec(
        self,
        argv: Sequence[str],
        *,
        cwd: str | None,
        env: Mapping[str, str] | None,
        stdin: bytes | None,
        timeout_s: float,
    ) -> CommandResult:
        """Execute ``argv`` in the environment and return its outcome."""
        ...

    def configure_egress(self, policy: EgressPolicy) -> None:
        """Push an egress policy to the environment's egress sidecar.

        The enforcement point is a sidecar process the *environment* owns —
        not something WINK runs; this call carries the policy to it over the
        control plane. A local environment with no sidecar records the policy
        without enforcing it.
        """
        ...

    def configure_credentials(self, bindings: Sequence[CredentialBinding]) -> None:
        """Push credential material to the environment's egress sidecar.

        Secret material lives only in the sidecar (never in the sandbox env
        or filesystem); the transport carries it over the control plane and
        retains only the names client-side.
        """
        ...

    def close(self) -> None:
        """Tear down the connection. Idempotent."""
        ...
