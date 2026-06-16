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

"""Loopback sandbox transport: the remote stack with no wire.

``LoopbackTransport`` implements :class:`SandboxTransport` in-process
over a :class:`~weakincentives.filesystem.HostBackend` and a
:class:`~weakincentives.sandbox.LocalShell` rooted at one host
directory. It exists to validate the remote facets against the full
filesystem and shell contracts — every fault translation round-trips
through real code — and doubles as a working transport for environments
that happen to be local.

Egress enforcement is the job of a sidecar the *environment* owns (see
``configure_egress`` on :class:`SandboxTransport`); a local environment
has no such sidecar, so ``configure_egress`` and ``configure_credentials``
here only record what they were given (observable via :attr:`egress` and
:attr:`credential_names`), mirroring the local sandbox's documented
bookkeeping-only semantics.
"""

from __future__ import annotations

import os
import shutil
import threading
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

from ..clock import SYSTEM_CLOCK, MonotonicClock
from ..filesystem import (
    FileEntry,
    FileStat,
    GlobMatch,
    GrepMatch,
    HostBackend,
    SnapshotRef,
    WriteMode,
)
from ._config import EgressPolicy
from ._sandbox import CredentialBinding, validate_bindings
from ._shell import MAX_COMMAND_OUTPUT_BYTES, CommandResult, LocalShell
from ._transport import TransportFault, fault_for_exception

__all__ = ["LoopbackTransport"]


class LoopbackTransport:
    """In-process :class:`SandboxTransport` over a host directory.

    Native exceptions from the underlying backend and shell are
    converted to :class:`TransportFault` so clients exercise the same
    fault path a wire transport produces. After :meth:`close`, every
    method raises a ``connectivity`` fault.

    Args:
        root: Host directory acting as the remote environment root.
        owns_root: When True, :meth:`close` removes the root directory.
        max_output_bytes: Per-stream output cap for executed commands.
        git_dir: Snapshot storage override for the host backend.
        clock: Monotonic clock for command durations.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        owns_root: bool = False,
        max_output_bytes: int = MAX_COMMAND_OUTPUT_BYTES,
        git_dir: str | None = None,
        clock: MonotonicClock = SYSTEM_CLOCK,
    ) -> None:
        super().__init__()
        self._root = Path(root).resolve()
        self._owns_root = owns_root
        self._backend = HostBackend(self._root, git_dir=git_dir)
        self._shell = LocalShell(
            self._root, max_output_bytes=max_output_bytes, clock=clock
        )
        self._egress = EgressPolicy()
        self._credential_names: frozenset[str] = frozenset()
        self._closed = False
        self._close_lock = threading.Lock()

    @property
    def root(self) -> str:
        """Environment root directory (resolved)."""
        return str(self._root)

    @property
    def egress(self) -> EgressPolicy:
        """Last egress policy applied through the transport."""
        return self._egress

    @property
    def credential_names(self) -> frozenset[str]:
        """Names of credentials applied through the transport."""
        return self._credential_names

    @property
    def closed(self) -> bool:
        """True once :meth:`close` has run."""
        return self._closed

    @contextmanager
    def _faults(self) -> Generator[None]:
        """Reject closed transports; convert native errors to faults."""
        if self._closed:
            msg = f"loopback transport to {self._root} is closed"
            raise TransportFault("connectivity", msg)
        try:
            yield
        except Exception as error:
            raise fault_for_exception(error) from error

    def stat(self, path: str) -> FileStat:
        """Get metadata for a path."""
        with self._faults():
            return self._backend.stat(path)

    def list(self, path: str) -> Sequence[FileEntry]:
        """List entries of an existing directory, sorted by name."""
        with self._faults():
            return self._backend.list(path)

    def read_range(self, path: str, *, offset: int, length: int | None) -> bytes:
        """Read up to ``length`` bytes starting at ``offset``."""
        with self._faults():
            return self._backend.read_range(path, offset=offset, length=length)

    def write(self, path: str, data: bytes, *, mode: WriteMode) -> None:
        """Write bytes to a file, creating parent directories as needed."""
        with self._faults():
            self._backend.write(path, data, mode=mode)

    def glob(self, pattern: str, *, path: str) -> Sequence[GlobMatch]:
        """Match files under a base directory (in-process is server-side)."""
        with self._faults():
            return self._backend.glob(pattern, path=path)

    def grep(
        self,
        pattern: str,
        *,
        path: str,
        glob: str | None,
        max_matches: int,
    ) -> Sequence[GrepMatch]:
        """Regex-search file contents (in-process is server-side)."""
        with self._faults():
            return self._backend.grep(
                pattern, path=path, glob=glob, max_matches=max_matches
            )

    def delete(self, path: str, *, recursive: bool) -> None:
        """Delete a file, or a directory when empty or ``recursive=True``."""
        with self._faults():
            self._backend.delete(path, recursive=recursive)

    def mkdir(self, path: str) -> None:
        """Create a directory and any missing parents (idempotent)."""
        with self._faults():
            self._backend.mkdir(path)

    def rename(self, src: str, dst: str) -> None:
        """Move a file or directory (with its contents) to a new path."""
        with self._faults():
            self._backend.rename(src, dst)

    def snapshot(self, *, tag: str | None) -> SnapshotRef:
        """Capture environment state via the host backend."""
        with self._faults():
            return self._backend.snapshot(tag=tag)

    def restore(self, ref: SnapshotRef) -> None:
        """Restore environment state captured by :meth:`snapshot`."""
        with self._faults():
            self._backend.restore(ref)

    def exec(
        self,
        argv: Sequence[str],
        *,
        cwd: str | None,
        env: Mapping[str, str] | None,
        stdin: bytes | None,
        timeout_s: float,
    ) -> CommandResult:
        """Execute ``argv`` through the in-process shell."""
        with self._faults():
            return self._shell.run(
                argv, cwd=cwd, env=env, stdin=stdin, timeout_s=timeout_s
            )

    def configure_egress(self, policy: EgressPolicy) -> None:
        """Record the egress policy (a local environment has no sidecar)."""
        with self._faults():
            self._egress = policy

    def configure_credentials(self, bindings: Sequence[CredentialBinding]) -> None:
        """Record credential names; material is held in process memory only."""
        with self._faults():
            self._credential_names = frozenset(validate_bindings(bindings))

    def close(self) -> None:
        """Drop the in-process backend and shell. Idempotent."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self._credential_names = frozenset()
        self._backend.cleanup()
        if self._owns_root and self._root.exists():
            shutil.rmtree(self._root, ignore_errors=True)
