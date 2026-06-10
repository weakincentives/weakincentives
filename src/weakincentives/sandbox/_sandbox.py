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

"""Sandbox aggregate: one environment, one owner.

A :class:`Sandbox` names the environment an agent acts on. It vends the
two effect facets — :class:`~weakincentives.filesystem.Filesystem` and
:class:`~weakincentives.sandbox.Shell` — and owns egress and lifecycle:
whoever opens a sandbox closes it; facets never self-close.

The egress/credential surface is **control-plane only**: it is driven by
the harness and policies and must never be reachable from model-visible
tool contexts — an agent cannot widen its own egress or read a bound
secret. Secret material lives outside config and serialized state;
:class:`CredentialBinding` redacts it from ``repr`` and it is never
logged.

Locally there is no egress proxy to reconfigure: ``configure_egress``
updates the policy the sandbox reports and ``configure_credentials`` holds
bindings in process memory — a documented no-op beyond bookkeeping.
Enforcing proxies arrive with the remote sandbox (M4).
"""

from __future__ import annotations

import shutil
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, override, runtime_checkable
from uuid import uuid4

from ..errors import WinkError
from ..filesystem import Filesystem, HostBackend, SnapshotRef
from ._config import EgressPolicy
from ._shell import Shell

__all__ = [
    "CredentialBinding",
    "LocalSandbox",
    "Sandbox",
    "SandboxClosedError",
    "SandboxError",
]


class SandboxError(WinkError, RuntimeError):
    """Base class for sandbox lifecycle and control-plane errors."""


class SandboxClosedError(SandboxError):
    """Raised when using a sandbox after :meth:`Sandbox.close`."""


@dataclass(slots=True, frozen=True)
class CredentialBinding:
    """Binds a credential name to secret material, at runtime only.

    The secret is excluded from ``repr`` and must never be serialized,
    logged, or written into a sandbox's environment or filesystem. Configs
    reference credentials by name
    (:class:`~weakincentives.sandbox.CredentialInjection`); bindings supply
    the material to the control plane.
    """

    name: str
    secret: str = field(repr=False)

    def __post_init__(self) -> None:
        if not self.name:
            msg = "credential name must not be empty"
            raise ValueError(msg)
        if not self.secret:
            msg = f"credential {self.name!r} must bind non-empty secret material"
            raise ValueError(msg)


@runtime_checkable
class Sandbox(Protocol):
    """Aggregate vending the environment's effect facets.

    ``snapshot``/``restore`` delegate to the filesystem backend so the
    whole environment can participate in transactional rollback. ``close``
    is idempotent and is the only teardown: facets never self-close.
    """

    @property
    def id(self) -> str:
        """Stable identifier for this sandbox instance."""
        ...

    @property
    def root(self) -> str:
        """Environment root; equals ``filesystem.root``."""
        ...

    @property
    def filesystem(self) -> Filesystem:
        """Filesystem facet rooted at :attr:`root`."""
        ...

    @property
    def shell(self) -> Shell:
        """Command-execution facet rooted at :attr:`root`."""
        ...

    @property
    def egress(self) -> EgressPolicy:
        """Outbound network policy currently in effect."""
        ...

    @property
    def credential_names(self) -> frozenset[str]:
        """Names of currently bound credentials (never the material)."""
        ...

    def configure_egress(self, policy: EgressPolicy) -> None:
        """Replace the egress policy live (control plane only)."""
        ...

    def configure_credentials(self, bindings: Sequence[CredentialBinding]) -> None:
        """Replace the full set of credential bindings (control plane only)."""
        ...

    def snapshot(self, *, tag: str | None = None) -> SnapshotRef:
        """Capture the environment state via the filesystem backend."""
        ...

    def restore(self, ref: SnapshotRef) -> None:
        """Restore environment state captured by :meth:`snapshot`."""
        ...

    def close(self) -> None:
        """Tear down the environment. Idempotent."""
        ...


class LocalSandbox:
    """Sandbox over a host directory: ``Filesystem(HostBackend)`` + shell.

    Owns its root directory: ``close()`` removes it (and the filesystem
    backend's snapshot storage) and is idempotent. Construct via
    :class:`~weakincentives.sandbox.LocalSandboxProvider` — or directly in
    tests, handing the sandbox ownership of ``root``.

    After ``close()``, facet access and control-plane calls raise
    :class:`SandboxClosedError`; identity and policy stay readable.
    """

    def __init__(
        self,
        *,
        root: str | Path,
        filesystem: Filesystem,
        shell: Shell,
        egress: EgressPolicy | None = None,
        sandbox_id: str | None = None,
    ) -> None:
        super().__init__()
        self._id = sandbox_id if sandbox_id is not None else uuid4().hex
        self._root = Path(root).resolve()
        self._filesystem = filesystem
        self._shell = shell
        self._egress = egress if egress is not None else EgressPolicy()
        self._credentials: dict[str, str] = {}
        self._closed = False
        self._close_lock = threading.Lock()

    @override
    def __repr__(self) -> str:
        return (
            f"LocalSandbox(id={self._id!r}, root={str(self._root)!r}, "
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
        """Environment root (resolved); equals ``filesystem.root``."""
        return str(self._root)

    @property
    def filesystem(self) -> Filesystem:
        """Filesystem facet rooted at :attr:`root`."""
        self._ensure_open()
        return self._filesystem

    @property
    def shell(self) -> Shell:
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
        return frozenset(self._credentials)

    def configure_egress(self, policy: EgressPolicy) -> None:
        """Replace the egress policy.

        Locally there is no proxy to reconfigure; the new policy is
        recorded and reported via :attr:`egress`.
        """
        self._ensure_open()
        self._egress = policy

    def configure_credentials(self, bindings: Sequence[CredentialBinding]) -> None:
        """Replace the full set of credential bindings.

        Material is held in process memory only — never written into the
        sandbox environment or filesystem, never logged.

        Raises:
            ValueError: Two bindings share a name.
        """
        self._ensure_open()
        replacement: dict[str, str] = {}
        for binding in bindings:
            if binding.name in replacement:
                msg = f"duplicate credential binding: {binding.name!r}"
                raise ValueError(msg)
            replacement[binding.name] = binding.secret
        self._credentials = replacement

    def snapshot(self, *, tag: str | None = None) -> SnapshotRef:
        """Capture the environment state via the filesystem backend."""
        self._ensure_open()
        return self._filesystem.snapshot(tag=tag)

    def restore(self, ref: SnapshotRef) -> None:
        """Restore environment state captured by :meth:`snapshot`."""
        self._ensure_open()
        self._filesystem.restore(ref)

    def close(self) -> None:
        """Remove the root directory and snapshot storage. Idempotent."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self._credentials = {}
        if self._root.exists():
            shutil.rmtree(self._root, ignore_errors=True)
        backend = self._filesystem.backend
        if isinstance(backend, HostBackend):
            backend.cleanup()
