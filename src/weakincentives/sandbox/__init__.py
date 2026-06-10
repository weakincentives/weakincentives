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

"""Sandbox: the named environment agents act on.

One environment, one truth. A :class:`Sandbox` aggregate vends the two
effect facets — :class:`~weakincentives.filesystem.Filesystem` and
:class:`Shell` — and owns egress and lifecycle. A :class:`SandboxConfig`
declares intent as a serde value; a :class:`SandboxProvider` materializes
it. Local first: remote sandboxes differ only in transport.

Roles
-----

:class:`Shell`
    The exec facet. Runs an ``argv`` vector — no shell interpretation.
    :class:`LocalShell` runs subprocesses rooted at a directory with
    output caps and a default timeout.

:class:`Sandbox` / :class:`LocalSandbox`
    The aggregate: ``id``, ``root``, ``filesystem``, ``shell``, ``egress``,
    snapshot/restore delegating to the filesystem backend, and idempotent
    ``close``. Whoever opens a sandbox closes it; facets never self-close.

:class:`SandboxConfig` / :class:`SandboxProvider`
    Declarative intent (mounts, read-only posture, env, setup commands,
    egress policy) and the factory that materializes it.
    :class:`LocalSandboxProvider` absorbs the workspace mount machinery:
    allowed-root validation, symlink and byte-budget guards.

Egress and credentials (control plane)
--------------------------------------

:class:`EgressPolicy` is default-deny; :class:`EgressRule` allow entries
may name a credential via :class:`CredentialInjection` — names only, never
material. ``configure_egress`` and ``configure_credentials`` are driven by
the harness and policies, never exposed to the model: an agent cannot
widen its own egress or read a bound secret. Locally there is no proxy;
the local sandbox records policy and holds bindings in process memory
only (a documented no-op beyond bookkeeping — enforcing proxies arrive
with the remote sandbox).

Usage
-----

Open, use, close::

    from weakincentives.sandbox import (
        HostMount, LocalSandboxProvider, SandboxConfig,
    )

    provider = LocalSandboxProvider()
    sandbox = provider.open(
        SandboxConfig(mounts=(HostMount(host_path="/path/to/project"),))
    )
    try:
        sandbox.filesystem.write("notes.md", "hello")
        result = sandbox.shell.run(["ls", "-la"])
        ref = sandbox.snapshot(tag="before-build")
        sandbox.restore(ref)
    finally:
        sandbox.close()
"""

from __future__ import annotations

from ._config import (
    CredentialInjection,
    EgressPolicy,
    EgressRule,
    SandboxConfig,
)
from ._mounts import (
    HostMount,
    HostMountPreview,
    WorkspaceBudgetExceededError,
    WorkspaceSecurityError,
    compute_workspace_fingerprint,
    materialize_mounts,
)
from ._provider import (
    LocalSandboxProvider,
    SandboxProvider,
    SandboxSetupError,
)
from ._sandbox import (
    CredentialBinding,
    LocalSandbox,
    Sandbox,
    SandboxClosedError,
    SandboxError,
)
from ._shell import (
    DEFAULT_SHELL_TIMEOUT_S,
    MAX_COMMAND_OUTPUT_BYTES,
    CommandResult,
    LocalShell,
    Shell,
)

__all__ = [
    "DEFAULT_SHELL_TIMEOUT_S",
    "MAX_COMMAND_OUTPUT_BYTES",
    "CommandResult",
    "CredentialBinding",
    "CredentialInjection",
    "EgressPolicy",
    "EgressRule",
    "HostMount",
    "HostMountPreview",
    "LocalSandbox",
    "LocalSandboxProvider",
    "LocalShell",
    "Sandbox",
    "SandboxClosedError",
    "SandboxConfig",
    "SandboxError",
    "SandboxProvider",
    "SandboxSetupError",
    "Shell",
    "WorkspaceBudgetExceededError",
    "WorkspaceSecurityError",
    "compute_workspace_fingerprint",
    "materialize_mounts",
]
