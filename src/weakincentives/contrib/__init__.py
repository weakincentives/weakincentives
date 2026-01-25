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

"""Contributed utilities for building domain-specific agents.

The ``contrib`` package provides optional, domain-specific extensions to the
core weakincentives library. These components are useful "batteries included"
tools for building agents but are not part of the minimal core. They may
require additional dependencies (e.g., ``redis``, ``podman``, ``asteval``).

Subpackages
-----------

contrib.mailbox
    Redis-backed message queue implementation with SQS-compatible visibility
    timeout semantics. Supports both standalone Redis and Redis Cluster
    deployments with durable, atomic operations via Lua scripts.

    Key exports:
        - ``RedisMailbox``: Durable message queue with visibility timeouts
        - ``RedisMailboxFactory``: Factory for creating mailboxes on shared clients
        - ``DEFAULT_TTL_SECONDS``: Default key expiration (3 days)

    Requires: ``redis`` package (``pip install weakincentives[redis]``)

contrib.optimizers
    Prompt optimization utilities that generate task-agnostic workspace summaries
    and digests. These optimizers analyze workspaces and cache results for
    efficient reuse across prompt evaluations.

    Key exports:
        - ``WorkspaceDigestOptimizer``: Generates workspace digests via LLM exploration

contrib.tools
    Domain-specific tool suites for LLM agents, providing filesystem operations,
    sandboxed code execution, containerized shell access, and planning capabilities.

    Tool categories:
        - **Planning**: Session-scoped todo lists with ``PlanningToolsSection``
        - **Virtual Filesystem**: File operations via ``VfsToolsSection``
        - **Python Evaluation**: Sandboxed execution via ``AstevalSection``
        - **Container Execution**: Podman-backed shell via ``PodmanSandboxSection``
        - **Workspace Digest**: Caching layer via ``WorkspaceDigestSection``

    Requires: Optional extras for specific features:
        - ``pip install weakincentives[asteval]`` for Python evaluation
        - ``pip install weakincentives[podman]`` for container execution

Example Usage
-------------

Basic tool section setup::

    from weakincentives.contrib.tools import (
        PlanningToolsSection,
        VfsToolsSection,
        AstevalSection,
        PodmanSandboxSection,
        HostMount,
    )
    from weakincentives.runtime.session import Session

    session = Session()

    # Planning tools for multi-step workflows
    planning = PlanningToolsSection(session=session)

    # Virtual filesystem with host mounts
    vfs = VfsToolsSection(
        session=session,
        mounts=(HostMount(host_path="src"),),
        allowed_host_roots=("/home/user/project",),
    )

    # Sandboxed Python evaluation
    asteval = AstevalSection(session=session)

Redis mailbox for inter-agent communication::

    from redis import Redis
    from weakincentives.contrib.mailbox import RedisMailbox

    client = Redis(host="localhost", port=6379)
    requests: RedisMailbox[MyEvent, MyResult] = RedisMailbox(
        name="requests",
        client=client,
    )

    # Send messages with reply routing
    requests.send(MyEvent(data="hello"), reply_to=responses)

    # Receive with visibility timeout
    for msg in requests.receive(visibility_timeout=60):
        result = process(msg.body)
        msg.reply(result)
        msg.acknowledge()

Workspace optimization::

    from weakincentives.contrib.optimizers import WorkspaceDigestOptimizer
    from weakincentives.optimizers.context import OptimizationContext

    optimizer = WorkspaceDigestOptimizer(context)
    result = optimizer.optimize(prompt, session=session)
    # Digest is now cached in session for future prompt renders

Architecture Notes
------------------

All contrib modules follow the core library's design patterns:

- **Immutable dataclasses**: Use ``@FrozenDataclass()`` for state objects
- **Session integration**: Tool sections bind to ``Session`` for state management
- **Resource injection**: Filesystems and dependencies use the resource protocol
- **Design-by-contract**: Public APIs use ``@require``/``@ensure`` decorators

Lazy Loading
------------

Submodules are lazily imported to avoid circular dependencies and reduce
startup time. Access them via attribute access or explicit import::

    from weakincentives.contrib import mailbox, tools, optimizers
    # or
    from weakincentives.contrib.mailbox import RedisMailbox
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

    from . import mailbox, optimizers, tools

__all__ = ["mailbox", "optimizers", "tools"]


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})


def __getattr__(name: str) -> ModuleType:
    """Lazy import submodules to avoid circular dependency issues.

    The optimizers submodule imports from tools, so eager import would fail
    if done in the wrong order. Lazy loading sidesteps this entirely.
    """
    if name in __all__:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
