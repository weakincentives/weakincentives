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
require additional dependencies (e.g., ``redis``).

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

contrib.tools
    Utilities for LLM agents, including workspace digest caching and
    in-memory filesystem for testing.

    Key exports:
        - ``WorkspaceDigestSection``: Renders cached workspace digests
        - ``InMemoryFilesystem``: In-memory filesystem implementation

contrib.optimizers
    Prompt optimization workflows using the Claude Agent SDK.

    Key exports:
        - ``WorkspaceDigestOptimizer``: Generates workspace digests
        - ``WorkspaceDigestResult``: Result of digest optimization

Example Usage
-------------

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

Workspace digest caching::

    from weakincentives.contrib.tools import (
        WorkspaceDigestSection,
        set_workspace_digest,
    )
    from weakincentives.runtime.session import Session

    session = Session()
    digest_section = WorkspaceDigestSection(session=session)

    set_workspace_digest(
        session,
        section_key="workspace-digest",
        body="Full project analysis...",
        summary="Python web app with FastAPI backend.",
    )

Architecture Notes
------------------

All contrib modules follow the core library's design patterns:

- **Immutable dataclasses**: Use ``@FrozenDataclass()`` for state objects
- **Session integration**: Sections bind to ``Session`` for state management
- **Resource injection**: Dependencies use the resource protocol
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
