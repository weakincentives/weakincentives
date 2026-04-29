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

"""Extension protocols for the WINK spine.

The spine itself ships no concrete implementations of provider integrations,
deadline clocks, or resource registries. Extras plug in by satisfying the
narrow protocols defined here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Protocol, runtime_checkable

__all__ = [
    "Budget",
    "Deadline",
    "EventListener",
    "PromptResponse",
    "ProviderAdapter",
    "ResourceProvider",
    "ToolCall",
    "Usage",
]


@dataclass(frozen=True, slots=True)
class Usage:
    """Token (or generic resource) accounting for a single provider call."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


def _empty_arguments() -> Mapping[str, object]:
    return {}


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A request from the model to invoke a named tool."""

    name: str
    arguments: Mapping[str, object] = field(default_factory=_empty_arguments)
    call_id: str | None = None


@dataclass(frozen=True, slots=True)
class PromptResponse:
    """Result of a single provider evaluation.

    Adapters return this from :meth:`ProviderAdapter.evaluate`. It is the
    only response shape the spine knows about.
    """

    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None


@runtime_checkable
class Deadline(Protocol):
    """Wall-clock cutoff a tool or adapter can poll."""

    def remaining(self) -> timedelta: ...

    def expired(self) -> bool: ...


@runtime_checkable
class Budget(Protocol):
    """Cumulative usage tracker that fails closed when exhausted."""

    def record(self, usage: Usage) -> None: ...

    def check(self) -> None: ...

    @property
    def consumed(self) -> Usage: ...


@runtime_checkable
class EventListener(Protocol):
    """Observer for session events.

    The spine emits :class:`weakincentives.core.session.PromptRendered`,
    :class:`weakincentives.core.session.ToolInvoked`, and
    :class:`weakincentives.core.session.PromptExecuted`. Extras subscribe to
    build transcripts, debug bundles, or evaluation traces.
    """

    def on_event(self, event: object) -> None: ...


@runtime_checkable
class ResourceProvider(Protocol):
    """Looks up shared resources by protocol type.

    The spine ships no DI container; tools call
    ``context.get_resource(MyProtocol)`` and the wired-in provider returns
    something satisfying ``MyProtocol`` (or raises ``LookupError``). Extras
    provide richer scoped/transactional implementations.
    """

    def get[T](self, protocol_type: type[T]) -> T: ...


@runtime_checkable
class ProviderAdapter(Protocol):
    """Single entry point through which extras integrate models.

    Implementations live in extras (`weakincentives.adapters.*`). The spine
    only ever talks to this protocol.
    """

    def evaluate(
        self,
        rendered: object,
        session: object,
        *,
        deadline: Deadline | None = None,
    ) -> PromptResponse: ...
